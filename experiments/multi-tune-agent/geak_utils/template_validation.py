"""Static validation for untrusted, generated GEAK task templates.

The validator deliberately does not import or execute any template content.
It combines filesystem checks, data-file validation, and conservative Python
AST heuristics.  The checks are intended as a generation-time safety gate, not
as a replacement for sandboxing or review.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml


REQUIRED_FILES = (
    "kernel.py",
    "config.yaml",
    "scripts/task_runner.py",
    "metadata.json",
)
COMMAND_MODES = {
    "compile_command": "compile",
    "correctness_command": "correctness",
    "performance_command": "performance",
}
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".hip",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".o",
    ".a",
    ".py",
    ".pyc",
    ".pyi",
    ".rs",
    ".sh",
    ".so",
    ".ts",
}
ALLOWED_SOURCE_FILES = {"kernel.py", "scripts/task_runner.py"}
SENSITIVE_TEMPLATE_FILES = {
    "kernel.py",
    "config.yaml",
    "metadata.json",
    "scripts/task_runner.py",
}
_PERF_RE = re.compile(
    r"Perf:\s*(?:%[-+0-9.]*[a-zA-Z]|\{[^}]+\}|\d+(?:\.\d+)?)"
    r"\s+ms\s+\((?:%[-+0-9.]*[a-zA-Z]|\{[^}]+\}|[^)]+)\)"
)


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic validation finding."""

    code: str
    message: str
    path: Optional[str] = None
    line: Optional[int] = None
    severity: str = "error"

    def __str__(self) -> str:
        location = self.path or "<template>"
        if self.line is not None:
            location += ":%d" % self.line
        return "%s: %s [%s]" % (location, self.message, self.code)


@dataclass(frozen=True)
class ValidationReport:
    """Result returned by :func:`validate_generated_template`."""

    root: Path
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def ok(self) -> bool:
        """Alias useful to callers that treat validation as a predicate."""

        return self.valid

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def __bool__(self) -> bool:
        return self.valid


class _Issues:
    def __init__(self) -> None:
        self.items: list[ValidationIssue] = []

    def add(
        self,
        code: str,
        message: str,
        path: Optional[str] = None,
        line: Optional[int] = None,
        severity: str = "error",
    ) -> None:
        self.items.append(ValidationIssue(code, message, path, line, severity))


def validate_generated_template(
    path: os.PathLike[str] | str,
    expected_contract: Optional[Mapping[str, Any]] = None,
) -> ValidationReport:
    """Validate a generated GEAK template directory without executing it.

    ``expected_contract`` is an optional recursive subset that must match
    ``metadata.json``.  It is useful for pinning request-specific details such
    as the operator, format, architecture, dtypes, shapes, or provenance.
    Unknown metadata fields are allowed.
    """

    supplied_root = Path(path).expanduser()
    issues = _Issues()
    if supplied_root.is_symlink():
        issues.add("root-symlink", "template root must not be a symlink")
    try:
        root = supplied_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        root = supplied_root.absolute()
        issues.add("invalid-root", "template directory cannot be resolved: %s" % exc)
        return ValidationReport(root, tuple(issues.items))
    if not root.is_dir():
        issues.add("invalid-root", "template path must be a directory")
        return ValidationReport(root, tuple(issues.items))

    _validate_tree(root, issues)
    for relative in REQUIRED_FILES:
        candidate = root / relative
        if not candidate.is_file() or candidate.is_symlink():
            issues.add(
                "missing-required-file",
                "required regular file is missing",
                relative,
            )

    config = _load_yaml_mapping(root, "config.yaml", issues)
    metadata = _load_json_mapping(root, "metadata.json", issues)
    kernel_tree, _ = _parse_python(root, "kernel.py", issues)
    runner_tree, runner_text = _parse_python(
        root, "scripts/task_runner.py", issues
    )

    if config is not None:
        _validate_config(config, kernel_tree, issues)
    if metadata is not None:
        _validate_metadata(metadata, expected_contract, issues)
    elif expected_contract is not None and not isinstance(expected_contract, Mapping):
        issues.add(
            "invalid-expected-contract",
            "expected_contract must be a mapping when supplied",
        )
    if runner_tree is not None:
        _validate_runner(runner_tree, runner_text, metadata or {}, issues)

    ordered = sorted(
        issues.items,
        key=lambda issue: (
            issue.path or "",
            issue.line or 0,
            issue.code,
            issue.message,
        ),
    )
    return ValidationReport(root, tuple(ordered))


def _validate_tree(root: Path, issues: _Issues) -> None:
    try:
        entries = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        issues.add("tree-read-error", "could not enumerate template: %s" % exc)
        return
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            issues.add("symlink", "symlinks are forbidden", relative)
            continue
        try:
            resolved = entry.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            issues.add(
                "path-escape",
                "resolved path must remain inside the template root",
                relative,
            )
            continue
        if not entry.is_file():
            continue
        if entry.suffix.lower() in SOURCE_SUFFIXES and relative not in ALLOWED_SOURCE_FILES:
            issues.add(
                "unexpected-source-file",
                "unexpected executable or source file",
                relative,
            )
        try:
            if entry.stat().st_mode & 0o111 and relative != "scripts/task_runner.py":
                issues.add(
                    "unexpected-executable",
                    "only scripts/task_runner.py may be executable",
                    relative,
                )
        except OSError as exc:
            issues.add("file-stat-error", "could not inspect file: %s" % exc, relative)


def _read_text(root: Path, relative: str, issues: _Issues) -> Optional[str]:
    candidate = root / relative
    if not candidate.is_file() or candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        issues.add("invalid-utf8", "file must be UTF-8 text", relative)
    except (OSError, RuntimeError, ValueError) as exc:
        issues.add("file-read-error", "could not safely read file: %s" % exc, relative)
    return None


def _load_yaml_mapping(
    root: Path, relative: str, issues: _Issues
) -> Optional[dict[str, Any]]:
    text = _read_text(root, relative, issues)
    if text is None:
        return None
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        issues.add("invalid-yaml", "invalid YAML: %s" % exc, relative)
        return None
    if not isinstance(value, dict):
        issues.add("invalid-config", "YAML root must be a mapping", relative)
        return None
    return value


def _load_json_mapping(
    root: Path, relative: str, issues: _Issues
) -> Optional[dict[str, Any]]:
    text = _read_text(root, relative, issues)
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        issues.add(
            "invalid-json",
            "invalid JSON: %s" % exc.msg,
            relative,
            exc.lineno,
        )
        return None
    if not isinstance(value, dict):
        issues.add("invalid-metadata", "JSON root must be an object", relative)
        return None
    return value


def _parse_python(
    root: Path, relative: str, issues: _Issues
) -> tuple[Optional[ast.Module], str]:
    text = _read_text(root, relative, issues)
    if text is None:
        return None, ""
    try:
        return ast.parse(text, filename=relative), text
    except SyntaxError as exc:
        issues.add(
            "invalid-python",
            "Python syntax error: %s" % exc.msg,
            relative,
            exc.lineno,
        )
        return None, text


def _validate_config(
    config: Mapping[str, Any],
    kernel_tree: Optional[ast.Module],
    issues: _Issues,
) -> None:
    relative = "config.yaml"
    if config.get("source_file_path") != ["kernel.py"]:
        issues.add(
            "unsafe-source-paths",
            "source_file_path must be exactly ['kernel.py']",
            relative,
        )
    targets = config.get("target_kernel_functions")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(name, str) or not name.strip() for name in targets)
    ):
        issues.add(
            "invalid-target-functions",
            "target_kernel_functions must be a non-empty list of names",
            relative,
        )
    elif len(set(targets)) != len(targets):
        issues.add(
            "invalid-target-functions",
            "target_kernel_functions must not contain duplicates",
            relative,
        )
    elif kernel_tree is not None:
        definitions = {
            node.name
            for node in kernel_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = sorted(set(targets) - definitions)
        if missing:
            issues.add(
                "missing-target-function",
                "target function(s) not defined in kernel.py: %s"
                % ", ".join(missing),
                relative,
            )

    for key, mode in COMMAND_MODES.items():
        raw = config.get(key)
        commands = raw if isinstance(raw, list) else None
        if (
            commands is None
            or len(commands) != 1
            or not isinstance(commands[0], str)
            or not commands[0].strip()
        ):
            issues.add(
                "invalid-command",
                "%s must contain exactly one command" % key,
                relative,
            )
            continue
        try:
            tokens = shlex.split(commands[0])
        except ValueError as exc:
            issues.add(
                "invalid-command",
                "%s is not shell-parseable: %s" % (key, exc),
                relative,
            )
            continue
        runner_positions = [
            index
            for index, token in enumerate(tokens)
            if token.replace("\\", "/").endswith("scripts/task_runner.py")
        ]
        if not runner_positions or not any(
            index + 1 < len(tokens) and tokens[index + 1] == mode
            for index in runner_positions
        ):
            issues.add(
                "invalid-command-mode",
                "%s must invoke scripts/task_runner.py with mode %s" % (key, mode),
                relative,
            )


def _validate_metadata(
    metadata: Mapping[str, Any],
    expected: Optional[Mapping[str, Any]],
    issues: _Issues,
) -> None:
    relative = "metadata.json"
    scalar_fields = ("name", "operator", "format")
    for field in scalar_fields:
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            issues.add(
                "incomplete-metadata",
                "metadata requires non-empty %r" % field,
                relative,
            )
    arches = metadata.get("supported_arches", metadata.get("supported_architectures"))
    if (
        not isinstance(arches, list)
        or not arches
        or any(not isinstance(arch, str) or not arch.strip() for arch in arches)
    ):
        issues.add(
            "incomplete-metadata",
            "metadata requires non-empty supported_arches",
            relative,
        )

    contract = metadata.get("contract", metadata.get("operand_contract"))
    if not isinstance(contract, dict) or not contract:
        issues.add(
            "incomplete-contract",
            "metadata requires a non-empty contract or operand_contract object",
            relative,
        )
    else:
        lowered = {str(key).lower() for key in contract}
        if not lowered.intersection({"input", "inputs", "operands", "activation"}):
            issues.add(
                "incomplete-contract",
                "metadata contract must describe inputs or operands",
                relative,
            )
        if not lowered.intersection({"output", "outputs", "result"}):
            issues.add(
                "incomplete-contract",
                "metadata contract must describe output",
                relative,
            )

    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        issues.add(
            "incomplete-provenance",
            "metadata requires a non-empty provenance object",
            relative,
        )
    else:
        provenance_keys = {str(key).lower() for key in provenance}
        if not provenance_keys.intersection(
            {"generator", "generated_by", "model", "producer"}
        ):
            issues.add(
                "incomplete-provenance",
                "provenance must identify the generator or model",
                relative,
            )
        if not provenance_keys.intersection(
            {"source", "source_request", "request_id", "origin"}
        ):
            issues.add(
                "incomplete-provenance",
                "provenance must identify the source request or origin",
                relative,
            )

    if expected is None:
        return
    if not isinstance(expected, Mapping):
        issues.add(
            "invalid-expected-contract",
            "expected_contract must be a mapping when supplied",
        )
        return
    for key, expected_value in sorted(expected.items(), key=lambda pair: str(pair[0])):
        _compare_expected(
            metadata.get(key, _MISSING),
            expected_value,
            str(key),
            issues,
        )


_MISSING = object()


def _compare_expected(
    actual: Any,
    expected: Any,
    dotted_key: str,
    issues: _Issues,
) -> None:
    if actual is _MISSING:
        issues.add(
            "contract-mismatch",
            "metadata is missing expected field %s" % dotted_key,
            "metadata.json",
        )
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            issues.add(
                "contract-mismatch",
                "metadata field %s has the wrong type" % dotted_key,
                "metadata.json",
            )
            return
        for key, value in sorted(expected.items(), key=lambda pair: str(pair[0])):
            _compare_expected(
                actual.get(key, _MISSING),
                value,
                "%s.%s" % (dotted_key, key),
                issues,
            )
    elif actual != expected:
        issues.add(
            "contract-mismatch",
            "metadata field %s does not match the expected contract" % dotted_key,
            "metadata.json",
        )


def _validate_runner(
    tree: ast.Module,
    text: str,
    metadata: Mapping[str, Any],
    issues: _Issues,
) -> None:
    relative = "scripts/task_runner.py"
    constants = _numeric_constants(tree)
    _validate_modes(tree, issues)
    _validate_seed(tree, issues)
    _validate_performance_output(tree, text, issues)
    _validate_reference_independence(tree, constants, issues)
    _validate_exception_handling(tree, issues)
    _validate_sensitive_writes(tree, issues)
    arches = metadata.get("supported_arches", metadata.get("supported_architectures"))
    if isinstance(arches, list) and arches:
        _validate_architecture_gate(tree, issues)

    # Ensure every issue emitted here defaults to the runner path.
    for index, issue in enumerate(issues.items):
        if issue.path is None and issue.code.startswith("runner-"):
            issues.items[index] = ValidationIssue(
                issue.code, issue.message, relative, issue.line, issue.severity
            )


def _validate_modes(tree: ast.Module, issues: _Issues) -> None:
    exact_choices = False
    words: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in COMMAND_MODES.values():
                words.add(node.value)
        if isinstance(node, ast.Call) and _call_name(node.func).endswith("add_argument"):
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(
                    keyword.value, (ast.Tuple, ast.List, ast.Set)
                ):
                    values = {
                        element.value
                        for element in keyword.value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    }
                    exact_choices = values == set(COMMAND_MODES.values())
        if (
            isinstance(node, ast.Compare)
            and any(isinstance(operator, (ast.In, ast.NotIn)) for operator in node.ops)
        ):
            for comparator in node.comparators:
                if isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                    values = {
                        element.value
                        for element in comparator.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    }
                    if values == set(COMMAND_MODES.values()):
                        exact_choices = True
    if words != set(COMMAND_MODES.values()) or not exact_choices:
        issues.add(
            "runner-modes",
            "runner must expose exactly compile, correctness, and performance modes",
            "scripts/task_runner.py",
        )


def _validate_seed(tree: ast.Module, issues: _Issues) -> None:
    seeded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func).endswith("manual_seed"):
            if node.args and _is_deterministic_seed(node.args[0]):
                seeded = True
                break
    if not seeded:
        issues.add(
            "runner-seed",
            "runner must set a fixed deterministic random seed",
            "scripts/task_runner.py",
        )


def _is_deterministic_seed(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id.lower() in {"seed", "index", "case_index"}
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        names = {item.id.lower() for item in ast.walk(node) if isinstance(item, ast.Name)}
        constants = [
            item.value
            for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, int)
        ]
        return bool(constants) and names <= {"index", "case_index"}
    return False


def _validate_performance_output(
    tree: ast.Module, text: str, issues: _Issues
) -> None:
    strings = [_string_shape(node) for node in ast.walk(tree)]
    perf_shapes = [value for value in strings if value and "Perf:" in value]
    if not perf_shapes or not any(
        "ms" in value and "(" in value and ")" in value for value in perf_shapes
    ):
        issues.add(
            "runner-perf-output",
            "performance mode must print parseable 'Perf: <ms> ms (<case_id>)' output",
            "scripts/task_runner.py",
        )
    elif not any(_PERF_RE.search(value) for value in perf_shapes):
        issues.add(
            "runner-perf-output",
            "Perf output format is not statically parseable",
            "scripts/task_runner.py",
        )
    if "performance_report.json" not in text:
        issues.add(
            "runner-performance-report",
            "performance mode must write build/performance_report.json",
            "scripts/task_runner.py",
        )
    required_report_terms = {"test_cases", "test_case_id", "execution_time_ms"}
    literal_values = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if not required_report_terms <= literal_values or not any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"json.dump", "json.dumps"}
        for node in ast.walk(tree)
    ):
        issues.add(
            "runner-performance-report",
            "performance report must JSON-serialize test case ids and latency values",
            "scripts/task_runner.py",
        )


def _string_shape(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) else "{value}"
            for value in node.values
        )
    return None


def _validate_reference_independence(
    tree: ast.Module,
    constants: Mapping[str, float],
    issues: _Issues,
) -> None:
    kernel_symbols: set[str] = set()
    kernel_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (
            node.module == "kernel" or (node.module or "").endswith(".kernel")
        ):
            kernel_symbols.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "kernel" or alias.name.endswith(".kernel"):
                    kernel_modules.add(alias.asname or alias.name.split(".")[-1])

    actual_call_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        targets, value = _assignment_parts(node)
        target_names = {
            item.id.lower()
            for target in targets
            for item in ast.walk(target)
            if isinstance(item, ast.Name)
        }
        if target_names.intersection({"actual", "result", "output", "out"}):
            actual_call_names.update(
                _call_name(item.func)
                for item in ast.walk(value)
                if isinstance(item, ast.Call)
            )

    compare_seen = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets, value = _assignment_parts(node)
            target_names = {
                item.id.lower()
                for target in targets
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
            }
            if target_names.intersection({"expected", "reference", "ref", "golden"}):
                value_call_names = {
                    _call_name(item.func)
                    for item in ast.walk(value)
                    if isinstance(item, ast.Call)
                }
                if _contains_kernel_call(
                    value, kernel_symbols, kernel_modules
                ) or bool(value_call_names & actual_call_names):
                    issues.add(
                        "runner-kernel-reference",
                        "expected/reference output must not come from the kernel under test",
                        "scripts/task_runner.py",
                        getattr(node, "lineno", None),
                    )
                if _depends_on_actual(value):
                    issues.add(
                        "runner-self-reference",
                        "expected/reference output must not derive from actual output",
                        "scripts/task_runner.py",
                        getattr(node, "lineno", None),
                    )

        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name.endswith(("assert_close", "allclose", "isclose")):
            compare_seen = True
            if len(node.args) >= 2 and _expr_key(node.args[0]) == _expr_key(node.args[1]):
                issues.add(
                    "runner-self-comparison",
                    "correctness comparison uses the same expression on both sides",
                    "scripts/task_runner.py",
                    node.lineno,
                )
            arg_names = [
                {
                    item.id.lower()
                    for item in ast.walk(argument)
                    if isinstance(item, ast.Name)
                }
                for argument in node.args[:2]
            ]
            if len(arg_names) == 2 and (
                (arg_names[0] & {"actual", "result", "output", "out"})
                and (arg_names[1] & {"actual", "result", "output", "out"})
                and not (arg_names[1] & {"expected", "reference", "ref", "golden"})
            ):
                issues.add(
                    "runner-self-comparison",
                    "correctness comparison does not use an independent reference",
                    "scripts/task_runner.py",
                    node.lineno,
                )
            _validate_tolerances(node, constants, issues)
    if not compare_seen:
        issues.add(
            "runner-correctness-bypass",
            "runner has no static numerical correctness comparison",
            "scripts/task_runner.py",
        )

    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            "correct" in node.name.lower()
            or node.name.lower() in {"validate_output", "verify_output"}
        )
    ):
        comparison_lines = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _call_name(node.func).endswith(("assert_close", "allclose", "isclose"))
        ]
        if not comparison_lines:
            issues.add(
                "runner-correctness-bypass",
                "correctness function performs no numerical comparison",
                "scripts/task_runner.py",
                function.lineno,
            )
        else:
            first_compare = min(comparison_lines)
            for statement in function.body:
                if isinstance(statement, ast.Return) and statement.lineno < first_compare:
                    issues.add(
                        "runner-correctness-bypass",
                        "correctness function returns before its comparison",
                        "scripts/task_runner.py",
                        statement.lineno,
                    )
                if (
                    isinstance(statement, ast.If)
                    and isinstance(statement.test, ast.Constant)
                    and statement.test.value is False
                    and any(line > statement.lineno for line in comparison_lines)
                ):
                    issues.add(
                        "runner-correctness-bypass",
                        "correctness comparison is guarded by a constant false branch",
                        "scripts/task_runner.py",
                        statement.lineno,
                    )


def _assignment_parts(
    node: ast.Assign | ast.AnnAssign | ast.NamedExpr,
) -> tuple[list[ast.AST], ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets), node.value
    return [node.target], node.value


def _contains_kernel_call(
    node: ast.AST, symbols: set[str], modules: set[str]
) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name) and item.func.id in symbols:
            return True
        if (
            isinstance(item.func, ast.Attribute)
            and isinstance(item.func.value, ast.Name)
            and item.func.value.id in modules
        ):
            return True
    return False


def _depends_on_actual(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Name)
        and item.id.lower() in {"actual", "result", "output", "out"}
        for item in ast.walk(node)
    )


def _expr_key(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _validate_tolerances(
    call: ast.Call, constants: Mapping[str, float], issues: _Issues
) -> None:
    call_name = _call_name(call.func)
    if call_name.endswith(("allclose", "isclose")):
        for index, tolerance_name, limit in ((2, "rtol", 0.1), (3, "atol", 1.0)):
            if len(call.args) <= index:
                continue
            value = _number(call.args[index], constants)
            if value is None or value < 0 or value > limit:
                issues.add(
                    "runner-permissive-tolerance",
                    "positional %s is not a safe static tolerance" % tolerance_name,
                    "scripts/task_runner.py",
                    call.args[index].lineno,
                )
    for keyword in call.keywords:
        if keyword.arg in {"rtol", "atol"}:
            value = _number(keyword.value, constants)
            limit = 0.1 if keyword.arg == "rtol" else 1.0
            if value is None:
                issues.add(
                    "runner-dynamic-tolerance",
                    "%s must be a statically bounded numeric value" % keyword.arg,
                    "scripts/task_runner.py",
                    keyword.value.lineno,
                )
            elif value < 0 or value > limit:
                issues.add(
                    "runner-permissive-tolerance",
                    "%s=%s is overly permissive" % (keyword.arg, value),
                    "scripts/task_runner.py",
                    keyword.value.lineno,
                )
        elif keyword.arg == "equal_nan" and isinstance(
            keyword.value, ast.Constant
        ) and keyword.value.value is True:
            issues.add(
                "runner-permissive-tolerance",
                "equal_nan=True can hide invalid kernel output",
                "scripts/task_runner.py",
                keyword.value.lineno,
            )


def _numeric_constants(tree: ast.Module) -> dict[str, float]:
    values: dict[str, float] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.AST)
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _number(node.value, {})
            if value is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = value
    return values


def _number(node: ast.AST, constants: Mapping[str, float]) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _number(node.operand, constants)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _validate_exception_handling(tree: ast.Module, issues: _Issues) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name)
            and node.type.id in {"Exception", "BaseException"}
        )
        swallowed = not node.body or all(
            isinstance(statement, (ast.Pass, ast.Continue, ast.Break, ast.Return))
            or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and _call_name(statement.value.func) in {"print", "logging.warning"}
            )
            for statement in node.body
        )
        if broad and swallowed:
            issues.add(
                "runner-swallowed-exception",
                "broad exceptions must not be swallowed",
                "scripts/task_runner.py",
                node.lineno,
            )


def _validate_sensitive_writes(tree: ast.Module, issues: _Issues) -> None:
    sensitive_variables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        targets, value = _assignment_parts(node)
        rendered = ast.unparse(value) if hasattr(ast, "unparse") else ""
        if any(
            sensitive in rendered or Path(sensitive).name in rendered
            for sensitive in SENSITIVE_TEMPLATE_FILES
        ):
            sensitive_variables.update(
                item.id
                for target in targets
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        write_like = name.endswith(
            (
                "write",
                "write_text",
                "write_bytes",
                "unlink",
                "rename",
                "replace",
            )
        ) or name in {"open", "os.remove", "os.rename", "shutil.copy", "shutil.move"}
        if not write_like:
            continue
        if name == "open":
            mode: Any = "r"
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            if isinstance(mode, str) and not any(flag in mode for flag in "wax+"):
                continue
        text = ast.unparse(node) if hasattr(ast, "unparse") else ""
        normalized = text.replace("\\", "/")
        mentions_sensitive = any(
            sensitive in normalized or Path(sensitive).name in normalized
            for sensitive in SENSITIVE_TEMPLATE_FILES
        ) or any(
            isinstance(item, ast.Name) and item.id in sensitive_variables
            for item in ast.walk(node)
        )
        if mentions_sensitive:
            issues.add(
                "runner-harness-mutation",
                "runner must not mutate template source or harness files",
                "scripts/task_runner.py",
                node.lineno,
            )


def _validate_architecture_gate(tree: ast.Module, issues: _Issues) -> None:
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(
                name == "aiter"
                or name.startswith("aiter.")
                or name == "kernel"
                or name.endswith(".kernel")
                for name in names
            ):
                issues.add(
                    "runner-early-heavy-import",
                    "kernel/AITER imports must occur only after the architecture gate",
                    "scripts/task_runner.py",
                    node.lineno,
                )

    main = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        ),
        None,
    )
    if main is None:
        main_guard = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.If)
                and "__main__" in ast.unparse(node.test)
            ),
            None,
        )
        if main_guard is None:
            issues.add(
                "runner-architecture-gate",
                "restricted templates require a main entry-point architecture gate",
                "scripts/task_runner.py",
            )
            return
        entry_body = main_guard.body
    else:
        entry_body = main.body
    gate_index: Optional[int] = None
    for index, statement in enumerate(entry_body):
        calls = [item for item in ast.walk(statement) if isinstance(item, ast.Call)]
        if any(_is_gate_call(call) for call in calls):
            gate_index = index
            break
    if gate_index is None:
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        entry_node: ast.AST = main if main is not None else main_guard
        delegated = {
            _call_name(call.func)
            for call in ast.walk(entry_node)
            if isinstance(call, ast.Call) and _call_name(call.func) in functions
        }
        handlers = [
            functions[name]
            for name in sorted(delegated)
            if any(token in name.lower() for token in ("compile", "correct", "performance", "benchmark"))
        ]
        if handlers and all(_function_gates_before_work(handler) for handler in handlers):
            return
        issues.add(
            "runner-architecture-gate",
            "restricted template must reject unsupported architecture before work",
            "scripts/task_runner.py",
            main.lineno if main is not None else main_guard.lineno,
        )
        return
    for statement in entry_body[:gate_index]:
        for call in (item for item in ast.walk(statement) if isinstance(item, ast.Call)):
            name = _call_name(call.func).lower()
            if _is_heavy_call(name):
                issues.add(
                    "runner-late-architecture-gate",
                    "architecture gate must precede kernel imports and tensor allocation",
                    "scripts/task_runner.py",
                    call.lineno,
                )


def _function_gates_before_work(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    gate_index: Optional[int] = None
    for index, statement in enumerate(function.body):
        calls = [item for item in ast.walk(statement) if isinstance(item, ast.Call)]
        if any(_is_gate_call(call) for call in calls):
            gate_index = index
            break
        if any(_is_heavy_call(_call_name(call.func).lower()) for call in calls):
            return False
    return gate_index is not None


def _is_gate_call(call: ast.Call) -> bool:
    name = _call_name(call.func).lower()
    return (
        (
            "require" in name
            or "check" in name
            or "validate" in name
            or "enforce" in name
        )
        and any(token in name for token in ("arch", "gfx", "device", "gpu"))
    )


def _is_heavy_call(name: str) -> bool:
    return (
        name.startswith("torch.")
        and any(token in name for token in ("empty", "zeros", "ones", "rand", "tensor"))
    ) or any(
        token in name
        for token in (
            "import_module",
            "load_kernel",
            "make_case",
            "compile_kernel",
            "check_correctness",
            "benchmark",
        )
    )


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_generated_template",
]
