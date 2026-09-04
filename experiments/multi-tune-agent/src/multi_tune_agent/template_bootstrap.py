"""Safe, direct-LLM-first bootstrap of arbitrary GEAK task templates.

This module only creates and statically validates source bundles.  It never
imports generated code, executes a GPU workload, or registers a task.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import yaml
from geak_utils import KernelSandbox, TaskSpec
from geak_utils.aiter_discovery import AiterCandidate, AiterQuery, discover_aiter
from geak_utils.template_validation import (
    ValidationIssue,
    ValidationReport,
    validate_generated_template,
)

from .agents import extract_json
from .runtime import ModelBackend


_BUNDLE_PATHS = (
    "kernel.py",
    "config.yaml",
    "scripts/task_runner.py",
    "metadata.json",
)
_MAX_FILE_BYTES = 512 * 1024
_MAX_BUNDLE_BYTES = 2 * 1024 * 1024
_MAX_EVIDENCE_FILE_CHARS = 8_000
_MAX_EVIDENCE_CHARS = 24_000
_VALUE_ALIASES = {
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float8": "fp8",
    "int8_t": "int8",
    "blockwise": "per_block",
    "block_wise": "per_block",
    "tensorwise": "per_tensor",
    "tokenwise": "per_token",
    "channelwise": "per_channel",
    "groupwise": "per_group",
}
_ARCH_ALIASES = {
    "mi300": "gfx942",
    "mi300x": "gfx942",
    "mi308": "gfx942",
    "mi308x": "gfx942",
    "mi325": "gfx942",
    "mi325x": "gfx942",
    "mi350": "gfx950",
    "mi350x": "gfx950",
    "mi355": "gfx950",
    "mi355x": "gfx950",
}


def _normalize(value: object | None, *, required: bool = False) -> Optional[str]:
    if value is None:
        if required:
            raise ValueError("required value must be non-empty")
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not normalized:
        if required:
            raise ValueError("required value must be non-empty")
        return None
    return _VALUE_ALIASES.get(normalized, normalized)


def _normalize_text(value: object, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("%s must be non-empty" % name)
    return normalized


def _normalize_shapes(
    shapes: Sequence[int] | Iterable[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    if shapes is None:
        return ()
    try:
        values = tuple(shapes)
    except TypeError as exc:
        raise ValueError("shapes must be a shape or iterable of shapes") from exc
    if not values:
        return ()
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        values = (values,)  # type: ignore[assignment]
    normalized = []
    for shape in values:
        if isinstance(shape, (str, bytes)):
            raise ValueError("shapes must contain integer dimensions")
        try:
            raw_dimensions = tuple(shape)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("shapes must contain sequences of dimensions") from exc
        if not raw_dimensions or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in raw_dimensions
        ):
            raise ValueError("shapes must contain only positive integer dimensions")
        normalized.append(tuple(raw_dimensions))
    return tuple(normalized)


def _normalize_block_size(
    value: int | Sequence[int] | None,
) -> int | tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("block_size must contain positive integers")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("block_size must be positive")
        return value
    if isinstance(value, (str, bytes)):
        raise ValueError("block_size must be an integer or integer sequence")
    values = tuple(value)
    if not values or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in values
    ):
        raise ValueError("block_size must contain positive integers")
    return values


@dataclass(frozen=True)
class KernelContract:
    """Canonical, hashable description of an operator template request."""

    operator: str
    request: str
    target_gpu: str
    architecture: str
    language: str = "python"
    input_dtype: Optional[str] = None
    weight_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    input_format: Optional[str] = None
    weight_format: Optional[str] = None
    input_scale_granularity: Optional[str] = None
    weight_scale_granularity: Optional[str] = None
    block_size: int | tuple[int, ...] | Sequence[int] | None = None
    shapes: tuple[tuple[int, ...], ...] | Sequence[int] | Iterable[Sequence[int]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", _normalize(self.operator, required=True))
        object.__setattr__(self, "request", _normalize_text(self.request, "request"))
        object.__setattr__(
            self, "target_gpu", _normalize(self.target_gpu, required=True)
        )
        architecture = _normalize(self.architecture, required=True)
        architecture = _ARCH_ALIASES.get(architecture.replace("_", ""), architecture)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "language", _normalize(self.language, required=True))
        for field_name in (
            "input_dtype",
            "weight_dtype",
            "output_dtype",
            "input_format",
            "weight_format",
            "input_scale_granularity",
            "weight_scale_granularity",
        ):
            normalized = _normalize(getattr(self, field_name))
            if field_name.endswith("_granularity") and normalized in {
                "tensor",
                "token",
                "channel",
                "group",
                "block",
            }:
                normalized = "per_" + normalized
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(self, "block_size", _normalize_block_size(self.block_size))
        object.__setattr__(self, "shapes", _normalize_shapes(self.shapes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "request": self.request,
            "target_gpu": self.target_gpu,
            "architecture": self.architecture,
            "language": self.language,
            "input_dtype": self.input_dtype,
            "weight_dtype": self.weight_dtype,
            "output_dtype": self.output_dtype,
            "input_format": self.input_format,
            "weight_format": self.weight_format,
            "input_scale_granularity": self.input_scale_granularity,
            "weight_scale_granularity": self.weight_scale_granularity,
            "block_size": (
                list(self.block_size)
                if isinstance(self.block_size, tuple)
                else self.block_size
            ),
            "shapes": [list(shape) for shape in self.shapes],
        }

    @property
    def contract_hash(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def metadata(self) -> dict[str, Any]:
        """Deterministic metadata fields pinned by static validation."""

        values = self.as_dict()
        template_format = self.input_format or self.input_dtype or self.language
        return {
            **values,
            "name": "%s-%s" % (self.operator, self.contract_hash[:12]),
            "format": template_format,
            "contract_hash": self.contract_hash,
            "supported_arches": [self.architecture],
            "contract": {
                "input": {
                    "dtype": self.input_dtype,
                    "format": self.input_format,
                    "scale_granularity": self.input_scale_granularity,
                },
                "weight": {
                    "dtype": self.weight_dtype,
                    "format": self.weight_format,
                    "scale_granularity": self.weight_scale_granularity,
                },
                "output": {"dtype": self.output_dtype},
                "shapes": values["shapes"],
                "block_size": values["block_size"],
                "request": self.request,
            },
        }

    @property
    def expected_contract(self) -> dict[str, Any]:
        return self.metadata


@dataclass(frozen=True)
class TemplateDraft:
    path: Path
    contract: KernelContract
    validation_report: ValidationReport
    generation_method: str
    candidate_summaries: tuple[Mapping[str, Any], ...] = ()

    @property
    def valid(self) -> bool:
        return self.validation_report.valid

    @property
    def contract_hash(self) -> str:
        return self.contract.contract_hash

    @property
    def metadata(self) -> dict[str, Any]:
        return self.contract.metadata


@dataclass(frozen=True)
class TemplateGateResult:
    """Deterministic evidence from the static and locked GPU validation gate."""

    static_valid: bool
    compiled: bool
    correct: bool
    performance_valid: bool
    per_case_ms: Mapping[str, float]
    command_summaries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    validation_workspace: Optional[Path] = None

    def __post_init__(self) -> None:
        normalized_ms = {
            str(name): float(value)
            for name, value in sorted(self.per_case_ms.items(), key=lambda item: str(item[0]))
        }
        object.__setattr__(self, "per_case_ms", MappingProxyType(normalized_ms))
        normalized_commands = {
            str(mode): MappingProxyType(dict(summary))
            for mode, summary in self.command_summaries.items()
        }
        object.__setattr__(
            self, "command_summaries", MappingProxyType(normalized_commands)
        )
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors))
        if self.validation_workspace is not None:
            object.__setattr__(
                self,
                "validation_workspace",
                Path(self.validation_workspace).expanduser().resolve(),
            )

    @property
    def trusted(self) -> bool:
        return (
            self.static_valid
            and self.compiled
            and self.correct
            and self.performance_valid
            and not self.errors
            and bool(self.per_case_ms)
            and all(
                math.isfinite(value) and value > 0
                for value in self.per_case_ms.values()
            )
        )

    @property
    def commands(self) -> Mapping[str, Mapping[str, Any]]:
        """Concise alias for command summaries."""

        return self.command_summaries


class BootstrapError(RuntimeError):
    """Failure to produce a statically valid template."""

    def __init__(
        self,
        message: str,
        *,
        validation_report: Optional[ValidationReport] = None,
        candidate_summaries: Sequence[Mapping[str, Any]] = (),
        draft_path: Optional[Path] = None,
    ) -> None:
        super().__init__(message)
        self.validation_report = validation_report
        self.report = validation_report
        self.candidate_summaries = tuple(dict(item) for item in candidate_summaries)
        self.draft_path = draft_path


class TemplateBootstrapper:
    """Generate an isolated template, using AITER only as repair evidence."""

    def __init__(
        self,
        backend: ModelBackend,
        draft_root: Path | str,
        aiter_root: Path | str | None = None,
        minimum_aiter_score: int = 85,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        if minimum_aiter_score < 0:
            raise ValueError("minimum_aiter_score must be non-negative")
        self.backend = backend
        self.draft_root = Path(draft_root).expanduser().resolve()
        self.aiter_root = (
            Path(aiter_root).expanduser().resolve() if aiter_root is not None else None
        )
        self.minimum_aiter_score = int(minimum_aiter_score)
        self.event_sink = event_sink

    def _event(self, phase: str, **details: Any) -> None:
        if self.event_sink is not None:
            self.event_sink({"phase": phase, **details})

    def generate(self, contract: KernelContract) -> TemplateDraft:
        if not isinstance(contract, KernelContract):
            raise TypeError("contract must be a KernelContract")
        self.draft_root.mkdir(parents=True, exist_ok=True)
        target = self.draft_root / contract.contract_hash
        self._event(
            "start",
            operator=contract.operator,
            contract_hash=contract.contract_hash[:12],
        )
        if target.is_dir() and not target.is_symlink():
            report = validate_generated_template(target, contract.expected_contract)
            if report.valid:
                metadata = json.loads((target / "metadata.json").read_text("utf-8"))
                method = metadata.get("provenance", {}).get(
                    "generation_method", "llm_direct"
                )
                self._event("reuse_valid_draft", method=method)
                return TemplateDraft(target, contract, report, str(method))

        direct_error: Optional[BootstrapError] = None
        try:
            self._event("direct_request")
            direct_bundle = self._request_bundle(self._direct_messages(contract))
            self._install_bundle(target, direct_bundle, contract, "llm_direct", ())
            direct_report = validate_generated_template(
                target, contract.expected_contract
            )
        except BootstrapError as exc:
            direct_error = exc
            direct_report = ValidationReport(
                target,
                (
                    ValidationIssue(
                        "generation-response",
                        str(exc),
                        severity="error",
                    ),
                ),
            )
        self._event(
            "direct_validation",
            valid=direct_report.valid,
            errors=len(direct_report.errors),
        )
        if direct_report.valid:
            self._event("direct_ready")
            return TemplateDraft(target, contract, direct_report, "llm_direct")

        candidates: tuple[AiterCandidate, ...] = ()
        summaries: tuple[Mapping[str, Any], ...] = ()
        if self.aiter_root is not None:
            self._event("aiter_discovery", root=str(self.aiter_root))
            query = AiterQuery(
                operator=contract.operator,
                input_dtype=contract.input_dtype,
                weight_dtype=contract.weight_dtype,
                input_format=contract.input_format,
                weight_format=contract.weight_format,
                input_scale_granularity=contract.input_scale_granularity,
                weight_scale_granularity=contract.weight_scale_granularity,
                block_size=contract.block_size,
                architecture=contract.architecture,
                shapes=contract.shapes,
            )
            discovered = discover_aiter(self.aiter_root, query, minimum_score=0)
            summaries = tuple(self._candidate_summary(item) for item in discovered[:5])
            candidates = tuple(
                item
                for item in discovered
                if item.score >= self.minimum_aiter_score
            )
            self._event(
                "aiter_candidates",
                discovered=len(discovered),
                eligible=len(candidates),
                best_score=discovered[0].score if discovered else None,
            )

        if not candidates:
            failed = (
                self._preserve_failed(target, contract.contract_hash)
                if target.exists() or target.is_symlink()
                else None
            )
            raise BootstrapError(
                (
                    str(direct_error)
                    if direct_error is not None
                    else "direct template failed validation and no high-confidence "
                    "AITER evidence was found"
                ),
                validation_report=direct_report,
                candidate_summaries=summaries,
                draft_path=failed,
            )

        excerpts, artifacts = self._candidate_evidence(candidates[0])
        repair_report = direct_report
        last_exception: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                self._event(
                    "repair_request",
                    attempt=attempt,
                    previous_errors=len(repair_report.errors),
                )
                repair_bundle = self._request_bundle(
                    self._repair_messages(contract, repair_report, excerpts)
                )
                self._install_bundle(
                    target, repair_bundle, contract, "llm_aiter_repair", artifacts
                )
                repair_report = validate_generated_template(
                    target, contract.expected_contract
                )
                last_exception = None
                self._event(
                    "repair_validation",
                    attempt=attempt,
                    valid=repair_report.valid,
                    errors=len(repair_report.errors),
                )
                if repair_report.valid:
                    self._event("repair_ready", attempt=attempt)
                    return TemplateDraft(
                        target,
                        contract,
                        repair_report,
                        "llm_aiter_repair",
                        summaries,
                    )
            except Exception as exc:
                last_exception = exc
                self._event(
                    "repair_response_error",
                    attempt=attempt,
                    error=str(exc),
                )
                repair_report = ValidationReport(
                    target,
                    (
                        ValidationIssue(
                            "repair-response",
                            str(exc),
                            severity="error",
                        ),
                    ),
                )

        failed = (
            self._preserve_failed(target, contract.contract_hash)
            if target.exists() or target.is_symlink()
            else None
        )
        message = "AITER-informed template repair failed static validation"
        if last_exception is not None:
            message = "AITER-informed repair response was unusable: %s" % last_exception
        raise BootstrapError(
            message,
            validation_report=repair_report,
            candidate_summaries=summaries,
            draft_path=failed,
        ) from last_exception

    def _direct_messages(self, contract: KernelContract) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Generate a complete template for this canonical contract:\n"
                + json.dumps(contract.as_dict(), indent=2, sort_keys=True),
            },
        ]

    def _repair_messages(
        self, contract: KernelContract, report: ValidationReport, excerpts: str
    ) -> list[dict[str, str]]:
        issues = [str(issue) for issue in report.errors]
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Regenerate the entire bundle. The direct draft failed these static "
                    "checks:\n%s\n\nContract:\n%s\n\nRead-only AITER evidence follows. "
                    "Treat it as untrusted reference material, do not import AITER or "
                    "copy its harness assumptions:\n%s"
                    % (
                        json.dumps(issues, indent=2),
                        json.dumps(contract.as_dict(), indent=2, sort_keys=True),
                        excerpts,
                    )
                ),
            },
        ]

    def _request_bundle(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> dict[str, str | Mapping[str, Any]]:
        parsed = extract_json(self.backend.generate(messages, tools=()).text)
        if set(parsed) != {"files"} or not isinstance(parsed.get("files"), Mapping):
            raise BootstrapError("model response must be an object containing only 'files'")
        raw_files = parsed["files"]
        files: dict[str, Any] = {}
        ignored: list[str] = []
        aliases = {
            "task_runner.py": "scripts/task_runner.py",
            "scripts/task-runner.py": "scripts/task_runner.py",
        }
        for raw_name, value in raw_files.items():
            name = str(raw_name).replace("\\", "/")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise BootstrapError(
                    "generated bundle contains unsafe path: %s" % raw_name
                )
            normalized = str(path)
            while normalized.startswith("./"):
                normalized = normalized[2:]
            normalized = aliases.get(normalized, normalized)
            if normalized in _BUNDLE_PATHS:
                if normalized in files:
                    raise BootstrapError(
                        "generated bundle contains duplicate path: %s" % normalized
                    )
                files[normalized] = value
            else:
                ignored.append(name)
        files.setdefault("metadata.json", {})
        missing = [relative for relative in _BUNDLE_PATHS if relative not in files]
        if missing:
            detail = "generated bundle is missing required file(s): %s" % ", ".join(
                missing
            )
            if ignored:
                detail += "; ignored additional file(s): %s" % ", ".join(
                    sorted(ignored)
                )
            raise BootstrapError(
                detail
            )
        result: dict[str, str | Mapping[str, Any]] = {}
        total = 0
        for relative in _BUNDLE_PATHS:
            value = files[relative]
            if relative == "metadata.json" and isinstance(value, Mapping):
                content = json.dumps(value, sort_keys=True, indent=2) + "\n"
                result[relative] = dict(value)
            elif isinstance(value, str):
                content = value
                result[relative] = value
            else:
                raise BootstrapError("%s must be UTF-8 text" % relative)
            size = len(content.encode("utf-8"))
            if size > _MAX_FILE_BYTES:
                raise BootstrapError("%s exceeds the generated file size limit" % relative)
            total += size
        if total > _MAX_BUNDLE_BYTES:
            raise BootstrapError("generated bundle exceeds the total size limit")
        return result

    def _install_bundle(
        self,
        target: Path,
        bundle: Mapping[str, str | Mapping[str, Any]],
        contract: KernelContract,
        method: str,
        source_artifacts: Sequence[Mapping[str, str]],
    ) -> None:
        temporary = self.draft_root / (
            ".tmp-%s-%s" % (contract.contract_hash, uuid.uuid4().hex)
        )
        backup: Optional[Path] = None
        try:
            temporary.mkdir(mode=0o700)
            metadata_value = bundle["metadata.json"]
            if isinstance(metadata_value, Mapping):
                metadata = dict(metadata_value)
            else:
                try:
                    metadata = json.loads(metadata_value)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise BootstrapError("metadata.json must contain a JSON object") from exc
                if not isinstance(metadata, dict):
                    raise BootstrapError("metadata.json must contain a JSON object")
            metadata.update(contract.metadata)
            metadata["provenance"] = {
                "generator": "multi_tune_agent.template_bootstrap",
                "source_request": contract.request,
                "generation_method": method,
                "model": self._model_name(),
                "source_artifacts": [dict(item) for item in source_artifacts],
                "contract_hash": contract.contract_hash,
            }
            rendered = dict(bundle)
            try:
                config = yaml.safe_load(str(bundle["config.yaml"]))
            except yaml.YAMLError as exc:
                raise BootstrapError("config.yaml must contain valid YAML") from exc
            if not isinstance(config, dict):
                raise BootstrapError("config.yaml must contain a YAML mapping")
            config["source_file_path"] = ["kernel.py"]
            if isinstance(config.get("target_kernel_functions"), str):
                config["target_kernel_functions"] = [
                    config["target_kernel_functions"]
                ]
            targets = config.get("target_kernel_functions")
            if not (
                isinstance(targets, list)
                and targets
                and all(isinstance(name, str) and name.strip() for name in targets)
            ):
                inferred_targets = _infer_target_kernel_functions(
                    str(bundle["kernel.py"])
                )
                if inferred_targets:
                    config["target_kernel_functions"] = inferred_targets
            for command_name in (
                "compile_command",
                "correctness_command",
                "performance_command",
            ):
                if isinstance(config.get(command_name), str):
                    config[command_name] = [config[command_name]]
            rendered["config.yaml"] = yaml.safe_dump(
                config, sort_keys=False, allow_unicode=True
            )
            rendered["scripts/task_runner.py"] = _ensure_template_root_importable(
                str(bundle["scripts/task_runner.py"])
            )
            rendered["metadata.json"] = (
                json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
            )
            for relative in _BUNDLE_PATHS:
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(str(rendered[relative]), encoding="utf-8")

            if target.exists() or target.is_symlink():
                backup = self.draft_root / (
                    ".replaced-%s-%s" % (contract.contract_hash, uuid.uuid4().hex)
                )
                os.replace(target, backup)
            os.replace(temporary, target)
            if backup is not None:
                if backup.is_dir() and not backup.is_symlink():
                    shutil.rmtree(backup)
                else:
                    backup.unlink(missing_ok=True)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _model_name(self) -> str:
        value = getattr(self.backend, "model", None)
        return str(value).strip() if value is not None and str(value).strip() else "unknown"

    def _candidate_summary(self, candidate: AiterCandidate) -> Mapping[str, Any]:
        assert self.aiter_root is not None
        return {
            "wrapper_path": candidate.wrapper_path.relative_to(self.aiter_root).as_posix(),
            "score": candidate.score,
            "reasons": list(candidate.reasons),
            "supported_architectures": list(candidate.supported_architectures),
        }

    def _candidate_evidence(
        self, candidate: AiterCandidate
    ) -> tuple[str, tuple[Mapping[str, str], ...]]:
        assert self.aiter_root is not None
        selected = [candidate.wrapper_path]
        for group in (
            candidate.test_paths,
            candidate.reference_paths,
            candidate.benchmark_paths,
        ):
            if group:
                selected.append(group[0])
        unique = []
        for path in selected:
            if path not in unique:
                unique.append(path)

        remaining = _MAX_EVIDENCE_CHARS
        sections = []
        artifacts = []
        for path in unique:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.aiter_root)
            data = resolved.read_bytes()
            relative = resolved.relative_to(self.aiter_root).as_posix()
            artifacts.append(
                {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}
            )
            text = data.decode("utf-8", errors="replace")
            allowance = min(_MAX_EVIDENCE_FILE_CHARS, remaining)
            excerpt = text[:allowance]
            remaining -= len(excerpt)
            suffix = "\n...[truncated]" if len(text) > len(excerpt) else ""
            sections.append("### %s\n%s%s" % (relative, excerpt, suffix))
            if remaining <= 0:
                break
        return "\n\n".join(sections), tuple(artifacts)

    def _preserve_failed(self, target: Path, contract_hash: str) -> Path:
        for index in range(10_000):
            suffix = "" if index == 0 else "-%d" % index
            failed = self.draft_root / (".failed-%s%s" % (contract_hash, suffix))
            if failed.exists() or failed.is_symlink():
                continue
            try:
                os.rename(target, failed)
                return failed
            except (FileExistsError, NotADirectoryError):
                continue
        raise BootstrapError("could not allocate a failed-draft diagnostics directory")


def _ensure_template_root_importable(runner: str) -> str:
    if "sys.path.insert" in runner or "sys.path.append" in runner:
        return runner
    lines = runner.splitlines(keepends=True)
    insertion = 0
    try:
        tree = compile(runner, "scripts/task_runner.py", "exec", ast.PyCF_ONLY_AST)
    except SyntaxError:
        return runner
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion = max(insertion, node.end_lineno or node.lineno)
    setup = (
        "import sys as _bootstrap_sys\n"
        "from pathlib import Path as _BootstrapPath\n"
        "_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))\n"
    )
    lines.insert(insertion, setup)
    return "".join(lines)


def _infer_target_kernel_functions(kernel: str) -> list[str]:
    try:
        tree = ast.parse(kernel, filename="kernel.py")
    except SyntaxError:
        return []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name)
            and target.id == "TARGET_KERNEL_FUNCTIONS"
            for target in targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            names = [
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.strip()
            ]
            if names:
                return names
    wrappers = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and node.name != "main"
        and not node.name.endswith("_kernel")
    ]
    return wrappers if len(wrappers) == 1 else []


def run_template_gpu_gate(
    draft: TemplateDraft,
    *,
    geak_root: Path | str,
    run_root: Path | str,
    gpu_ids: str = "1",
    command_timeout: int = 300,
) -> TemplateGateResult:
    """Validate a draft through GEAK's locked GPU command path without a baseline."""

    if not isinstance(draft, TemplateDraft):
        raise TypeError("draft must be a TemplateDraft")
    if not draft.valid:
        raise ValueError("GPU validation requires a statically valid TemplateDraft")

    fresh_report = validate_generated_template(
        draft.path, draft.contract.expected_contract
    )
    if not fresh_report.valid:
        return TemplateGateResult(
            static_valid=False,
            compiled=False,
            correct=False,
            performance_valid=False,
            per_case_ms={},
            errors=tuple(str(issue) for issue in fresh_report.errors),
        )

    root = Path(run_root).expanduser().resolve()
    episode = root / (
        "template-validation-%s-%s"
        % (draft.contract_hash[:16], uuid.uuid4().hex)
    )
    workspace = episode / "workspace"
    summaries: list[Mapping[str, Any]] = []
    errors: list[str] = []
    try:
        task = TaskSpec(
            task_id="template-%s" % draft.contract_hash[:16],
            task_type=draft.contract.operator,
            kernel_path=draft.path,
        )
        sandbox = KernelSandbox(
            upstream_root=geak_root,
            run_root=root,
            gpu_ids=gpu_ids,
            command_timeout=command_timeout,
        )
        workspace = sandbox.prepare(task, episode)
    except Exception as exc:
        errors.append("sandbox setup failed: %s: %s" % (type(exc).__name__, exc))
        return TemplateGateResult(
            static_valid=True,
            compiled=False,
            correct=False,
            performance_valid=False,
            per_case_ms={},
            command_summaries=_summaries_by_mode(summaries),
            errors=tuple(errors),
            validation_workspace=workspace,
        )

    states = {
        "compile": False,
        "correctness": False,
        "performance": False,
    }
    performance_ms: dict[str, float] = {}
    for mode in ("compile", "correctness", "performance"):
        try:
            result = sandbox.run_mode(mode)
        except Exception as exc:
            errors.append(
                "%s command failed: %s: %s" % (mode, type(exc).__name__, exc)
            )
            break
        summary = _command_summary(result)
        summaries.append(summary)
        if not bool(getattr(result, "ok", False)):
            errors.append("%s command failed" % mode)
            break
        states[mode] = True
        if mode == "performance":
            raw_ms = getattr(result, "per_case_ms", {})
            try:
                performance_ms = {
                    str(name): float(value)
                    for name, value in sorted(
                        raw_ms.items(), key=lambda item: str(item[0])
                    )
                }
            except (AttributeError, TypeError, ValueError):
                performance_ms = {}
            if not performance_ms:
                states["performance"] = False
                errors.append("performance command produced no per-case latency")
            elif any(
                not math.isfinite(value) or value <= 0
                for value in performance_ms.values()
            ):
                states["performance"] = False
                errors.append(
                    "performance command produced non-positive or non-finite latency"
                )

    return TemplateGateResult(
        static_valid=True,
        compiled=states["compile"],
        correct=states["correctness"],
        performance_valid=states["performance"],
        per_case_ms=performance_ms,
        command_summaries=_summaries_by_mode(summaries),
        errors=tuple(errors),
        validation_workspace=workspace,
    )


def promote_validated_template(
    draft: TemplateDraft,
    gate_result: TemplateGateResult,
    verified_root: Path | str,
) -> Path:
    """Atomically promote the four canonical files after a trusted GPU gate."""

    if not isinstance(draft, TemplateDraft):
        raise TypeError("draft must be a TemplateDraft")
    if not isinstance(gate_result, TemplateGateResult):
        raise TypeError("gate_result must be a TemplateGateResult")
    if not draft.valid:
        raise BootstrapError("cannot promote a statically invalid draft")
    if not gate_result.trusted:
        raise BootstrapError("cannot promote a template without a trusted GPU gate")

    fresh_report = validate_generated_template(
        draft.path, draft.contract.expected_contract
    )
    if not fresh_report.valid:
        raise BootstrapError(
            "draft no longer passes static validation",
            validation_report=fresh_report,
            draft_path=draft.path,
        )

    root = Path(verified_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / draft.contract_hash
    temporary = root / (
        ".tmp-%s-%s" % (draft.contract_hash, uuid.uuid4().hex)
    )
    trust = _trust_metadata(draft, gate_result)
    try:
        temporary.mkdir(mode=0o700)
        for relative in _BUNDLE_PATHS:
            source = draft.path / relative
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == "metadata.json":
                metadata = json.loads(source.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise BootstrapError("draft metadata.json must contain an object")
                metadata["trust"] = trust
                target.write_text(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                shutil.copyfile(source, target)

        expected = dict(draft.contract.expected_contract)
        expected["trust"] = trust
        promoted_report = validate_generated_template(temporary, expected)
        if not promoted_report.valid:
            raise BootstrapError(
                "promoted template failed static validation",
                validation_report=promoted_report,
                draft_path=draft.path,
            )

        if destination.exists() or destination.is_symlink():
            if not _canonical_trees_equal(temporary, destination):
                raise BootstrapError(
                    "verified destination exists with mismatched content"
                )
            existing_report = validate_generated_template(destination, expected)
            if not existing_report.valid:
                raise BootstrapError(
                    "existing verified template failed static validation",
                    validation_report=existing_report,
                    draft_path=destination,
                )
            shutil.rmtree(temporary)
            return destination

        os.rename(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _command_summary(result: Any) -> dict[str, Any]:
    return {
        "mode": str(getattr(result, "mode", "")),
        "command": str(getattr(result, "command", "")),
        "returncode": int(getattr(result, "returncode", 1)),
        "timed_out": bool(getattr(result, "timed_out", False)),
        "ok": bool(getattr(result, "ok", False)),
        "stdout": str(getattr(result, "stdout", ""))[-4000:],
        "stderr": str(getattr(result, "stderr", ""))[-4000:],
    }


def _trust_metadata(
    draft: TemplateDraft, result: TemplateGateResult
) -> dict[str, Any]:
    return {
        "verification_method": "kernel_sandbox_gpu_gate",
        "contract_hash": draft.contract_hash,
        "trusted": True,
        "static_valid": result.static_valid,
        "compiled": result.compiled,
        "correct": result.correct,
        "performance_valid": result.performance_valid,
        "per_case_ms": dict(result.per_case_ms),
        "commands": {
            mode: dict(summary)
            for mode, summary in result.command_summaries.items()
        },
        "errors": list(result.errors),
    }


def _summaries_by_mode(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(summary.get("mode", "")): dict(summary) for summary in summaries}


def _canonical_trees_equal(left: Path, right: Path) -> bool:
    if not right.is_dir() or right.is_symlink():
        return False
    try:
        entries = {
            path.relative_to(right).as_posix()
            for path in right.rglob("*")
            if path.is_file()
        }
    except OSError:
        return False
    if entries != set(_BUNDLE_PATHS):
        return False
    for relative in _BUNDLE_PATHS:
        left_path = left / relative
        right_path = right / relative
        if (
            not right_path.is_file()
            or right_path.is_symlink()
            or left_path.read_bytes() != right_path.read_bytes()
        ):
            return False
    return True


_SYSTEM_PROMPT = """\
You generate a safe, standalone GEAK task template for an arbitrary GPU operator.
Return one JSON object only, matching this exact schema:
{"files":{"kernel.py":"...","config.yaml":"...","scripts/task_runner.py":"...",
"metadata.json":{}}}
The files object must have exactly those four paths and no others. metadata.json may
be either an object or a JSON string. Do not use markdown fences.

Requirements:
- kernel.py defines every target_kernel_functions entry from config.yaml.
- config commands invoke scripts/task_runner.py in exactly compile, correctness,
  and performance modes through `docker exec -e
  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} -w "$PWD"
  ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} python3 ...`.
- Because the runner lives under scripts/, add its template root
  (`Path(__file__).resolve().parents[1]`) to sys.path before importing kernel.py.
- The runner uses fixed torch random seeds and an independent torch high-precision
  reference; the reference must never invoke or derive from the kernel under test.
- Gate the requested GPU architecture before kernel import, tensor allocation,
  compilation, correctness, or benchmarking. Normalize ROCm gcnArchName by
  splitting at `:` before comparing it with gfx942/gfx950.
- Never import kernel or AITER at module scope, even below a module-level
  architecture check. Define main(); its first operational call must enforce the
  exact requested architecture, and mode handlers may import kernel locally only
  after that gate returns successfully.
- Performance prints exactly `Perf: <milliseconds> ms (<case_id>)` for each case
  and writes build/performance_report.json with test_cases containing test_case_id
  and execution_time_ms.
- Do not import AITER internally from kernel.py or the runner.
- Use strict, dtype-appropriate static tolerances. Never weaken tolerance, skip a
  comparison, swallow failures, or compare an output with itself.
- Do not use a bare except or catch Exception/BaseException anywhere in the
  runner; let unexpected failures propagate as a nonzero exit.
- Never mutate template sources or metadata and never execute work at import time.
"""


__all__ = [
    "BootstrapError",
    "KernelContract",
    "TemplateBootstrapper",
    "TemplateDraft",
    "TemplateGateResult",
    "promote_validated_template",
    "run_template_gpu_gate",
]
