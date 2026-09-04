"""Isolated kernel workspace and deterministic benchmark adapter."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .errors import SandboxError
from .evaluation import (
    CommandResult,
    EvaluationResult,
    geomean,
    median_cases,
    parse_performance_output,
    parse_performance_report,
)
from .paths import resolve_upstream_paths
from .task import TaskSpec


_SOURCE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".hip",
    ".h",
    ".hpp",
    ".inl",
    ".triton",
}
_SKIP_DIRS = {
    ".git",
    ".torch_ext",
    "__pycache__",
    "build",
    "dist",
    "logs",
}


class KernelSandbox:
    """Run task-declared commands in an isolated upstream-GEAK workspace."""

    def __init__(
        self,
        upstream_root: Path | str | None = None,
        run_root: Path | str | None = None,
        gpu_ids: str = "0",
        command_timeout: int = 300,
        *,
        repository_root: Path | str | None = None,
    ) -> None:
        if upstream_root is not None and repository_root is not None:
            if Path(upstream_root).expanduser().resolve() != Path(
                repository_root
            ).expanduser().resolve():
                raise ValueError("upstream_root and repository_root disagree")
        root = upstream_root if upstream_root is not None else repository_root
        if root is None:
            raise TypeError("an explicit upstream_root is required")

        upstream = resolve_upstream_paths(root)
        self.upstream_root = upstream.root
        self.repository_root = upstream.root
        self.materialize_workspace = upstream.materialize_workspace
        self.gpu_lock = upstream.gpu_lock
        self.run_root = (
            Path(run_root).expanduser().resolve() if run_root is not None else None
        )
        self.gpu_ids = str(gpu_ids)
        self.command_timeout = int(command_timeout)
        if self.command_timeout < 1:
            raise ValueError("command_timeout must be >= 1")

        self.workspace: Path | None = None
        self.commands: dict[str, str] = {}
        self.allowed_write_paths: list[str] = []
        self.baseline_ms: dict[str, float] = {}

    def prepare(self, task: TaskSpec, episode_dir: Path | str) -> Path:
        workspace = Path(episode_dir).expanduser().resolve() / "workspace"
        if workspace.exists():
            raise SandboxError("workspace already exists: %s" % workspace)
        workspace.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "bash",
                str(self.materialize_workspace),
                "--src",
                str(task.kernel_path),
                "--dst",
                str(workspace),
                "--shared-root",
                str(workspace.parent / "_shared"),
                "--link-aiter",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise SandboxError(
                "failed to materialize kernel workspace: %s"
                % (result.stderr or result.stdout)[-4000:]
            )

        self.workspace = workspace.resolve()
        self.commands = self._discover_commands(self.workspace)
        self.allowed_write_paths = self._discover_write_paths(self.workspace)
        if not self.allowed_write_paths:
            raise SandboxError("no editable kernel source found in %s" % self.workspace)
        self.baseline_ms = {}
        return self.workspace

    def establish_baseline(self, repeats: int = 3) -> dict[str, Any]:
        self._require_prepared()
        started = time.monotonic()
        compile_result = self.run_mode("compile") if "compile" in self.commands else None
        if compile_result and not compile_result.ok:
            raise SandboxError(
                "baseline compilation failed: %s"
                % (compile_result.stderr or compile_result.stdout)[-4000:]
            )

        correctness = self.run_mode("correctness")
        if not correctness.ok:
            raise SandboxError(
                "baseline correctness failed: %s"
                % (correctness.stderr or correctness.stdout)[-4000:]
            )

        samples: list[dict[str, float]] = []
        performance_results: list[CommandResult] = []
        for _ in range(max(1, int(repeats))):
            result = self.run_mode("performance")
            if not result.ok:
                raise SandboxError(
                    "baseline benchmark failed: %s"
                    % (result.stderr or result.stdout)[-4000:]
                )
            if not result.per_case_ms:
                raise SandboxError(
                    "could not parse per-case latency from baseline benchmark output"
                )
            samples.append(result.per_case_ms)
            performance_results.append(result)

        self.baseline_ms = median_cases(samples)
        if not self.baseline_ms:
            raise SandboxError("baseline repeats produced no common performance cases")
        return {
            "per_case_ms": self.baseline_ms,
            "geomean_ms": geomean(self.baseline_ms.values()),
            "repeats": len(samples),
            "commands": dict(self.commands),
            "timing": {
                "compile_seconds": (
                    compile_result.elapsed_seconds if compile_result else 0.0
                ),
                "correctness_seconds": correctness.elapsed_seconds,
                "performance_seconds": [
                    result.elapsed_seconds for result in performance_results
                ],
                "total_seconds": time.monotonic() - started,
            },
        }

    def evaluate(self) -> EvaluationResult:
        if not self.baseline_ms:
            raise SandboxError("establish_baseline() must run before evaluate()")

        compile_result = self.run_mode("compile") if "compile" in self.commands else None
        compiled = compile_result is None or compile_result.ok
        if not compiled:
            failed = CommandResult(
                mode="correctness",
                command=self.commands["correctness"],
                returncode=1,
                stdout="",
                stderr="skipped because compilation failed",
                elapsed_seconds=0.0,
            )
            return EvaluationResult(
                compiled=False,
                correct=False,
                speedup_geomean=0.0,
                speedup_arithmetic=0.0,
                baseline_ms=dict(self.baseline_ms),
                candidate_ms={},
                compile=compile_result,
                correctness=failed,
                performance=None,
                error="compilation failed",
            )

        correctness = self.run_mode("correctness")
        if not correctness.ok:
            return EvaluationResult(
                compiled=True,
                correct=False,
                speedup_geomean=0.0,
                speedup_arithmetic=0.0,
                baseline_ms=dict(self.baseline_ms),
                candidate_ms={},
                compile=compile_result,
                correctness=correctness,
                performance=None,
                error="correctness failed",
            )

        performance = self.run_mode("performance")
        shared = sorted(
            name
            for name in set(self.baseline_ms) & set(performance.per_case_ms)
            if self.baseline_ms[name] > 0 and performance.per_case_ms[name] > 0
        )
        if not performance.ok or not shared:
            return EvaluationResult(
                compiled=True,
                correct=True,
                speedup_geomean=0.0,
                speedup_arithmetic=0.0,
                baseline_ms=dict(self.baseline_ms),
                candidate_ms=performance.per_case_ms,
                compile=compile_result,
                correctness=correctness,
                performance=performance,
                error="benchmark failed or produced no baseline-matching cases",
            )

        ratios = [
            self.baseline_ms[name] / performance.per_case_ms[name] for name in shared
        ]
        return EvaluationResult(
            compiled=True,
            correct=True,
            speedup_geomean=geomean(ratios),
            speedup_arithmetic=sum(ratios) / len(ratios),
            baseline_ms=dict(self.baseline_ms),
            candidate_ms=performance.per_case_ms,
            compile=compile_result,
            correctness=correctness,
            performance=performance,
        )

    def run_mode(self, mode: str) -> CommandResult:
        self._require_prepared()
        if mode not in self.commands:
            raise SandboxError("task does not declare a %s command" % mode)
        command = self.commands[mode]
        env = os.environ.copy()
        env["GEAK_GPU_ALLOWED"] = self.gpu_ids
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [
                    "bash",
                    str(self.gpu_lock),
                    self.gpu_ids,
                    "bash",
                    "-lc",
                    command,
                ],
                cwd=str(self.workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.command_timeout,
            )
            result = CommandResult(
                mode=mode,
                command=command,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_seconds=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                mode=mode,
                command=command,
                returncode=124,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                elapsed_seconds=time.monotonic() - started,
                timed_out=True,
            )

        if mode == "performance":
            result.per_case_ms = self.parse_performance(result.stdout)
            if not result.per_case_ms:
                result.per_case_ms = self._parse_performance_report()
        return result

    def tool_definitions(self) -> list[dict[str, Any]]:
        evaluation_modes = [
            mode
            for mode in ("compile", "correctness", "performance")
            if mode in self.commands
        ]
        return [
            _tool(
                "list_files",
                "List files in the isolated kernel workspace.",
                {"path": {"type": "string", "description": "Relative directory, default '.'"}},
                [],
            ),
            _tool(
                "read_file",
                "Read a UTF-8 source or task file from the isolated workspace.",
                {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 4000},
                },
                ["path"],
            ),
            _tool(
                "write_file",
                "Replace one editable kernel source file. Tests and harness files are blocked.",
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                ["path", "content"],
            ),
            _tool(
                "evaluate",
                "Run a task-declared command through GEAK's GPU lock.",
                {"mode": {"type": "string", "enum": evaluation_modes}},
                ["mode"],
            ),
        ]

    def execute_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            if "_malformed_json" in arguments:
                raise SandboxError("tool arguments were not valid JSON")
            if name == "list_files":
                return {
                    "ok": True,
                    "files": self.list_files(str(arguments.get("path") or ".")),
                }
            if name == "read_file":
                return {
                    "ok": True,
                    "content": self.read_file(
                        str(arguments["path"]),
                        int(arguments.get("offset") or 1),
                        int(arguments.get("limit") or 1000),
                    ),
                }
            if name == "write_file":
                self.write_file(str(arguments["path"]), str(arguments["content"]))
                return {"ok": True, "path": str(arguments["path"])}
            if name == "evaluate":
                return self.run_mode(str(arguments["mode"])).to_dict()
            raise SandboxError("unknown tool: %s" % name)
        except (KeyError, TypeError, ValueError, OSError, SandboxError) as exc:
            return {"ok": False, "error": str(exc)}

    def list_files(self, relative: str = ".") -> list[str]:
        root = self._safe_path(relative)
        if not root.is_dir():
            raise SandboxError("not a directory: %s" % relative)
        files: list[str] = []
        for path in root.rglob("*"):
            if any(part in _SKIP_DIRS for part in path.relative_to(self.workspace).parts):
                continue
            if path.is_file():
                files.append(str(path.relative_to(self.workspace)))
            if len(files) >= 1000:
                break
        return sorted(files)

    def read_file(self, relative: str, offset: int = 1, limit: int = 1000) -> str:
        path = self._safe_path(relative)
        if not path.is_file():
            raise SandboxError("not a file: %s" % relative)
        if path.stat().st_size > 4 * 1024 * 1024:
            raise SandboxError("file is too large to read through this tool")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, offset - 1)
        return "\n".join(lines[start : start + limit])

    def write_file(self, relative: str, content: str) -> None:
        path = self._safe_path(relative)
        rel = str(path.relative_to(self.workspace))
        if not any(_within(rel, allowed) for allowed in self.allowed_write_paths):
            raise SandboxError(
                "write blocked for %s; allowed paths: %s"
                % (relative, ", ".join(self.allowed_write_paths))
            )
        if len(content.encode("utf-8")) > 4 * 1024 * 1024:
            raise SandboxError("refusing to write a source file larger than 4 MiB")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def parse_performance(output: str) -> dict[str, float]:
        return parse_performance_output(output)

    def _parse_performance_report(self) -> dict[str, float]:
        self._require_prepared()
        return parse_performance_report(
            self.workspace / "build" / "performance_report.json"
        )

    def _discover_commands(self, workspace: Path) -> dict[str, str]:
        for name in ("config.yaml", "config.yml"):
            path = workspace / name
            if path.is_file():
                try:
                    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except (OSError, yaml.YAMLError) as exc:
                    raise SandboxError("invalid task config %s: %s" % (path, exc)) from exc
                return self._commands_from_mapping(payload)
        config_json = workspace / "config.json"
        if config_json.is_file():
            try:
                payload = json.loads(config_json.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SandboxError(
                    "invalid task config %s: %s" % (config_json, exc)
                ) from exc
            return self._commands_from_mapping(payload)
        if (workspace / "unittest.py").is_file() and (
            workspace / "meta.json"
        ).is_file():
            return {
                "correctness": "python3 unittest.py",
                "performance": "python3 unittest.py",
            }
        commandment = workspace / "COMMANDMENT.md"
        if commandment.is_file():
            return self._commands_from_commandment(
                commandment.read_text(encoding="utf-8")
            )
        raise SandboxError("no runnable kernel contract found in %s" % workspace)

    @staticmethod
    def _commands_from_mapping(payload: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(payload, Mapping):
            raise SandboxError("kernel config must be a mapping")
        commands: dict[str, str] = {}
        keys = {
            "compile": "compile_command",
            "correctness": "correctness_command",
            "performance": "performance_command",
        }
        for mode, key in keys.items():
            value = payload.get(key)
            if isinstance(value, list):
                value = " && ".join(str(part) for part in value if str(part).strip())
            if value:
                commands[mode] = str(value).strip()
        if "correctness" not in commands or "performance" not in commands:
            raise SandboxError(
                "kernel config must declare correctness_command and performance_command"
            )
        return commands

    @staticmethod
    def _commands_from_commandment(text: str) -> dict[str, str]:
        commands: dict[str, str] = {}
        for mode, label in (
            ("compile", "COMPILE"),
            ("correctness", "CORRECTNESS"),
            ("performance", "FULL_BENCHMARK"),
        ):
            pattern = re.compile(
                r"(?:^|\n)(?:[-*# ]*)\*{0,2}%s\*{0,2}\s*[:—-]\s*`?([^\n`]+)"
                % label,
                re.I,
            )
            match = pattern.search(text)
            if match:
                commands[mode] = _unwrap_gpu_locked_command(match.group(1).strip())
        if "correctness" not in commands or "performance" not in commands:
            raise SandboxError(
                "COMMANDMENT.md must expose CORRECTNESS and FULL_BENCHMARK commands"
            )
        return commands

    @staticmethod
    def _discover_write_paths(workspace: Path) -> list[str]:
        kernel_src = workspace / "kernel_src"
        if kernel_src.is_dir():
            return ["kernel_src"]

        allowed: list[str] = []
        for path in workspace.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            rel_path = path.relative_to(workspace)
            if any(
                part in _SKIP_DIRS or part == "scripts"
                for part in rel_path.parts[:-1]
            ):
                continue
            lower_name = path.name.lower()
            if lower_name.startswith("test_") or lower_name.endswith("_test.py"):
                continue
            if lower_name in {"unittest.py", "harness_lib.py", "leg_runner.py"}:
                continue
            allowed.append(str(rel_path))
        return sorted(allowed)

    def _safe_path(self, relative: str) -> Path:
        self._require_prepared()
        raw = Path(relative)
        if raw.is_absolute():
            raise SandboxError("tool paths must be relative to the workspace")
        path = (self.workspace / raw).resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as exc:
            raise SandboxError("path escapes the isolated workspace") from exc
        return path

    def _require_prepared(self) -> None:
        if self.workspace is None:
            raise SandboxError("sandbox has not been prepared")


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def _within(path: str, allowed: str) -> bool:
    return path == allowed or path.startswith(allowed.rstrip("/") + "/")


def _unwrap_gpu_locked_command(command: str) -> str:
    value = re.sub(r"^cd\s+(?:\"[^\"]+\"|'[^']+'|\S+)\s*&&\s*", "", command)
    locked = re.search(
        r"(?:^|\s)(?:bash\s+)?(?:\"[^\"]*gpu_lock\.sh\"|'[^']*gpu_lock\.sh'|"
        r"\S*gpu_lock\.sh)\s+(?:\"[^\"]+\"|'[^']+'|\S+)\s+(.+)$",
        value,
    )
    return locked.group(1).strip() if locked else value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
