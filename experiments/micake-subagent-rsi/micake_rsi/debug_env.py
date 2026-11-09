"""Debug flow: frozen DSV4.1 patches MICake when an evaluation outcome is unknown.

Candidate failures (schema, compile, correctness, resource limits) stay with
the specialists. An unknown outcome means the harness itself failed. This
environment lets the Manager's frozen policy read and edit only the configured
roots, run one allowlisted unittest, and ask for a single retry of the
submission that failed.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from lumenrl.agentic.core.protocols import (
    AgentEnvironment,
    EnvironmentCapabilities,
    ResetResult,
    StepResult,
    ToolCall,
)

logger = logging.getLogger(__name__)

READ_TOOL = "read_file"
EDIT_TOOL = "edit_file"
CHECK_TOOL = "run_check"
REPORT_TOOL = "report_repair"
TEST_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
DENY_PARTS = frozenset({".git", ".venv", "__pycache__", ".cursor"})
ENV_KEEP = frozenset({"PATH", "HOME", "USER", "LANG", "LC_ALL", "ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"})
ENV_BLOCK = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION")
CheckRunner = Callable[[Sequence[str], Path, float], Awaitable[Mapping[str, Any]]]


DEBUG_PROMPT = """You are the frozen DSV4.1 Manager, paused in a debug session because MICake returned an unknown outcome.

An unknown outcome is a harness failure (worker crash, unclassified compiler or GPU launch), not a bad schedule.
Do not search for a new GEMM config. Make the smallest source change that turns this failure into a classified
candidate result, or that prevents the harness from dying.

Tools, one per turn:
- `read_file` and `edit_file` only under the editable roots in the observation. `edit_file` replaces one exact snippet.
- `run_check` runs a single unittest whose id starts with the configured prefix.
- `report_repair` ends the session. Set `retry` true only after a patch that addresses this failure; the same
  submission is then evaluated once on a fresh worker. Set `retry` false if you cannot fix it.

Guards: tests, the measurement contract and verification receipts are read-only. Pinned snippets (tolerances,
timing protocol) must survive every edit. After your last edit, `run_check` must pass before `retry` is accepted.
If the retried submission still fails with an unknown outcome, all edits of this session are rolled back.
"""


class WorkspaceError(ValueError):
    """A debug tool asked for a path or command outside the allowlist."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class DebugSettings:
    """Bounds for one debug session (values come from YAML)."""

    enabled: bool
    max_repairs_per_episode: int
    max_turns: int
    max_read_bytes: int
    max_file_bytes: int
    editable_roots: tuple[str, ...]
    check_python: str
    check_cwd: str
    test_prefix: str
    check_timeout_s: float
    protected_globs: tuple[str, ...] = ()
    pinned: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    require_check_before_retry: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DebugSettings":
        check = value["check"]
        return cls(
            protected_globs=tuple(value["protected_globs"]),
            pinned={str(Path(path).resolve()): tuple(snippets) for path, snippets in value["pinned"].items()},
            require_check_before_retry=bool(value["require_check_before_retry"]),
            enabled=bool(value["enabled"]),
            max_repairs_per_episode=int(value["max_repairs_per_episode"]),
            max_turns=int(value["max_turns"]),
            max_read_bytes=int(value["max_read_bytes"]),
            max_file_bytes=int(value["max_file_bytes"]),
            editable_roots=tuple(value["editable_roots"]),
            check_python=str(check["python"]),
            check_cwd=str(check["cwd"]),
            test_prefix=str(check["test_prefix"]),
            check_timeout_s=float(check["timeout_s"]),
        )


def _scrubbed_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key not in ENV_KEEP or any(part in key.upper() for part in ENV_BLOCK):
            continue
        env[key] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def _subprocess_check(argv: Sequence[str], cwd: Path, timeout_s: float) -> Mapping[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=_scrubbed_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return {"returncode": None, "stdout": "", "stderr": f"timed out after {timeout_s}s"}
    return {
        "returncode": process.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated, {len(text) - limit} chars omitted"


class DebugEnvironment(AgentEnvironment):
    """Read/edit/check tools scoped to the MICake trees named in the config."""

    def __init__(
        self,
        settings: DebugSettings,
        failure: Mapping[str, Any],
        check_runner: CheckRunner | None = None,
    ) -> None:
        self.settings = settings
        self.failure = dict(failure)
        self._check = check_runner or _subprocess_check
        self._roots = tuple(Path(root).resolve() for root in settings.editable_roots)
        self.retry = False
        self.summary = ""
        self.files: list[str] = []
        self._originals: dict[Path, str] = {}
        self._checked_since_edit = False

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(backend="micake_debug", snapshot=False, async_step=True)

    def _resolve(self, raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise WorkspaceError("PATH_REQUIRED", "path must be an absolute path under an editable root")
        path = Path(raw)
        if not path.is_absolute():
            raise WorkspaceError("PATH_NOT_ABSOLUTE", "path must be absolute")
        resolved = path.resolve()
        name = resolved.name.lower()
        if name.endswith(".key") or any(part in name for part in ("secret", "credential", "api_key")):
            raise WorkspaceError("PATH_DENIED", "refusing to touch secret material")
        if any(part in DENY_PARTS for part in resolved.parts):
            raise WorkspaceError("PATH_DENIED", "path is under a denied directory")
        if not any(resolved == root or root in resolved.parents for root in self._roots):
            raise WorkspaceError("PATH_OUTSIDE_ROOT", f"path is outside editable roots: {list(self.settings.editable_roots)}")
        return resolved

    def _tools(self) -> list[dict[str, Any]]:
        return [
            _schema(READ_TOOL, "Read a UTF-8 source file under an editable root.", {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "1-based line to start from."},
                "limit": {"type": "integer", "description": "Maximum lines to return."},
            }, ["path"]),
            _schema(EDIT_TOOL, "Replace one exact snippet in a file under an editable root.", {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            }, ["path", "old", "new"]),
            _schema(CHECK_TOOL, "Run one unittest id that starts with the configured prefix.", {
                "test": {"type": "string"},
            }, ["test"]),
            _schema(REPORT_TOOL, "End the debug session. retry=true re-evaluates the failed submission once.", {
                "summary": {"type": "string"},
                "retry": {"type": "boolean"},
            }, ["summary", "retry"]),
        ]

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        diagnostic = self.failure.get("diagnostic")
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("message"), str):
            diagnostic = {**diagnostic, "message": _truncate(diagnostic["message"], 4000)}
        elif isinstance(diagnostic, str):
            diagnostic = _truncate(diagnostic, 4000)
        observation = {
            "stage": self.failure.get("stage"),
            "diagnostic": diagnostic,
            "config": self.failure.get("config"),
            "hypothesis": self.failure.get("hypothesis"),
            "submission": self.failure.get("submission"),
            "editable_roots": list(self.settings.editable_roots),
            "test_prefix": self.settings.test_prefix,
        }
        return ResetResult(
            DEBUG_PROMPT,
            observation,
            self._tools(),
            {"backend": "micake_debug", "role": "debugger", "tools": self._tools()},
        )

    async def step(self, action: ToolCall) -> StepResult:
        try:
            if action.name == READ_TOOL:
                return StepResult(self._read(action.arguments or {}), 0.0, False, "RUNNING", "read")
            if action.name == EDIT_TOOL:
                return StepResult(self._edit(action.arguments or {}), 0.0, False, "RUNNING", "edit")
            if action.name == CHECK_TOOL:
                return StepResult(await self._check_tool(action.arguments or {}), 0.0, False, "RUNNING", "check")
            if action.name == REPORT_TOOL:
                return self._report(action.arguments or {})
        except WorkspaceError as exc:
            return StepResult({"error": exc.code, "detail": exc.detail}, 0.0, False, "RUNNING", "schema_invalid")
        return StepResult(
            {"error": "UNKNOWN_TOOL", "detail": f"available tools: {READ_TOOL}, {EDIT_TOOL}, {CHECK_TOOL}, {REPORT_TOOL}"},
            0.0, False, "RUNNING", "schema_invalid",
        )

    def _read(self, args: Mapping[str, Any]) -> dict[str, Any]:
        path = self._resolve(args.get("path"))
        if not path.is_file():
            raise WorkspaceError("NOT_A_FILE", f"{path} is not a file")
        if path.stat().st_size > self.settings.max_file_bytes:
            raise WorkspaceError("FILE_TOO_LARGE", f"{path} exceeds {self.settings.max_file_bytes} bytes")
        lines = path.read_text(encoding="utf-8").splitlines()
        offset = max(int(args.get("offset") or 1), 1)
        limit = int(args.get("limit") or 200)
        if limit < 1:
            raise WorkspaceError("BAD_LIMIT", "limit must be positive")
        selected = lines[offset - 1:offset - 1 + limit]
        text = "\n".join(f"{offset + index}|{line}" for index, line in enumerate(selected))
        return {"path": str(path), "line_count": len(lines), "text": _truncate(text, self.settings.max_read_bytes)}

    def _edit(self, args: Mapping[str, Any]) -> dict[str, Any]:
        path = self._resolve(args.get("path"))
        if self._protected(path):
            raise WorkspaceError("PATH_PROTECTED", f"{path.name} is part of the frozen oracle (tests, contract, receipts)")
        old, new = args.get("old"), args.get("new")
        if not isinstance(old, str) or not old:
            raise WorkspaceError("EDIT_OLD_REQUIRED", "old must be a non-empty string")
        if not isinstance(new, str):
            raise WorkspaceError("EDIT_NEW_REQUIRED", "new must be a string")
        if not path.is_file():
            raise WorkspaceError("NOT_A_FILE", f"{path} is not a file")
        original = path.read_text(encoding="utf-8")
        if path.stat().st_size > self.settings.max_file_bytes or len(original) + len(new) > self.settings.max_file_bytes * 2:
            raise WorkspaceError("FILE_TOO_LARGE", f"{path} exceeds the edit size cap")
        found = original.count(old)
        if found != 1:
            raise WorkspaceError("EDIT_NOT_UNIQUE", f"old snippet occurs {found} times; it must occur once")
        updated = original.replace(old, new, 1)
        broken = [
            snippet for snippet in self.settings.pinned.get(str(path), ())
            if updated.count(snippet) < original.count(snippet)
        ]
        if broken:
            raise WorkspaceError("CONTRACT_PIN_VIOLATED", f"edit removes pinned measurement-contract text: {broken}")
        self._originals.setdefault(path, original)
        path.write_text(updated, encoding="utf-8")
        self._checked_since_edit = False
        if str(path) not in self.files:
            self.files.append(str(path))
        logger.info("debug edit %s (%d -> %d bytes)", path, len(original), len(updated))
        return {"path": str(path), "bytes": len(updated)}

    def _protected(self, path: Path) -> bool:
        return any(fnmatch.fnmatch(str(path), pattern) or fnmatch.fnmatch(path.name, pattern)
                   for pattern in self.settings.protected_globs)

    def diff(self) -> str:
        """Unified diff of every edit made in this session."""
        chunks = []
        for path, original in self._originals.items():
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            chunks.extend(difflib.unified_diff(
                original.splitlines(keepends=True), current.splitlines(keepends=True),
                fromfile=f"a{path}", tofile=f"b{path}",
            ))
        return "".join(chunks)

    def rollback(self) -> list[str]:
        """Restore every edited file to its content before the session."""
        restored = []
        for path, original in self._originals.items():
            path.write_text(original, encoding="utf-8")
            restored.append(str(path))
        self._originals.clear()
        logger.info("debug rollback %s", restored)
        return restored

    async def _check_tool(self, args: Mapping[str, Any]) -> dict[str, Any]:
        test = args.get("test")
        prefix = self.settings.test_prefix
        if not isinstance(test, str) or not TEST_ID.match(test) or not test.startswith(prefix):
            raise WorkspaceError("CHECK_DENIED", f"test id must match {prefix}<unittest id>")
        cwd = Path(self.settings.check_cwd).resolve()
        if not cwd.is_dir():
            raise WorkspaceError("CHECK_CWD_MISSING", f"check cwd does not exist: {cwd}")
        argv = [self.settings.check_python, "-m", "unittest", test]
        result = await self._check(argv, cwd, self.settings.check_timeout_s)
        output = _truncate(str(result.get("stdout", "")), 4000) + _truncate(str(result.get("stderr", "")), 4000)
        self._checked_since_edit = result.get("returncode") == 0
        logger.info("debug check %s -> %s", test, result.get("returncode"))
        return {"test": test, "returncode": result.get("returncode"), "output": output}

    def _report(self, args: Mapping[str, Any]) -> StepResult:
        summary = args.get("summary")
        retry = args.get("retry")
        if not isinstance(summary, str) or not summary.strip() or not isinstance(retry, bool):
            return StepResult(
                {"error": "REPORT_INVALID", "detail": "summary must be a non-empty string and retry a boolean"},
                0.0, False, "RUNNING", "schema_invalid",
            )
        if retry and self._originals and self.settings.require_check_before_retry and not self._checked_since_edit:
            return StepResult(
                {"error": "REPAIR_UNCHECKED", "detail": "run_check must pass after the last edit before retry=true"},
                0.0, False, "RUNNING", "schema_invalid",
            )
        self.summary = summary.strip()[:1500]
        self.retry = retry
        logger.info("debug report retry=%s files=%s", retry, self.files)
        return StepResult(
            {"status": "reported", "retry": retry, "files": list(self.files)},
            0.0, True, "AGENT_FINISHED", "report_repair",
        )

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        return 0.0, {
            "role": "debugger",
            "terminal_state": terminal_state,
            "retry": self.retry,
            "summary": self.summary,
            "files": list(self.files),
        }

    async def close(self) -> None:
        return None


def _schema(name: str, description: str, properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
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
