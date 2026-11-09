"""Path validation for an upstream GEAK checkout."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import SandboxError


@dataclass(frozen=True)
class UpstreamPaths:
    root: Path
    materialize_workspace: Path
    gpu_lock: Path


def example_tasks_root() -> Path:
    """Locate geak_utils example tasks in a source tree or installed wheel."""

    candidates = (
        Path(__file__).resolve().parents[1] / "examples" / "tasks",
        Path(sys.prefix).resolve() / "share" / "geak-utils" / "examples" / "tasks",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "geak_utils example tasks are unavailable; checked: %s"
        % ", ".join(str(path) for path in candidates)
    )


def example_task_path(task_type: str) -> Path:
    """Return one bundled example task directory."""

    name = str(task_type).strip()
    if not name or Path(name).name != name:
        raise ValueError("example task type must be one path component")
    task = example_tasks_root() / name
    if not task.is_dir():
        raise FileNotFoundError("geak_utils example task is missing: %s" % task)
    return task


def resolve_upstream_paths(root: Path | str) -> UpstreamPaths:
    """Resolve the helper scripts required from an upstream GEAK checkout."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise SandboxError("upstream GEAK root is not a directory: %s" % resolved)
    scripts = resolved / "kernel_workflow" / "scripts"
    materialize = scripts / "materialize_workspace.sh"
    gpu_lock = scripts / "gpu_lock.sh"
    missing = [path.name for path in (materialize, gpu_lock) if not path.is_file()]
    if missing:
        raise SandboxError(
            "upstream GEAK checkout is missing required helper(s): %s (under %s)"
            % (", ".join(missing), scripts)
        )
    return UpstreamPaths(resolved, materialize, gpu_lock)
