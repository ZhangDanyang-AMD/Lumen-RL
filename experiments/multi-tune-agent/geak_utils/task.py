"""Task specifications and task-contract validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CONTRACT_CONFIGS = ("config.yaml", "config.yml", "config.json")


def normalize_task_type(value: str) -> str:
    """Normalize any non-empty consumer-defined task type."""

    normalized = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    if not normalized:
        raise ValueError("task type must be non-empty")
    return normalized


def has_task_contract(path: Path) -> bool:
    """Return whether *path* contains a supported runnable-task contract."""

    if (path / "COMMANDMENT.md").is_file():
        return True
    if (path / "unittest.py").is_file() and (path / "meta.json").is_file():
        return True
    return any((path / name).is_file() for name in _CONTRACT_CONFIGS)


@dataclass(frozen=True, init=False)
class TaskSpec:
    """One kernel optimization task.

    ``case_id`` and ``case_type`` are accepted as compatibility aliases so
    existing orchestration code can adopt this package without changing its
    persisted identifiers.
    """

    task_id: str
    task_type: str
    kernel_path: Path
    direction: str = ""
    max_turns: int | None = None

    def __init__(
        self,
        task_id: str | None = None,
        task_type: str | None = None,
        kernel_path: Path | str | None = None,
        direction: str = "",
        max_turns: int | None = None,
        *,
        case_id: str | None = None,
        case_type: str | None = None,
    ) -> None:
        resolved_id = _coalesce_alias("task_id", task_id, "case_id", case_id)
        resolved_type = _coalesce_alias(
            "task_type", task_type, "case_type", case_type
        )
        if not _TASK_ID_RE.fullmatch(resolved_id):
            raise ValueError(
                "task id %r must contain only letters, digits, '.', '_', and '-'"
                % resolved_id
            )
        if kernel_path is None or not str(kernel_path).strip():
            raise ValueError("task %r requires 'kernel_path'" % resolved_id)
        if max_turns is not None and int(max_turns) < 1:
            raise ValueError("task %r max_turns must be >= 1" % resolved_id)

        object.__setattr__(self, "task_id", resolved_id)
        object.__setattr__(self, "task_type", normalize_task_type(resolved_type))
        object.__setattr__(
            self, "kernel_path", Path(kernel_path).expanduser().resolve()
        )
        object.__setattr__(self, "direction", str(direction).strip())
        object.__setattr__(
            self, "max_turns", None if max_turns is None else int(max_turns)
        )

    @property
    def case_id(self) -> str:
        """Compatibility alias for orchestration code using case terminology."""

        return self.task_id

    @property
    def case_type(self) -> str:
        """Compatibility alias for orchestration code using case terminology."""

        return self.task_type

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], base_dir: Path | None = None
    ) -> "TaskSpec":
        task_id = str(
            value.get("id") or value.get("task_id") or value.get("case_id") or ""
        ).strip()
        if not task_id:
            raise ValueError("each task requires a non-empty 'id'")

        task_type = str(
            value.get("type")
            or value.get("task_type")
            or value.get("case_type")
            or ""
        )
        raw_path = str(value.get("kernel_path") or value.get("path") or "").strip()
        if not raw_path:
            raise ValueError("task %r requires 'kernel_path'" % task_id)
        kernel_path = Path(raw_path).expanduser()
        if not kernel_path.is_absolute():
            kernel_path = (base_dir or Path.cwd()) / kernel_path
        kernel_path = kernel_path.resolve()
        if not kernel_path.is_dir():
            raise ValueError(
                "task %r kernel_path is not a directory: %s"
                % (task_id, kernel_path)
            )
        if not has_task_contract(kernel_path):
            raise ValueError(
                "task %r is not a runnable GEAK kernel task: expected config.yaml, "
                "COMMANDMENT.md, or unittest.py + meta.json in %s"
                % (task_id, kernel_path)
            )

        max_turns_raw = value.get("max_turns")
        return cls(
            task_id=task_id,
            task_type=task_type,
            kernel_path=kernel_path,
            direction=str(value.get("direction") or ""),
            max_turns=(
                None if max_turns_raw is None else int(max_turns_raw)
            ),
        )


def _coalesce_alias(
    primary_name: str,
    primary_value: str | None,
    alias_name: str,
    alias_value: str | None,
) -> str:
    primary = "" if primary_value is None else str(primary_value).strip()
    alias = "" if alias_value is None else str(alias_value).strip()
    if primary and alias and primary != alias:
        raise ValueError("%s and %s disagree" % (primary_name, alias_name))
    value = primary or alias
    if not value:
        raise ValueError("%s must be non-empty" % primary_name)
    return value
