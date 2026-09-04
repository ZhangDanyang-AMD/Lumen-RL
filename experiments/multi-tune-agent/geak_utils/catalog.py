"""YAML task-catalog loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .task import TaskSpec


def load_tasks(path: Path | str) -> list[TaskSpec]:
    """Load and validate a non-empty task catalog.

    A catalog may be a top-level list or a mapping with a ``tasks`` list.
    Task types are consumer-defined and are not restricted here.
    """

    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError("task catalog not found: %s" % catalog_path)
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError("could not read task catalog %s: %s" % (catalog_path, exc)) from exc
    except yaml.YAMLError as exc:
        raise ValueError("invalid task catalog YAML in %s: %s" % (catalog_path, exc)) from exc

    raw_tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("task catalog must contain a non-empty 'tasks' list")
    if any(not isinstance(item, dict) for item in raw_tasks):
        raise ValueError("every task catalog entry must be a mapping")

    tasks = [
        TaskSpec.from_mapping(item, base_dir=catalog_path.parent)
        for item in raw_tasks
    ]
    duplicate_ids = _duplicates(task.task_id for task in tasks)
    if duplicate_ids:
        raise ValueError("duplicate task id(s): %s" % ", ".join(duplicate_ids))
    return tasks


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
