"""Persistent checkpoint discovery for interrupted MultiTune runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class RunCheckpoint:
    run_dir: Path
    case_id: str
    case_type: str
    user_request: str
    workspace: Path
    session_id: str
    baseline: dict[str, Any]
    resume_context: dict[str, Any]


def find_latest_checkpoint(trajectory_root: Path) -> Optional[RunCheckpoint]:
    runs_dir = Path(trajectory_root).expanduser().resolve() / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        if (run_dir / "summary.json").is_file():
            continue
        if (run_dir / "continued.json").is_file():
            continue
        checkpoint = load_checkpoint(run_dir)
        if checkpoint is not None:
            return checkpoint
    return None


def load_checkpoint(run_dir: Path) -> Optional[RunCheckpoint]:
    path = Path(run_dir).expanduser().resolve() / "trajectory.jsonl"
    if not path.is_file():
        return None
    run_start: dict[str, Any] = {}
    latest_session: dict[str, Any] = {}
    last_failure: dict[str, Any] = {}
    event_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            event_count += 1
            payload = record.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            if record.get("event") == "run_start":
                run_start = payload
            elif record.get("event") == "environment_create":
                role = str(payload.get("role") or "")
                if role in {"engineer", "integrator"} or not latest_session:
                    latest_session = payload
            elif record.get("event") == "tool_result":
                result = payload.get("result")
                result = result if isinstance(result, dict) else {}
                if not result.get("ok"):
                    last_failure = {
                        "action": payload.get("action"),
                        "parameters": _without_large_content(payload.get("parameters")),
                        "result": _truncate_value(result),
                    }

    case = run_start.get("case")
    case = case if isinstance(case, dict) else {}
    initial = run_start.get("initial")
    initial = initial if isinstance(initial, dict) else {}
    baseline = initial.get("baseline")
    baseline = baseline if isinstance(baseline, dict) else {}
    workspace = Path(str(latest_session.get("workspace") or "")).expanduser()
    case_id = str(case.get("case_id") or "")
    session_id = str(latest_session.get("session_id") or "")
    if (
        not case_id
        or not session_id
        or not workspace.is_dir()
        or not isinstance(baseline.get("per_case_ms"), dict)
    ):
        return None
    return RunCheckpoint(
        run_dir=path.parent,
        case_id=case_id,
        case_type=str(case.get("case_type") or "gemm"),
        user_request=str(case.get("user_request") or case.get("direction") or ""),
        workspace=workspace.resolve(),
        session_id=session_id,
        baseline=baseline,
        resume_context={
            "source_run_dir": str(path.parent),
            "source_session_id": session_id,
            "events_recovered": event_count,
            "last_failure": last_failure,
            "instruction": (
                "Continue from the recovered source. Inspect and test the current "
                "kernel before making another change."
            ),
        },
    )


def mark_checkpoint_continued(checkpoint: RunCheckpoint) -> None:
    marker = {
        "continued_at": time.time(),
        "case_id": checkpoint.case_id,
        "workspace": str(checkpoint.workspace),
        "session_id": checkpoint.session_id,
    }
    (checkpoint.run_dir / "continued.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _without_large_content(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    copied = dict(value)
    content = copied.get("content")
    if isinstance(content, str):
        copied["content"] = "<omitted %d characters>" % len(content)
    return copied


def _truncate_value(value: Any, limit: int = 12000) -> Any:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {"truncated": text[:limit] + "...(truncated)"}
