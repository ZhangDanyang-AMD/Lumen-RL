"""Thread-safe event logging for hierarchical agent trajectories."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


class TrajectoryWriter:
    def __init__(
        self,
        run_dir: Path,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        sft_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "trajectory.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._lock = threading.Lock()
        self._event_sink = event_sink
        self._sft_sink = sft_sink

    def append(
        self,
        event: str,
        payload: Mapping[str, Any],
        *,
        role: str = "orchestrator",
        phase: str = "",
        round_index: int = 0,
    ) -> None:
        record = {
            "event": event,
            "timestamp": time.time(),
            "role": role,
            "phase": phase,
            "round": round_index,
            "payload": dict(payload),
        }
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
        if self._sft_sink is not None:
            # Dataset provenance is authoritative. Unlike terminal/UI subscribers,
            # a failed SFT sink must fail the run rather than silently lose data.
            self._sft_sink(record)
        if self._event_sink is not None:
            try:
                self._event_sink(record)
            except Exception:
                # A terminal/UI subscriber must never break an optimization run.
                pass

    def finalize(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(summary)
        value["finished_at"] = time.time()
        with self._lock:
            self.summary_path.write_text(
                json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        self.append("run_end", value)
        return value

