"""Planner contexts captured from end-to-end Manager episodes.

Each context is one real delegation: the Manager's instruction plus the shared
history the Planner saw. Episodes with ``sample < split_at`` form the training
pool, the rest the held-out benchmark, so the two never share an episode.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


def e2e_planner_contexts(directory: Path, split: str, split_at: int = 12) -> list[dict[str, Any]]:
    """De-duplicated ``{id, instruction, history}`` contexts for ``split`` ("train" or "test")."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}")
    contexts: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(Path(directory) / "e2e_*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if (record["sample"] < split_at) != (split == "train"):
                continue
            for child in record.get("children") or ():
                if child.get("role") != "planner":
                    continue
                packet = json.loads(child["messages"][1]["content"])
                history = [{"config": h["config"], "hypothesis": h.get("hypothesis") or "earlier candidate"}
                           for h in packet.get("history") or () if h.get("config")]
                key = json.dumps([packet["manager_instruction"], [h["config"] for h in history]], sort_keys=True)
                contexts.setdefault(key, {"id": f"{record['key']}#{child['episode_id'][:8]}",
                                          "instruction": packet["manager_instruction"], "history": history})
    return [contexts[k] for k in sorted(contexts)]
