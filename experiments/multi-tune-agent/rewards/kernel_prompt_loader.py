"""Load kernel optimization prompts for GRPO rollout.

Reads SFT dataset tasks and constructs prompts in the same format
as SFT training (system + user JSON with contract/parent_source).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are an expert software engineer. Return only the requested patch."


def load_kernel_prompts(
    data_path: str | Path,
    max_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """Load kernel optimization prompts from SFT dataset.

    Each prompt contains:
    - messages: [system, user] in chat format
    - task_id: for tracking
    - task_dir: workspace path for evaluation
    - baseline_ms: baseline performance
    - family: operator family name
    - lane: triton_gfx942 or hip_gfx942
    """
    data_path = Path(data_path)
    prompts = []

    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("sample_domain") != "kernel" and row.get("split") != "train":
                continue

            # Extract fields
            sample_id = row.get("sample_id", "")
            task_type = row.get("task_type", "cold_start")
            inp = row.get("input", {})
            contract = inp.get("contract", {})
            parent_source = inp.get("parent_source", {})
            baseline = inp.get("baseline", {})

            # Build user message (same format as SFT training)
            frozen_input = json.dumps({
                "input": {
                    "contract": contract,
                    "parent_source": parent_source,
                    "baseline": {"commands": baseline.get("commands", {})},
                    "direction": inp.get("direction"),
                    "profile": inp.get("profile"),
                    "error_feedback": None,
                },
                "task_type": task_type,
            })

            prompt = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": frozen_input},
                ],
                "task_id": sample_id,
                "contract": contract,
                "family": contract.get("case_type", ""),
                "lane": f"{contract.get('backend', 'triton')}_gfx942",
                "baseline_ms": baseline.get("geomean_ms", 1.0),
                "parent_source": parent_source,
            }
            prompts.append(prompt)

            if max_prompts and len(prompts) >= max_prompts:
                break

    logger.info("Loaded %d kernel prompts from %s", len(prompts), data_path)
    return prompts


def load_held_out_prompts(
    held_out_root: str | Path,
    baseline_results_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load held-out tasks as evaluation prompts.

    Only includes tasks that pass baseline verification.
    """
    held_out_root = Path(held_out_root)
    prompts = []

    # Load passing tasks
    passing = set()
    if baseline_results_path:
        br_path = Path(baseline_results_path)
        if br_path.exists():
            with open(br_path) as f:
                passing = {r["task"] for r in json.load(f) if r["passed"]}

    with open(held_out_root / "tasks" / "kernel.jsonl") as f:
        for line in f:
            task = json.loads(line)
            tid = task["task_id"]

            if passing and tid not in passing:
                continue

            task_dir = held_out_root / "artifacts" / "kernel" / tid / "initial"
            if not (task_dir / "kernel.py").exists():
                continue

            kernel_source = (task_dir / "kernel.py").read_text()
            contract = task.get("contract", {})

            frozen_input = json.dumps({
                "input": {
                    "contract": contract,
                    "parent_source": {"kernel.py": kernel_source},
                    "baseline": {"commands": {
                        "compile": "python3 scripts/task_runner.py compile",
                        "correctness": "python3 scripts/task_runner.py correctness",
                        "performance": "python3 scripts/task_runner.py performance",
                    }},
                    "direction": None,
                    "profile": None,
                    "error_feedback": None,
                },
                "task_type": "cold_start",
            })

            prompt = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": frozen_input},
                ],
                "task_id": tid,
                "task_dir": str(task_dir),
                "family": task.get("top10_family", ""),
                "lane": task.get("lane", ""),
                "baseline_ms": 1.0,  # will be computed during eval
            }
            prompts.append(prompt)

    logger.info("Loaded %d held-out prompts", len(prompts))
    return prompts
