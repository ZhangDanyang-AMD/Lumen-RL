#!/usr/bin/env python3
"""Submit each regime's opening once through the real ``mmkca_gemm`` backend on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from lumenrl.agentic.core.protocols import ToolCall  # noqa: E402


async def run(gpu: int) -> int:
    cfg = yaml.safe_load((EXPERIMENT_DIR / "configs" / "mmkca_planner.yaml").read_text(encoding="utf-8"))
    env = build_environment(cfg["micake"]["backend"], cfg["micake"]["module"], **{**cfg["micake"]["kwargs"], "gpu_id": gpu})
    failures = 0
    try:
        for task in ({"portfolio": "decode", "dtype": "fp8", "split": "train", "bundle": 999},
                     {"portfolio": "prefill", "dtype": "bf16", "split": "heldout"}):
            reset = await env.reset({**task, "max_submissions": 1})
            opening = reset.observation["opening"]
            result = await env.step(ToolCall("smoke", "submit_schedule", {**opening, "hypothesis": "opening"}, ""))
            speedup = (result.observation.get("performance") or {}).get("geomean_speedup_vs_baseline")
            print(f"{task['split']} {task['portfolio']} {task['dtype']}: {result.execution_result} speedup={speedup} "
                  f"workloads={len(reset.observation['portfolio'])}", flush=True)
            failures += result.execution_result != "measured"
    finally:
        await env.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    raise SystemExit(asyncio.run(run(parser.parse_args().gpu)))


if __name__ == "__main__":
    main()
