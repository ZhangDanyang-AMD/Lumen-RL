"""Run one MICake specialist-subagent RSI generation.

Example:
    export DSV4_API_KEY=...   # never written to configs or logs
    python experiments/micake-subagent-rsi/scripts/run_generation.py \
        --config experiments/micake-subagent-rsi/configs/gen0_planner_critic.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from micake_rsi.loop import run_generation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--generation-id", help="override rsi.generation_id")
    parser.add_argument("--parent-generation-id", help="override rsi.parent_generation_id")
    parser.add_argument("--episodes-per-task", type=int, help="override rsi.episodes_per_task")
    parser.add_argument("--root", help="override rsi.root")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rsi = cfg["rsi"]
    if args.generation_id:
        rsi["generation_id"] = args.generation_id
    if args.parent_generation_id:
        rsi["parent_generation_id"] = args.parent_generation_id
    if args.episodes_per_task:
        rsi["episodes_per_task"] = args.episodes_per_task
    if args.root:
        rsi["root"] = args.root

    manifest = asyncio.run(run_generation(cfg))
    logging.getLogger(__name__).info(
        "metrics:\n%s\npromotion: %s",
        json.dumps(manifest["metrics"], indent=2),
        json.dumps(manifest["promotion"]),
    )


if __name__ == "__main__":
    main()
