#!/usr/bin/env python3
"""SFT Qwen3-Coder-30B-A3B on MMKCA GEMM search episodes and merge a planner checkpoint.

The episodes are the improving submits collected by MICake-MMKCA/scripts/collect_planner_data.py.
Held-out Qwen3 shapes are not in this data.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ab_eval import DEFAULT_CONFIG, REPO_ROOT  # noqa: E402
from run_cycle import _merge, _train  # noqa: E402
from train_rft import DATA_PARALLEL, specialist_containers  # noqa: E402
from micake_rsi.cycle import MERGE_MANIFEST, training_config  # noqa: E402
from micake_rsi.sft_data import write_role_sft  # noqa: E402

logger = logging.getLogger("train_mmkca_planner")
DATA_DIR = Path("/home/danyzhan/MICake-MMKCA/build/planner-data")
GIB = 1024 ** 3


def load_episodes(directory: Path) -> list[dict]:
    seen: set[str] = set()
    episodes = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            key = sample.get("source_key")
            if key in seen or not any(step.get("advantage", 0) > 0 for step in sample.get("steps") or ()):
                continue
            seen.add(key)
            episodes.append(sample)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "mmkca-planner" / "g0")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--min-episodes", type=int, default=160)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    episodes = load_episodes(args.data)
    positives = sum(step["advantage"] > 0 for episode in episodes for step in episode["steps"])
    logger.info("%d episodes, %d positive turns", len(episodes), positives)
    if len(episodes) < args.min_episodes:
        raise SystemExit(f"only {len(episodes)} episodes; want at least {args.min_episodes}")
    free = shutil.disk_usage(args.out if args.out.exists() else args.out.parent).free
    if free < 110 * GIB:
        raise SystemExit(f"disk free {free / GIB:.0f} GiB is below the 110 GiB needed to merge a 30B checkpoint")

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    train = cfg["train"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(train["base_model"], trust_remote_code=True)
    data_dir = args.out / "train" / "planner"
    manifest = write_role_sft(episodes, tokenizer, data_dir, min_advantage=0.0,
                              max_sequence_length=int(train["max_sequence_length"]),
                              tokenizer_sha256=train["tokenizer_sha256"], tokenizer_path=train["base_model"])
    rows = manifest["artifacts"]["train"]["rows"]
    steps = max(1, math.ceil(args.epochs * rows / DATA_PARALLEL))
    config = training_config({**train, "max_steps": steps}, role="planner", base_model=train["base_model"],
                             data_dir=data_dir, manifest=manifest)
    config_path = data_dir / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes), encoding="utf-8")
    logger.info("tokenized %d rows, %d steps, skipped_long=%s", rows, steps, manifest["skipped_long_rows"])
    if args.dry_run:
        return
    checkpoint = args.out / "checkpoint"
    if (checkpoint / MERGE_MANIFEST).exists():
        logger.info("checkpoint already merged at %s", checkpoint)
        return
    for name in specialist_containers():
        logger.info("removing specialist container %s", name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _train(config_path, train["image"], DATA_PARALLEL)
    _merge(train["image"], Path(train["base_model"]), Path(config["output"]["adapter_dir"]), checkpoint)
    logger.info("merged planner checkpoint at %s", checkpoint)


if __name__ == "__main__":
    main()
