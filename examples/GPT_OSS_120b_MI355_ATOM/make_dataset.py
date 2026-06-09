#!/usr/bin/env python3
"""Combine UltraChat-200K + Magpie-300K into a single shuffled JSONL for Eagle3 training.

Mirrors Model-Optimizer's make_dataset.py approach: preprocess multiple HuggingFace
datasets into a single JSONL file so training runs as one pass instead of two phases.

NVIDIA's gpt-oss-120b-Eagle3-long-context recipe trains on ~503K prompts:
  - HuggingFaceH4/ultrachat_200k (train_sft split, ~208K samples)
  - Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered (train split, ~300K samples)

Per the NVIDIA model card: "only prompts from the datasets were used for data
synthesis (the original responses from GPT were not used for data synthesis)."
Model-Optimizer's make_dataset.py reads only s["prompt"] from UltraChat and
s["instruction"] from Magpie, stripping all assistant turns.

Output format: one JSON object per line with a "messages" field (prompt only):
  {"messages": [{"role": "user", "content": "..."}]}

Usage:
  python3 examples/GPT_OSS_120b_MI355_ATOM/make_dataset.py
  python3 examples/GPT_OSS_120b_MI355_ATOM/make_dataset.py --output-dir /dev/shm/gpt_oss_120b_dataset
  python3 examples/GPT_OSS_120b_MI355_ATOM/make_dataset.py --seed 42
"""

import argparse
import json
import logging
import os
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_ultrachat(split="train_sft"):
    """Load UltraChat-200K, extracting prompt-only (user turn).

    Model-Optimizer reads only s["prompt"] (line 279 of make_dataset.py),
    ignoring the full conversation messages.
    """
    from datasets import load_dataset

    logger.info("Loading HuggingFaceH4/ultrachat_200k (split=%s)...", split)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    ds = ds.shuffle(seed=42)
    logger.info("Loaded %d samples from ultrachat_200k", len(ds))

    samples = []
    skipped = 0
    for row in ds:
        prompt = row.get("prompt", "")
        if isinstance(prompt, str) and prompt.strip():
            samples.append({
                "messages": [{"role": "user", "content": prompt.strip()}],
                "source": "ultrachat_200k",
            })
        else:
            skipped += 1

    if skipped:
        logger.warning("Skipped %d empty samples from ultrachat_200k", skipped)
    logger.info("Kept %d samples from ultrachat_200k", len(samples))
    return samples


def load_magpie(split="train"):
    """Load Magpie-Llama-3.1-Pro-300K-Filtered, extracting prompt-only (instruction).

    Model-Optimizer reads only s["instruction"] (line 353 of make_dataset.py),
    ignoring the conversation/response fields.
    """
    from datasets import load_dataset

    logger.info(
        "Loading Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered (split=%s)...", split
    )
    ds = load_dataset(
        "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", split=split
    )
    ds = ds.shuffle(seed=42)
    logger.info("Loaded %d samples from Magpie", len(ds))

    samples = []
    skipped = 0
    for row in ds:
        instruction = row.get("instruction", "")
        if isinstance(instruction, str) and instruction.strip():
            samples.append({
                "messages": [{"role": "user", "content": instruction.strip()}],
                "source": "magpie_300k",
            })
        else:
            skipped += 1

    if skipped:
        logger.warning("Skipped %d empty samples from Magpie", skipped)
    logger.info("Kept %d samples from Magpie", len(samples))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Combine UltraChat + Magpie prompts into a single JSONL for Eagle3 training"
    )
    parser.add_argument(
        "--output-dir",
        default="/dev/shm/gpt_oss_120b_dataset",
        help="Output directory for the combined JSONL",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists",
    )
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, "train.jsonl")

    if args.skip_existing and os.path.isfile(output_path):
        logger.info("Output already exists, skipping: %s", output_path)
        return

    ultrachat_samples = load_ultrachat("train_sft")
    magpie_samples = load_magpie("train")

    combined = ultrachat_samples + magpie_samples
    logger.info(
        "Combined: %d prompts (ultrachat=%d, magpie=%d)",
        len(combined),
        len(ultrachat_samples),
        len(magpie_samples),
    )

    random.seed(args.seed)
    random.shuffle(combined)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in combined:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        "Wrote %d prompts to %s (%.1f MB)", len(combined), output_path, file_size_mb
    )


if __name__ == "__main__":
    main()
