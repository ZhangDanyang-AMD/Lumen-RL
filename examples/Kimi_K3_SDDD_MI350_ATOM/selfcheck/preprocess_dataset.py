#!/usr/bin/env python3
"""Build a dataset cache ahead of the training run, and report what survived.

Preprocessing normally happens inside the trainer, which means it happens after
the teacher has claimed the GPUs -- a bad place to discover how many samples a
new ``max_prompt_tokens`` or ``thinking`` setting leaves behind, since
``num_training_steps`` has to be derived from that count to keep the run off the
eval slice. This runs the identical function on CPU so the count is known first
and the training run finds a warm cache.

The arguments must match the trainer's call exactly: every one of them is part of
the cache key, so a single mismatch produces a second cache and the run pays the
full preprocessing cost again.

    python3 preprocess_dataset.py --max-prompt-tokens 1024 --max-length 2048
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/dev/shm/kimi-mtp-dataset-full/train.jsonl")
    ap.add_argument(
        "--tokenizer", default=os.environ.get("MODEL_PATH", ""), required=False,
        help="must equal the resolved policy.model_name; it is in the cache key",
    )
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--max-prompt-tokens", type=int, required=True)
    ap.add_argument("--thinking", default="true")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--cache-dir", default="/dev/shm/lumenrl_cache")
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not args.tokenizer:
        ap.error("--tokenizer is required (or set MODEL_PATH)")

    from lumenrl.data.dataset import load_and_preprocess_dataset

    t0 = time.time()
    data = load_and_preprocess_dataset(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
        chat_template="kimi-k3",
        seed=42,
        last_turn_loss_only="false",
        min_loss_tokens=0,
        num_workers=args.workers,
        cache_dir=args.cache_dir,
        dataset_split="train",
        drop_overlong=True,
        max_prompt_tokens=args.max_prompt_tokens,
        thinking=args.thinking.lower() == "true",
    )

    n = len(data)
    # The sampler walks (step * bs) % len sequentially from 0 and eval holds the
    # final `eval_samples` rows, so this is the last step that cannot reach them.
    steps = (n - args.eval_samples) // args.batch_size
    print("\n" + "=" * 72)
    print(f"thinking={args.thinking}  max_prompt_tokens={args.max_prompt_tokens}  "
          f"max_length={args.max_length}")
    print(f"surviving samples : {n}")
    print(f"preprocess time   : {(time.time() - t0) / 60:.1f} min")
    print(f"num_training_steps: {steps}   "
          f"(covers 0..{steps * args.batch_size - 1}, eval starts at "
          f"{n - args.eval_samples})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
