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
    # /!\ Both of these are in the cache key and both changed with the TorchSpec
    # alignment (false -> true, 0 -> 14). They used to be hardcoded here, which
    # meant pre-warming built a cache the trainer would then ignore and rebuild.
    ap.add_argument("--last-turn-loss-only", default="true")
    ap.add_argument("--min-loss-tokens", type=int, default=14)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--cache-dir", default="/dev/shm/lumenrl_cache")
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    # Usable tmpfs for the hidden-state cache, in GiB, after the mooncake segment
    # and anything else living on /dev/shm. Only used for the cache_batches report.
    # /!\ GiB, matching `df -h`, and matching the units the batch sizes below are
    # reported in. Mixing GB and GiB here overstates the ceiling by 7%, which is
    # most of the margin.
    #   1.5 TiB total - 238 GiB mooncake (256 GB) - ~10 GiB dataset and caches
    #   = ~1288 GiB, taken at 85% for headroom.
    ap.add_argument("--tmpfs-budget-gib", type=float, default=1095.0)
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
        last_turn_loss_only=args.last_turn_loss_only,
        min_loss_tokens=args.min_loss_tokens,
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
          f"max_length={args.max_length}  "
          f"last_turn_loss_only={args.last_turn_loss_only}  "
          f"min_loss_tokens={args.min_loss_tokens}")
    print(f"surviving samples : {n}")
    print(f"preprocess time   : {(time.time() - t0) / 60:.1f} min")
    print(f"num_training_steps: {steps}   "
          f"(covers 0..{steps * args.batch_size - 1}, eval starts at "
          f"{n - args.eval_samples})")

    _report_shape(data, args)
    return 0


def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    i = min(int(q * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[i]


def _report_shape(data: list, args) -> None:
    """Report what the surviving rows imply for the window and cache_batches.

    Both of those were previously set from worst-case arithmetic -- 128 x the full
    window x 98 KB/token -- which assumes every batch pads to the cap. Real
    batches pad to their own longest row, so the honest number comes from the
    length distribution and nothing else. Run this at the largest window under
    consideration: survival at any smaller one is then just a count.
    """
    lengths = sorted(len(d["input_ids"]) for d in data)
    n = len(lengths)
    if not n:
        return

    print("-" * 72)
    print("sequence length of surviving rows")
    print(f"  mean {sum(lengths) / n:8.0f}   p50 {_pct(lengths, 0.50):6d}   "
          f"p90 {_pct(lengths, 0.90):6d}   p99 {_pct(lengths, 0.99):6d}   "
          f"max {lengths[-1]:6d}")

    print("survival if max_total_sequence_length were:")
    for w in (4096, 8192, 12288, 16384):
        kept = sum(1 for x in lengths if x <= w)
        if kept == 0:
            continue
        print(f"  {w:6d}: {kept:7d} rows ({100.0 * kept / n:5.1f}%)  "
              f"-> {(kept - args.eval_samples) // args.batch_size:5d} steps")

    # A cached batch is padded to its own longest row, and the sampler takes rows
    # sequentially, so these are the batches the run will actually build.
    # (5 aux + token_embeds + last_hidden) x 7168 x 2 B = 98304 B per token.
    per_token = 7 * 7168 * 2
    order = [len(d["input_ids"]) for d in data]
    batch_max = sorted(
        max(order[i:i + args.batch_size])
        for i in range(0, n - args.batch_size + 1, args.batch_size)
    )
    if not batch_max:
        return
    gib = 2 ** 30
    worst = args.batch_size * batch_max[-1] * per_token / gib
    p99 = args.batch_size * _pct(batch_max, 0.99) * per_token / gib
    median = args.batch_size * _pct(batch_max, 0.50) * per_token / gib
    print("cached batch size (padded to each batch's longest row)")
    print(f"  median {median:6.1f} GiB   p99 {p99:6.1f} GiB   "
          f"worst {worst:6.1f} GiB")
    print(f"cache_batches ceiling at {args.tmpfs_budget_gib:.0f} GiB of usable tmpfs")
    print(f"  against worst batch: {int(args.tmpfs_budget_gib / max(worst, 1e-9)):4d}"
          f"   against p99: {int(args.tmpfs_budget_gib / max(p99, 1e-9)):4d}")
    print("  /!\\ size against the worst batch, not the median: one long batch in "
          "a round is enough to fill tmpfs.")


if __name__ == "__main__":
    sys.exit(main())
