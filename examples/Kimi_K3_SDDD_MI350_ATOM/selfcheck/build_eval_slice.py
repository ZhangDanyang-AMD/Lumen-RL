#!/usr/bin/env python3
"""Materialise a fixed eval slice so acceptance length stays comparable.

``_build_eval_cache`` holds out the tail of the shuffled training set, which
means the eval slice moves whenever the training set does. gen-3's AL of 1.617
was measured on the tail 128 rows of the regenerated kimi-mtp set; a run on a
different dataset producing "1.9" on its own tail says nothing about whether the
draft improved, because the two numbers come from different rows. (This is the
mistake recorded as problem 17 in the regeneration runbook: a larger, more
diverse dataset correctly reads *lower*, and reading that as a regression sent
someone down a long wrong path.)

So: rebuild the previous run's slice by re-running the identical preprocessing on
the identical dataset with the identical seed, take the same tail, and write it
to a file the trainer loads under its own metric prefix.

    python3 build_eval_slice.py \
        --dataset /shared/datasets/atom_regen_kimi_mtp/train.jsonl \
        --tokenizer /mnt/.../Kimi-K3 --max-length 8192 --max-prompt-tokens 0 \
        --num-samples 128 --out /shared/eval/gen3_old_128.pt

Every argument that feeds the cache key must match the run being reproduced or
the shuffle lands elsewhere and the slice is a different 128 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# The regenerated kimi-mtp set stores image turns as a literal JSON string in
# content (runbook 10.5). K3 never saw the image, so those answers are invented;
# they are also the most template-like rows in the set and therefore the easiest
# to predict, which is why the split is worth tagging rather than ignoring.
_IMAGE_SHELL = re.compile(r'\[\s*\{\s*"type"\s*:')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--max-prompt-tokens", type=int, required=True)
    ap.add_argument("--thinking", default="true")
    ap.add_argument("--last-turn-loss-only", default="true")
    ap.add_argument("--min-loss-tokens", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--cache-dir", default="/dev/shm/lumenrl_cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from lumenrl.data.dataset import load_and_preprocess_dataset
    from lumenrl.data.kimi_k25_parser import unpack_loss_mask

    t0 = time.time()
    data = load_and_preprocess_dataset(
        dataset_path=a.dataset,
        tokenizer_path=a.tokenizer,
        max_length=a.max_length,
        chat_template="kimi-k3",
        seed=a.seed,
        last_turn_loss_only=a.last_turn_loss_only,
        min_loss_tokens=a.min_loss_tokens,
        num_workers=a.workers,
        cache_dir=a.cache_dir,
        dataset_split="train",
        drop_overlong=True,
        max_prompt_tokens=a.max_prompt_tokens,
        thinking=a.thinking.lower() == "true",
    )
    n = len(data)
    if a.num_samples > n:
        raise SystemExit(f"asked for {a.num_samples} rows, dataset has {n}")
    logging.info("preprocessed %d rows in %.1f min", n, (time.time() - t0) / 60)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    rows = []
    tags = []
    lengths = []
    for idx in range(n - a.num_samples, n):
        item = data[idx]
        ids = item["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        ids = ids[: a.max_length - 1]
        lm = unpack_loss_mask(item["packed_loss_mask"])[: len(ids)]
        if len(lm) < len(ids):
            lm = torch.cat([lm, torch.zeros(len(ids) - len(lm), dtype=torch.long)])
        text = tok.decode(ids[: min(len(ids), 512)].tolist())
        tags.append("image" if _IMAGE_SHELL.search(text) else "text")
        rows.append({"input_ids": ids, "loss_mask": lm})
        lengths.append(int(len(ids)))

    params = {
        "dataset": a.dataset,
        "tokenizer": a.tokenizer,
        "max_length": a.max_length,
        "max_prompt_tokens": a.max_prompt_tokens,
        "thinking": a.thinking,
        "last_turn_loss_only": a.last_turn_loss_only,
        "min_loss_tokens": a.min_loss_tokens,
        "seed": a.seed,
        "num_samples": a.num_samples,
        "dataset_rows_after_preprocess": n,
    }
    # A slice is only comparable to another if it came out of the same inputs, so
    # store a fingerprint of the rows themselves alongside the parameters.
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["input_ids"].numpy().tobytes())
    fingerprint = digest.hexdigest()[:16]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save(
        {
            "rows": rows,
            "tags": tags,
            "meta": {
                "dataset": a.dataset,
                "params": params,
                "fingerprint": fingerprint,
            },
        },
        a.out,
    )

    lengths_sorted = sorted(lengths)
    print("\n" + "=" * 72)
    print(f"slice          : rows {n - a.num_samples}..{n - 1} of {n}")
    print(f"fingerprint    : {fingerprint}")
    print(f"tags           : image={tags.count('image')} text={tags.count('text')}")
    print(f"length mean    : {sum(lengths) / len(lengths):.1f}")
    print(f"length p50/p90 : {lengths_sorted[len(lengths) // 2]} / "
          f"{lengths_sorted[int(0.9 * len(lengths))]}")
    print(f"supervised tok : "
          f"{sum(int(r['loss_mask'].sum()) for r in rows) / len(rows):.1f} per row")
    print(f"-> {a.out}")
    print(json.dumps(params, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
