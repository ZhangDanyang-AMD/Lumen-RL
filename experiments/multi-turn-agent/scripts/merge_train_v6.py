#!/usr/bin/env python3
"""Merge validated datasets into train_v6.jsonl with deduplication."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def content_hash(sample: dict) -> str:
    msgs = sample.get("messages", [])
    text = "".join(m.get("content", "") for m in msgs)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="train_v5.jsonl (validated base)")
    parser.add_argument("--extra", type=Path, nargs="+", help="Additional jsonl files to merge")
    parser.add_argument("--output", type=Path, required=True, help="Output train_v6.jsonl")
    args = parser.parse_args()

    seen = set()
    samples = []

    def add_file(path: Path, label: str):
        added = 0
        skipped = 0
        with open(path) as f:
            for line in f:
                s = json.loads(line)
                h = content_hash(s)
                if h in seen:
                    skipped += 1
                    continue
                seen.add(h)
                samples.append(s)
                added += 1
        print(f"  {label}: {added} added, {skipped} duplicates")

    print("Loading base...")
    add_file(args.base, str(args.base.name))

    if args.extra:
        print("Loading extras...")
        for p in args.extra:
            add_file(p, str(p.name))

    print(f"\nTotal: {len(samples)} unique samples")

    lang_counts = Counter()
    op_counts = Counter()
    task_type_counts = Counter()
    source_counts = Counter()
    for s in samples:
        msgs = s.get("messages", [])
        sys_msg = msgs[0].get("content", "").lower() if msgs else ""
        if "flydsl" in sys_msg:
            lang = "flydsl"
        elif "opus" in sys_msg or "hip c++" in sys_msg or "hip kernel" in sys_msg:
            lang = "opus"
        else:
            lang = "triton"
        lang_counts[lang] += 1
        op_counts[s.get("operator", "unknown")] += 1
        task_type_counts[s.get("task_type", "unknown")] += 1
        source_counts[s.get("data_source", "unknown")] += 1

    print(f"\nBy language: {dict(lang_counts)}")
    print(f"\nBy data_source:")
    for k, v in source_counts.most_common():
        print(f"  {k}: {v}")
    print(f"\nBy operator (top 20):")
    for k, v in op_counts.most_common(20):
        print(f"  {k}: {v}")
    print(f"\nBy task_type:")
    for k, v in task_type_counts.most_common():
        print(f"  {k}: {v}")

    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(samples)} samples to {args.output}")

    meta = {
        "total": len(samples),
        "by_language": dict(lang_counts),
        "by_data_source": dict(source_counts),
        "by_operator": dict(op_counts),
        "by_task_type": dict(task_type_counts),
    }
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
