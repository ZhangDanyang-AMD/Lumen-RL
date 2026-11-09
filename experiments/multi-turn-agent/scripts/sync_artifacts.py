#!/usr/bin/env python3
"""Sync artifacts/ and metadata/ in agent-kernel-sft-gfx950-triton-HIP with train_v4.jsonl + dev.jsonl.

1. Delete stale artifact dirs not referenced by any sample
2. Create artifact dirs for new samples
3. Update metadata/ files

Usage:
    python3 scripts/sync_artifacts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

DATASET_ROOT = Path("/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP")
TRAIN_FILE = DATASET_ROOT / "train_v4.jsonl"
DEV_FILE = DATASET_ROOT / "dev.jsonl"
ARTIFACTS_DIR = DATASET_ROOT / "artifacts"
METADATA_DIR = DATASET_ROOT / "metadata"


def content_hash(messages: list) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def artifact_id(full_hash: str, existing_ids: set[str]) -> str:
    short = full_hash[:16]
    if short not in existing_ids:
        return short
    long = full_hash[:20]
    return long


def detect_lang(d: dict) -> str:
    all_code = ""
    for m in d["messages"]:
        if m.get("role") == "assistant":
            all_code += m.get("content", "") + "\n"
            for tc in m.get("tool_calls", []):
                try:
                    args = json.loads(tc["function"]["arguments"])
                    all_code += args.get("content", "") + "\n"
                except Exception:
                    pass
        if m["role"] == "user":
            all_code += m.get("content", "") + "\n"

    kernel_code = ""
    for m in d["messages"]:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []):
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if args.get("action") == "write_file":
                        kernel_code += args.get("content", "") + "\n"
                except Exception:
                    pass

    if kernel_code:
        if "flyc" in kernel_code or "flydsl" in kernel_code:
            return "flydsl"
        if "@triton.jit" in kernel_code or "import triton" in kernel_code:
            return "triton"
        if "__global__" in kernel_code or "#include" in kernel_code:
            return "opus"

    if "flyc" in all_code or "flydsl" in all_code.lower():
        return "flydsl"
    if "@triton.jit" in all_code or "import triton" in all_code or "tl.load" in all_code:
        return "triton"
    if "__global__" in all_code or "#include" in all_code:
        return "opus"
    return "unknown"


def extract_kernel(d: dict) -> str:
    for m in reversed(d["messages"]):
        if m.get("role") == "assistant":
            for tc in reversed(m.get("tool_calls", [])):
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if args.get("action") == "write_file":
                        return args.get("content", "")
                except Exception:
                    pass

    for m in d["messages"]:
        if m.get("role") == "assistant":
            text = m.get("content", "")
            fence = re.search(r"```(?:python|triton|cpp|c\+\+|cuda)?\n(.*?)```", text, re.DOTALL)
            if fence:
                return fence.group(1)
            if text.strip():
                return text.strip()

    return ""


def is_multiturn(d: dict) -> bool:
    return any(m.get("tool_calls") for m in d["messages"] if m.get("role") == "assistant")


def load_samples(path: Path) -> list[dict]:
    samples = []
    for line in open(path):
        if line.strip():
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading train: {TRAIN_FILE}")
    train = load_samples(TRAIN_FILE)
    print(f"  {len(train)} samples")

    dev = []
    if DEV_FILE.exists():
        print(f"Loading dev: {DEV_FILE}")
        dev = load_samples(DEV_FILE)
        print(f"  {len(dev)} samples")

    all_samples = train + dev

    # Step 1a: Index samples — compute content hash for each
    print("\nIndexing samples...")
    sample_hashes: dict[str, dict] = {}  # full_hash -> sample
    for d in all_samples:
        sid = d.get("source_sample_id", "")
        if sid and len(sid) >= 16:
            h = sid
        else:
            h = content_hash(d["messages"])
        sample_hashes[h] = d

    print(f"  {len(sample_hashes)} unique hashes")

    # Step 1b: Map hashes to artifact IDs
    # First pass: collect all 16-char prefixes
    prefix_counts: Counter = Counter()
    for h in sample_hashes:
        prefix_counts[h[:16]] += 1

    id_to_hash: dict[str, str] = {}
    hash_to_id: dict[str, str] = {}
    collision_prefixes = {p for p, c in prefix_counts.items() if c > 1}

    for h in sorted(sample_hashes):
        if h[:16] in collision_prefixes:
            aid = h[:20]
        else:
            aid = h[:16]
        id_to_hash[aid] = h
        hash_to_id[h] = aid

    print(f"  {len(collision_prefixes)} prefix collisions (using 20-char IDs)")

    # Step 1c: Delete stale artifacts
    valid_ids = set(id_to_hash.keys())
    existing_ids = set(os.listdir(ARTIFACTS_DIR)) if ARTIFACTS_DIR.exists() else set()

    stale = existing_ids - valid_ids
    keep = existing_ids & valid_ids
    create = valid_ids - existing_ids

    print(f"\nArtifacts: {len(existing_ids)} existing")
    print(f"  Keep: {len(keep)}")
    print(f"  Delete (stale): {len(stale)}")
    print(f"  Create (new): {len(create)}")

    if not args.dry_run:
        for aid in stale:
            shutil.rmtree(ARTIFACTS_DIR / aid)
        print(f"  Deleted {len(stale)} stale artifacts")

        # Step 1d: Create missing artifacts
        created = 0
        for aid in sorted(create):
            h = id_to_hash[aid]
            d = sample_hashes[h]

            candidate_dir = ARTIFACTS_DIR / aid / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            kernel = extract_kernel(d)
            (candidate_dir / "kernel.py").write_text(kernel)

            operator = d.get("operator", "unknown")
            config = {
                "source_file_path": ["kernel.py"],
                "task_type": operator,
            }
            with open(candidate_dir / "config.yaml", "w") as f:
                for k, v in config.items():
                    if isinstance(v, list):
                        f.write(f"{k}:\n")
                        for item in v:
                            f.write(f"- {item}\n")
                    else:
                        f.write(f"{k}: {v}\n")

            metadata = {
                "operator": operator,
                "task_type": d.get("task_type", "unknown"),
                "data_source": d.get("data_source", "unknown"),
                "schema_version": d.get("schema_version", "messages_v1"),
                "content_hash": h,
            }
            with open(candidate_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            created += 1

        print(f"  Created {created} new artifacts")

    # Step 2: Update metadata
    print("\nUpdating metadata...")
    all_operators = sorted(set(d.get("operator", "unknown") for d in all_samples))
    all_task_types = sorted(set(d.get("task_type", "unknown") for d in all_samples))
    all_data_sources = sorted(set(d.get("data_source", "unknown") for d in all_samples))
    all_schemas = sorted(set(d.get("schema_version", "unknown") for d in all_samples))

    train_langs = Counter(detect_lang(d) for d in train)
    dev_langs = Counter(detect_lang(d) for d in dev)
    all_langs = sorted(set(list(train_langs.keys()) + list(dev_langs.keys())))

    train_ops = sorted(set(d.get("operator", "unknown") for d in train))
    train_tasks = sorted(set(d.get("task_type", "unknown") for d in train))
    dev_ops = sorted(set(d.get("operator", "unknown") for d in dev))
    dev_tasks = sorted(set(d.get("task_type", "unknown") for d in dev))

    if not args.dry_run:
        # catalog.yaml
        catalog = {
            "architecture": "gfx950",
            "created": "2026-10-12",
            "dataset_name": "agent-kernel-sft-gfx950-triton-HIP",
            "dataset_version": "sft-gfx950-v9",
            "version": "sft-gfx950-v4",
            "dev_samples": len(dev),
            "format": "messages_v1",
            "gpu_sku": "MI350X",
            "languages": all_langs,
            "last_updated": str(date.today()),
            "operators": all_operators,
            "task_types": all_task_types,
            "data_sources": all_data_sources,
            "total_samples": len(all_samples),
            "train_samples": len(train),
        }
        with open(METADATA_DIR / "catalog.yaml", "w") as f:
            for k, v in catalog.items():
                if isinstance(v, list):
                    f.write(f"{k}:\n")
                    for item in v:
                        f.write(f"- {item}\n")
                elif isinstance(v, int):
                    f.write(f"{k}: {v}\n")
                else:
                    f.write(f"{k}: '{v}'\n")

        # split_manifest.json
        manifest = {
            "train": {
                "count": len(train),
                "source": "GEAK rollouts (Triton+HIP) + OPUS + FlyDSL + claude_generated + synthetic_multiturn",
                "operators": train_ops,
                "task_types": train_tasks,
                "languages": dict(train_langs),
            },
            "dev": {
                "count": len(dev),
                "operators": dev_ops,
                "task_types": dev_tasks,
                "languages": dict(dev_langs),
            },
            "updated": str(date.today()),
        }
        with open(METADATA_DIR / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # support_matrix.json
        support = []
        for lang in sorted(train_langs):
            support.append({
                "language": lang,
                "architecture": "gfx950",
                "status": "ready",
                "train_count": train_langs[lang],
            })
        with open(METADATA_DIR / "support_matrix.json", "w") as f:
            json.dump(support, f, indent=2)

        print("  Updated catalog.yaml, split_manifest.json, support_matrix.json")

    # Step 3: Summary
    final_count = len(valid_ids)
    print(f"\nFinal artifacts: {final_count}")
    print(f"Train: {len(train)}, Dev: {len(dev)}, Total: {len(all_samples)}")
    print(f"Languages: {dict(train_langs)}")

    # Step 4: Print examples
    print("\n=== FlyDSL Example Artifacts ===")
    flydsl_examples = []
    for aid in sorted(valid_ids):
        h = id_to_hash[aid]
        d = sample_hashes[h]
        if detect_lang(d) == "flydsl" and len(flydsl_examples) < 5:
            op = d.get("operator", "?")
            tt = d.get("task_type", "?")
            mt = "multi-turn" if is_multiturn(d) else "single-turn"
            flydsl_examples.append((aid, op, tt, mt))
    for aid, op, tt, mt in flydsl_examples:
        print(f"  {ARTIFACTS_DIR}/{aid}/candidate/kernel.py  ({op}, {tt}, {mt})")

    print("\n=== Multi-turn Examples ===")
    mt_examples = []
    for aid in sorted(valid_ids):
        h = id_to_hash[aid]
        d = sample_hashes[h]
        if is_multiturn(d) and len(mt_examples) < 3:
            op = d.get("operator", "?")
            lang = detect_lang(d)
            mt_examples.append((aid, op, lang))
    for aid, op, lang in mt_examples:
        print(f"  {ARTIFACTS_DIR}/{aid}/candidate/kernel.py  ({op}, {lang})")

    print("\n=== Single-turn Examples ===")
    st_examples = []
    for aid in sorted(valid_ids):
        h = id_to_hash[aid]
        d = sample_hashes[h]
        if not is_multiturn(d) and len(st_examples) < 3:
            op = d.get("operator", "?")
            lang = detect_lang(d)
            st_examples.append((aid, op, lang))
    for aid, op, lang in st_examples:
        print(f"  {ARTIFACTS_DIR}/{aid}/candidate/kernel.py  ({op}, {lang})")


if __name__ == "__main__":
    main()
