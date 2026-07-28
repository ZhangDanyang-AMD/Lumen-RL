"""Merge multiple datasets into a unified JSONL for DSpark training.

Sources (from Inferact/Kimi-K3-DSpark training recipe):
  - lightseekorg/kimi-mtp-dataset       (~477K, perfectblend)
  - nvidia/OpenCodeInstruct              (~5M code instruction pairs)
  - CohereForAI/aya_dataset              (~204K multilingual)
  - nvidia/Nemotron-4-340B-Instruct      (SFT collection, if available)

Output format (one JSONL per phase):
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

This is the format expected by LumenRL's kimi-k3 chat template for
prefill-mode hidden state extraction (responses already included).

Usage:
  python merge_datasets.py --output-dir /dev/shm/kimi-mtp-dataset-merged
  python merge_datasets.py --output-dir /dev/shm/kimi-mtp-dataset-merged --max-per-source 100000
  python merge_datasets.py --output-dir /dev/shm/kimi-mtp-dataset-merged --sources kimi,opencode
"""
import argparse
import json
import logging
import os
import random
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_SOURCES = ["kimi", "opencode", "aya", "nemotron"]


def load_kimi_mtp(path: str, max_samples: Optional[int] = None) -> list[dict]:
    """Load lightseekorg/kimi-mtp-dataset from local JSONL or HuggingFace."""
    records = []
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".jsonl"))
        for fname in files:
            with open(os.path.join(path, fname)) as f:
                for line in f:
                    row = json.loads(line)
                    records.append(row)
                    if max_samples and len(records) >= max_samples:
                        return records
    elif os.path.isfile(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                records.append(row)
                if max_samples and len(records) >= max_samples:
                    return records
    else:
        logger.info("Downloading kimi-mtp-dataset from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("lightseekorg/kimi-mtp-dataset", split="train")
        for row in ds:
            records.append(dict(row))
            if max_samples and len(records) >= max_samples:
                return records
    return records


def normalize_kimi(row: dict) -> Optional[dict]:
    """Normalize kimi-mtp-dataset row to messages format."""
    if "messages" in row:
        return {"messages": row["messages"], "source": "kimi"}
    if "conversations" in row:
        messages = []
        for turn in row["conversations"]:
            role = turn.get("role", turn.get("from", "user"))
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "model"):
                role = "assistant"
            content = turn.get("content", turn.get("value", ""))
            messages.append({"role": role, "content": content})
        return {"messages": messages, "source": "kimi"}
    if "prompt" in row and "response" in row:
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
            "source": "kimi",
        }
    if "input" in row and "output" in row:
        return {
            "messages": [
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": row["output"]},
            ],
            "source": "kimi",
        }
    logger.warning("Skipping kimi row with unknown format: %s", list(row.keys()))
    return None


def load_opencode(max_samples: Optional[int] = None) -> list[dict]:
    """Load nvidia/OpenCodeInstruct."""
    logger.info("Loading OpenCodeInstruct from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    records = []
    for row in ds:
        prompt = row.get("instruction", row.get("prompt", row.get("input", "")))
        response = row.get("response", row.get("output", row.get("completion", "")))
        if not prompt or not response:
            continue
        lang = row.get("lang", row.get("language", ""))
        system_msg = f"You are an expert programmer. Write code in {lang}." if lang else "You are an expert programmer."
        records.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "source": "opencode",
        })
        if max_samples and len(records) >= max_samples:
            break
    return records


def load_aya(max_samples: Optional[int] = None) -> list[dict]:
    """Load CohereForAI/aya_dataset."""
    logger.info("Loading aya_dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("CohereForAI/aya_dataset", split="train", streaming=True)
    records = []
    for row in ds:
        prompt = row.get("inputs", row.get("instruction", ""))
        response = row.get("targets", row.get("output", ""))
        if not prompt or not response:
            continue
        lang = row.get("language", "")
        system_msg = f"You are a helpful multilingual assistant. Respond in {lang}." if lang else "You are a helpful multilingual assistant."
        records.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "source": "aya",
        })
        if max_samples and len(records) >= max_samples:
            break
    return records


def load_nemotron(max_samples: Optional[int] = None) -> list[dict]:
    """Load Nemotron SFT/RL collections (nvidia/HelpSteer2 as proxy)."""
    logger.info("Loading Nemotron collection (HelpSteer2) from HuggingFace...")
    from datasets import load_dataset
    records = []
    try:
        ds = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)
        for row in ds:
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            if not prompt or not response:
                continue
            records.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "source": "nemotron",
            })
            if max_samples and len(records) >= max_samples:
                break
    except Exception as e:
        logger.warning("Failed to load Nemotron collection: %s", e)
    return records


def write_jsonl(records: list[dict], path: str):
    """Write records to JSONL file."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s (%.2f GB)", len(records), path,
                os.path.getsize(path) / 1e9)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets for DSpark training")
    parser.add_argument("--output-dir", default="/dev/shm/kimi-mtp-dataset-merged",
                        help="Output directory for merged dataset")
    parser.add_argument("--kimi-path", default="/dev/shm/kimi-mtp-dataset",
                        help="Path to local kimi-mtp-dataset (dir or JSONL)")
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Max samples per source (for testing)")
    parser.add_argument("--sources", default="kimi,opencode,aya,nemotron",
                        help="Comma-separated list of sources to include")
    parser.add_argument("--val-ratio", type=float, default=0.002,
                        help="Fraction of data for validation split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-ratio", type=float, default=1.0,
                        help="Fraction of data for phase 1 (rest is phase 2)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    sources = [s.strip() for s in args.sources.split(",")]

    all_records = []

    if "kimi" in sources:
        logger.info("=== Loading kimi-mtp-dataset ===")
        raw = load_kimi_mtp(args.kimi_path, args.max_per_source)
        for row in raw:
            normalized = normalize_kimi(row)
            if normalized:
                all_records.append(normalized)
        logger.info("kimi: %d records", sum(1 for r in all_records if r["source"] == "kimi"))

    if "opencode" in sources:
        logger.info("=== Loading OpenCodeInstruct ===")
        records = load_opencode(args.max_per_source)
        all_records.extend(records)
        logger.info("opencode: %d records", len(records))

    if "aya" in sources:
        logger.info("=== Loading aya_dataset ===")
        records = load_aya(args.max_per_source)
        all_records.extend(records)
        logger.info("aya: %d records", len(records))

    if "nemotron" in sources:
        logger.info("=== Loading Nemotron collection ===")
        records = load_nemotron(args.max_per_source)
        all_records.extend(records)
        logger.info("nemotron: %d records", len(records))

    logger.info("=== Total: %d records ===", len(all_records))

    # Shuffle
    random.shuffle(all_records)

    # Source distribution
    source_counts = {}
    for r in all_records:
        src = r["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    logger.info("Source distribution: %s", json.dumps(source_counts, indent=2))

    # Validation split
    val_size = int(len(all_records) * args.val_ratio)
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    # Phase split
    if args.phase1_ratio < 1.0:
        phase1_size = int(len(train_records) * args.phase1_ratio)
        phase1 = train_records[:phase1_size]
        phase2 = train_records[phase1_size:]
        write_jsonl(phase1, os.path.join(args.output_dir, "train_phase1.jsonl"))
        write_jsonl(phase2, os.path.join(args.output_dir, "train_phase2.jsonl"))
    else:
        write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))

    if val_records:
        write_jsonl(val_records, os.path.join(args.output_dir, "val.jsonl"))

    # Write metadata
    meta = {
        "total_records": len(all_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "source_distribution": source_counts,
        "sources": sources,
        "max_per_source": args.max_per_source,
        "seed": args.seed,
        "phase1_ratio": args.phase1_ratio,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=== Done ===")
    logger.info("Output: %s", args.output_dir)
    logger.info("  train: %d records", len(train_records))
    logger.info("  val:   %d records", len(val_records))


if __name__ == "__main__":
    main()
