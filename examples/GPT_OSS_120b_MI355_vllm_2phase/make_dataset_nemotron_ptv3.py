#!/usr/bin/env python3
"""Prepare Nemotron Post-Training V3 datasets for two-phase Eagle3 training.

Phase 1 (short context): Prompts with tokenized length ≤ max_tokens_phase1 (default 4096).
Phase 2 (long context):  Prompts with tokenized length > max_tokens_phase1.

On-policy Eagle3 training only needs prompts — the teacher generates responses.
We strip all assistant turns and keep only user messages, matching NVIDIA's
Eagle3 methodology: "only prompts from the datasets were used for data synthesis".

Datasets are loaded from the Nemotron Post-Training V3 collection:
https://huggingface.co/collections/nvidia/nemotron-post-training-v3

Output format: one JSON object per line with a "messages" field (prompt only):
  {"messages": [{"role": "user", "content": "..."}], "source": "dataset_name"}

Usage:
    python3 make_dataset_nemotron_ptv3.py --output-dir /dev/shm/gpt_oss_120b_dataset_v3
    python3 make_dataset_nemotron_ptv3.py --output-dir /dev/shm/gpt_oss_120b_dataset_v3 \
        --tokenizer /dev/shm/gpt-oss-120b --max-tokens-phase1 4096
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    repo_id: str
    splits: list[str]
    cap_per_split: int | None = None
    include_tool_turns: bool = False


NEMOTRON_DATASETS: list[DatasetSpec] = [
    # Math
    DatasetSpec(
        repo_id="nvidia/Nemotron-Math-v2",
        splits=["high_part00", "high_part01", "high_part02", "medium", "low"],
        cap_per_split=40000,
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-Math-v3",
        splits=["train"],
        cap_per_split=200000,
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-Math-Proofs-v1",
        splits=["lean"],
        cap_per_split=50000,
    ),
    # Code / SWE
    DatasetSpec(
        repo_id="nvidia/Nemotron-SWE-v1",
        splits=["r2e_gym"],
        include_tool_turns=True,
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-SWE-v2",
        splits=["agentless", "openhands_swe"],
        include_tool_turns=True,
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-Competitive-Programming-v1",
        splits=[
            "competitive_coding_cpp_part00",
            "competitive_coding_cpp_part01",
            "competitive_coding_python_part00",
            "competitive_coding_python_part01",
            "infinibyte_part00",
            "infinibyte_part01",
        ],
        cap_per_split=50000,
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-Competitive-Programming-v2",
        splits=["exercism", "competitive_coding_python", "competitive_coding_cpp", "text_to_sql"],
        cap_per_split=50000,
    ),
    # Science
    DatasetSpec(
        repo_id="nvidia/Nemotron-Science-v1",
        splits=["MCQ", "RQA"],
    ),
    # Chat / Instruction Following
    DatasetSpec(
        repo_id="nvidia/Nemotron-Instruction-Following-Chat-v1",
        splits=["chat_if", "structured_outputs"],
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
        splits=["reasoning_off", "reasoning_on"],
    ),
    # Agentic
    DatasetSpec(
        repo_id="nvidia/Nemotron-Agentic-v1",
        splits=["interactive_agent", "tool_calling"],
        include_tool_turns=True,
    ),
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-Agentic-v2",
        splits=["interactive_agent", "search", "tool_calling"],
        include_tool_turns=True,
    ),
    # Safety
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-Safety-v1",
        splits=["train"],
    ),
    # Finance
    DatasetSpec(
        repo_id="nvidia/Nemotron-SpecializedDomains-Finance-v1",
        splits=["train"],
        cap_per_split=100000,
    ),
    # Multilingual
    DatasetSpec(
        repo_id="nvidia/Nemotron-SFT-Multilingual-v1",
        splits=[
            "code_de", "code_es", "code_fr", "code_it", "code_ja", "code_zh",
            "math_de", "math_es", "math_fr", "math_it", "math_ja", "math_zh",
            "stem_de", "stem_es", "stem_fr", "stem_it", "stem_ja", "stem_zh",
        ],
        cap_per_split=10000,
    ),
]


def extract_prompt(row):
    """Extract prompt-only messages from a row, stripping all assistant turns."""
    messages = row.get("messages", [])
    if not messages:
        return None

    prompt_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            break
        if role in ("user", "system") and isinstance(content, str) and content.strip():
            prompt_messages.append({"role": role, "content": content.strip()})

    if not prompt_messages or not any(m["role"] == "user" for m in prompt_messages):
        return None
    return prompt_messages


def has_tool_turns(row):
    """Check if a conversation has tool turns."""
    messages = row.get("messages", [])
    return any(m.get("role") in ("tool", "function") for m in messages)


def load_dataset_prompts(spec, skip_tool_turns=True):
    """Load prompts from a single dataset spec."""
    from datasets import load_dataset

    source_name = spec.repo_id.split("/")[-1]
    all_prompts = []

    for split in spec.splits:
        logger.info("  Loading %s split=%s ...", spec.repo_id, split)
        try:
            ds = load_dataset(spec.repo_id, split=split)
        except Exception as e:
            logger.warning("  Failed to load %s/%s: %s", spec.repo_id, split, e)
            continue

        if spec.cap_per_split is not None and len(ds) > spec.cap_per_split:
            ds = ds.shuffle(seed=42).select(range(spec.cap_per_split))
            logger.info("    Capped to %d rows", spec.cap_per_split)

        skipped = 0
        for row in ds:
            if skip_tool_turns and not spec.include_tool_turns and has_tool_turns(row):
                skipped += 1
                continue

            prompt = extract_prompt(row)
            if prompt is None:
                skipped += 1
                continue

            all_prompts.append({
                "messages": prompt,
                "source": source_name,
            })

        if skipped:
            logger.info("    Skipped %d rows from %s/%s", skipped, source_name, split)

    logger.info("  Total from %s: %d prompts", source_name, len(all_prompts))
    return all_prompts


def count_tokens(tokenizer, messages):
    """Count tokens for a list of messages."""
    text = ""
    for msg in messages:
        text += msg.get("content", "") + " "
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Nemotron PTv3 datasets for two-phase Eagle3 training"
    )
    parser.add_argument(
        "--output-dir",
        default="/dev/shm/gpt_oss_120b_dataset_v3",
        help="Output directory for phase1 and phase2 JSONL files",
    )
    parser.add_argument(
        "--tokenizer",
        default="/dev/shm/gpt-oss-120b",
        help="Tokenizer path for token counting (to split phases)",
    )
    parser.add_argument(
        "--max-tokens-phase1",
        type=int,
        default=4096,
        help="Maximum prompt token count for phase 1 (short context)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip if output files already exist"
    )
    parser.add_argument(
        "--skip-tool-turns",
        action="store_true",
        default=True,
        help="Skip conversations with tool/function turns",
    )
    args = parser.parse_args()

    phase1_path = os.path.join(args.output_dir, "phase1_short.jsonl")
    phase2_path = os.path.join(args.output_dir, "phase2_long.jsonl")

    if args.skip_existing and os.path.isfile(phase1_path) and os.path.isfile(phase2_path):
        logger.info("Output files already exist, skipping.")
        return

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer from %s ...", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    all_prompts = []
    for spec in NEMOTRON_DATASETS:
        logger.info("Loading %s ...", spec.repo_id)
        prompts = load_dataset_prompts(spec, skip_tool_turns=args.skip_tool_turns)
        all_prompts.extend(prompts)

    logger.info("Total prompts loaded: %d", len(all_prompts))

    phase1_samples = []
    phase2_samples = []
    for sample in all_prompts:
        ntokens = count_tokens(tokenizer, sample["messages"])
        if ntokens <= args.max_tokens_phase1:
            phase1_samples.append(sample)
        else:
            phase2_samples.append(sample)

    logger.info(
        "Phase 1 (≤%d tokens): %d samples", args.max_tokens_phase1, len(phase1_samples)
    )
    logger.info(
        "Phase 2 (>%d tokens): %d samples", args.max_tokens_phase1, len(phase2_samples)
    )

    random.seed(args.seed)
    random.shuffle(phase1_samples)
    random.shuffle(phase2_samples)

    os.makedirs(args.output_dir, exist_ok=True)

    for path, samples, label in [
        (phase1_path, phase1_samples, "Phase 1"),
        (phase2_path, phase2_samples, "Phase 2"),
    ]:
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("%s: wrote %d samples to %s (%.1f MB)", label, len(samples), path, size_mb)

    stats = {
        "total": len(all_prompts),
        "phase1_count": len(phase1_samples),
        "phase2_count": len(phase2_samples),
        "max_tokens_phase1": args.max_tokens_phase1,
        "seed": args.seed,
    }
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
