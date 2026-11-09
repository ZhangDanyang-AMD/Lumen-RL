#!/usr/bin/env python3
"""Merge collected GEAK trajectories into train_v3 to produce train_v4.

Reuses system prompts from align_sft_system_prompts.py to keep consistency.

Usage:
    python3 scripts/merge_collected_trajectories.py \
        --base /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v3.jsonl \
        --collected collected_flydsl.jsonl collected_triton.jsonl \
        --output /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v4.jsonl \
        --triton-target 3164 \
        --flydsl-target 3164
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# Import system prompts from align_sft_system_prompts (sibling script)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "align_sft_system_prompts", ROOT / "scripts" / "align_sft_system_prompts.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LONG_SYS_TRITON = _mod.SYSTEM_PROMPT_TRITON
LONG_SYS_FLYDSL = _mod.SYSTEM_PROMPT_FLYDSL


def detect_lang(d: dict) -> str:
    all_code = ""
    for m in d["messages"]:
        if m["role"] == "assistant":
            all_code += m.get("content", "") + "\n"
            for tc in m.get("tool_calls", []):
                try:
                    args = json.loads(tc["function"]["arguments"])
                    all_code += args.get("content", "") + "\n"
                except Exception:
                    pass
        if m["role"] == "user":
            all_code += m.get("content", "") + "\n"

    # Check write_file content first for accurate detection
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


def assign_system_prompt(sample: dict) -> dict:
    """Assign proper long system prompt for multi-turn collected trajectories."""
    has_tc = any(m.get("tool_calls") for m in sample["messages"] if m.get("role") == "assistant")
    if not has_tc:
        return sample

    lang = detect_lang(sample)
    if lang == "triton":
        target_sys = LONG_SYS_TRITON
    elif lang == "flydsl":
        target_sys = LONG_SYS_FLYDSL
    else:
        return sample

    for m in sample["messages"]:
        if m["role"] == "system":
            m["content"] = target_sys
            break
    else:
        sample["messages"].insert(0, {"role": "system", "content": target_sys})

    return sample


def dedup_trajectories(samples: list[dict]) -> list[dict]:
    """Deduplicate by exact write_file kernel content (raw, including all variations)."""
    groups: dict[str, list[dict]] = {}
    for d in samples:
        kernel_codes = []
        for m in d["messages"]:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls", []):
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        if args.get("action") == "write_file":
                            content = args.get("content", "")
                            kernel_codes.append(content)
                    except Exception:
                        pass
        if kernel_codes:
            fingerprint = hashlib.md5("\n".join(kernel_codes).encode()).hexdigest()
        else:
            asst_msgs = [m for m in d["messages"] if m["role"] == "assistant"]
            if asst_msgs:
                fingerprint = hashlib.md5(
                    (asst_msgs[-1].get("content") or "")[:500].encode()
                ).hexdigest()
            else:
                fingerprint = hashlib.md5(json.dumps(d).encode()).hexdigest()
        groups.setdefault(fingerprint, []).append(d)

    return [items[0] for items in groups.values()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v3.jsonl")
    parser.add_argument("--collected", nargs="+", required=True)
    parser.add_argument("--output", default="/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v4.jsonl")
    parser.add_argument("--triton-target", type=int, default=3164)
    parser.add_argument("--flydsl-target", type=int, default=3164)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    base = [json.loads(l) for l in open(args.base) if l.strip()]
    print(f"Base: {len(base)} samples from {args.base}")

    collected = []
    for path in args.collected:
        samples = []
        bad = 0
        for l in open(path):
            if not l.strip():
                continue
            try:
                samples.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1
        suffix = f" ({bad} bad lines skipped)" if bad else ""
        print(f"Collected: {len(samples)} from {path}{suffix}")
        collected.extend(samples)

    collected = dedup_trajectories(collected)
    collected = [assign_system_prompt(d) for d in collected]
    print(f"After dedup: {len(collected)} collected trajectories")

    base_by_lang = Counter(detect_lang(d) for d in base)
    collected_by_lang = Counter(detect_lang(d) for d in collected)

    print(f"\nBase languages: {dict(base_by_lang)}")
    print(f"Collected languages: {dict(collected_by_lang)}")

    triton_gap = max(0, args.triton_target - base_by_lang.get("triton", 0))
    flydsl_gap = max(0, args.flydsl_target - base_by_lang.get("flydsl", 0))

    print(f"\nTriton gap: {triton_gap} (target {args.triton_target}, have {base_by_lang.get('triton', 0)})")
    print(f"FlyDSL gap: {flydsl_gap} (target {args.flydsl_target}, have {base_by_lang.get('flydsl', 0)})")

    triton_collected = [d for d in collected if detect_lang(d) == "triton"]
    flydsl_collected = [d for d in collected if detect_lang(d) == "flydsl"]

    add_triton = triton_collected[:triton_gap] if len(triton_collected) >= triton_gap else triton_collected
    add_flydsl = flydsl_collected[:flydsl_gap] if len(flydsl_collected) >= flydsl_gap else flydsl_collected

    if len(add_triton) < triton_gap:
        print(f"  WARNING: Only {len(add_triton)} triton collected, need {triton_gap}")
    if len(add_flydsl) < flydsl_gap:
        print(f"  WARNING: Only {len(add_flydsl)} flydsl collected, need {flydsl_gap}")

    output = base + add_triton + add_flydsl
    random.shuffle(output)

    with open(args.output, "w") as f:
        for d in output:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    final_langs = Counter(detect_lang(d) for d in output)
    print(f"\nFinal dataset: {len(output)} samples → {args.output}")
    print(f"  Languages: {dict(final_langs)}")

    tt = Counter(d.get("task_type", "?") for d in output)
    print(f"  Task types:")
    for t, c in tt.most_common():
        print(f"    {t:30s} {c:5d}")

    ds = Counter(d.get("data_source", "?") for d in output)
    print(f"  Data sources:")
    for s, c in ds.most_common():
        print(f"    {s:30s} {c:5d}")


if __name__ == "__main__":
    main()
