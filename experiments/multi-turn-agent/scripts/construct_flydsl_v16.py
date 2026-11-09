#!/usr/bin/env python3
"""Generate replacement FlyDSL trajectories for splitk_gemm + attention family.

Uses new real-operator templates + OPERATOR_GUIDANCE injection.
Only processes operators that need template replacement.

Usage:
    python3 scripts/construct_flydsl_v16.py --workers 4 --gpu-ids 2,3,4,5 --min-count 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import yaml
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from construct_trajectories import (
    TEMPLATES_DIR, CASES_YAML, DOCKER_CONTAINER,
    SYSTEM_PROMPT_FLYDSL, OPERATOR_GUIDANCE,
    FLYDSL_TEMPLATE_VARIANTS,
    customize_kernel, build_trajectory,
    docker_test, docker_cp_test, check_docker_accessible,
    _structural_hash,
)

TARGET_OPS = {'splitk_gemm', 'attention', 'fmha', 'mla'}


def process_case_targeted(case: dict, gpu_id: str, global_seed: int = 42) -> dict | None:
    case_id = case["id"]
    task_dir = Path(case.get("kernel_path", ""))
    if not task_dir.exists():
        return None

    cfg_path = task_dir / "config.yaml"
    if not cfg_path.exists():
        return None

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg['_global_seed'] = global_seed
    language = cfg.get("language", "triton")
    operator = cfg.get("operator", "unknown")

    if language != "flydsl":
        return None
    if operator not in TARGET_OPS:
        return None

    variants = FLYDSL_TEMPLATE_VARIANTS.get(operator, ["generic_flydsl.py"])
    seed_salt = str(global_seed)
    case_rng = random.Random(hashlib.md5((case_id + seed_salt).encode()).hexdigest())
    template_name = case_rng.choice(variants)

    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        print(f"  SKIP: template {template_name} not found")
        return None

    kernel_code = template_path.read_text()
    kernel_code = customize_kernel(kernel_code, case_id, operator, cfg)
    kernel_dst = task_dir / "kernel.py"
    kernel_dst.write_text(kernel_code)

    task_runner_path = task_dir / "scripts" / "task_runner.py"
    task_runner_content = task_runner_path.read_text() if task_runner_path.exists() else "# task_runner.py not found"
    config_content = cfg_path.read_text()

    shapes = cfg.get("benchmark_shapes", [])
    kernel_prompt = ""
    if shapes and isinstance(shapes[0], dict):
        kernel_prompt = shapes[0].get("kernel_prompt", "")

    for _attempt in range(2):
        if check_docker_accessible(task_dir):
            eval_result = docker_test(task_dir, gpu_id)
        else:
            eval_result = docker_cp_test(task_dir, gpu_id)
        if eval_result["compiled"] and eval_result["correct"]:
            break
        time.sleep(2)

    if not eval_result["compiled"] or not eval_result["correct"]:
        return None

    task_type = random.choices(
        ["cold_start", "profile_guided", "direction_conditioned",
         "error_recovery", "regression_balance"],
        weights=[15, 15, 45, 15, 10],
        k=1,
    )[0]

    return build_trajectory(
        case_id=case_id,
        operator=operator,
        language=language,
        kernel_prompt=kernel_prompt,
        task_runner_content=task_runner_content,
        config_content=config_content,
        kernel_code=kernel_code,
        eval_result=eval_result,
        task_type=task_type,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default="2,3,4,5")
    parser.add_argument("--min-count", type=int, default=400)
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cases_data = yaml.safe_load(CASES_YAML.read_text())
    all_cases = cases_data.get("tasks", [])
    cases = [c for c in all_cases
             if "-flydsl-" in c["id"]
             and any(f"-{op}-" in c["id"] for op in TARGET_OPS)]

    gpu_ids = args.gpu_ids.split(",")
    output_path = args.output or str(ROOT / "constructed_flydsl_v16.jsonl")

    print(f"Targeted FlyDSL Generation (v16)")
    print(f"  Target ops: {TARGET_OPS}")
    print(f"  Cases: {len(cases)}")
    print(f"  Workers: {args.workers} | GPUs: {args.gpu_ids}")
    print(f"  Min count: {args.min_count} | Max passes: {args.max_passes}")
    print(f"  Output: {output_path}")
    print()

    collected = []
    seen_hashes = set()
    stats = Counter()
    start = time.monotonic()

    with open(output_path, "w") as out_file:
        for pass_idx in range(args.max_passes):
            seed = 42 + pass_idx * 1000
            if len(collected) >= args.min_count:
                break

            print(f"--- Pass {pass_idx + 1} (seed={seed}) ---")
            random.seed(seed)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for i, case in enumerate(cases):
                    gpu = gpu_ids[i % len(gpu_ids)]
                    fut = pool.submit(process_case_targeted, case, gpu, seed)
                    futures[fut] = case["id"]

                for fut in as_completed(futures):
                    cid = futures[fut]
                    stats["tested"] += 1
                    try:
                        result = fut.result()
                    except Exception as e:
                        stats["error"] += 1
                        continue

                    if result is None:
                        stats["failed"] += 1
                        continue

                    stats["passed"] += 1
                    kernel_code = ""
                    for msg in result.get("messages", []):
                        if msg.get("role") == "assistant":
                            for tc in msg.get("tool_calls", []):
                                args_str = tc.get("function", {}).get("arguments", "")
                                try:
                                    args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                                    if isinstance(args_obj, dict) and args_obj.get("action") == "write_file":
                                        c = args_obj.get("content", "")
                                        if "def " in c and "flyc" in c:
                                            kernel_code = c
                                except (json.JSONDecodeError, AttributeError):
                                    pass

                    if kernel_code:
                        h = _structural_hash(kernel_code)
                        if h in seen_hashes:
                            stats["dup"] += 1
                            continue
                        seen_hashes.add(h)

                    collected.append(result)
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
                    stats["ok"] += 1

            elapsed = time.monotonic() - start
            op_counts = Counter(r.get("operator") for r in collected)
            print(f"  Collected: {len(collected)} | "
                  f"Tested: {stats['tested']} | Passed: {stats['passed']} | "
                  f"Dup: {stats['dup']} | OK: {stats['ok']}")
            print(f"  Per-op: {dict(op_counts)}")
            print(f"  Elapsed: {elapsed:.0f}s")
            print()

    elapsed = time.monotonic() - start
    op_counts = Counter(r.get("operator") for r in collected)
    print(f"\n=== DONE ===")
    print(f"Total: {len(collected)} samples in {elapsed:.0f}s")
    print(f"Per-op: {dict(op_counts)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
