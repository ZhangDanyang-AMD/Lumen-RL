#!/usr/bin/env python3
"""Run GEAKflow SFT collection using the full Director→TechLead→Engineer→Verifier→Integrator pipeline.

This script:
1. Reads training cases from sft-training-cases/sft_training_cases.yaml
2. Runs GEAKflow via multi_tune_agent.cli on each case with sft_enabled=true
3. Collects successful SFT trajectories into JSONL files

Usage:
    # Step 1: Setup training cases (only needs to run once)
    python3 scripts/setup_sft_training_cases.py --lang all --repeats 30

    # Step 2: Run GEAKflow collection
    python3 scripts/run_geakflow_sft_collection.py --workers 6 --gpu-ids 2,3,4,5,6,7

    # Or run specific language
    python3 scripts/run_geakflow_sft_collection.py --lang flydsl --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = ROOT / "sft-training-cases"
CONFIG_PATH = CASES_ROOT / "geakflow_sft_config.yaml"
OUTPUT_DIR = Path("/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP")

SFT_TASK_TYPES = [
    "direction_conditioned",
    "cold_start",
    "profile_guided",
    "error_recovery",
]


def detect_model(base_url: str) -> str:
    try:
        import requests
        resp = requests.get(f"{base_url}/models", timeout=10)
        return resp.json()["data"][0]["id"]
    except Exception:
        return "model"


def run_single_case(
    case_id: str,
    config_path: Path,
    sft_task_type: str,
    gpu_id: str,
    model: str,
    stream: bool = True,
) -> dict | None:
    """Run a single GEAKflow case via CLI subprocess."""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpu_id
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        sys.executable, "-m", "multi_tune_agent.cli",
        "--config", str(config_path),
        "run",
        "--case", case_id,
        "--sft-task-type", sft_task_type,
    ]
    if stream:
        cmd.append("--stream")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(ROOT),
            env=env,
        )
        if result.returncode == 0:
            try:
                summaries = json.loads(result.stdout.split("\n")[-2] if result.stdout.strip() else "{}")
                return summaries
            except (json.JSONDecodeError, IndexError):
                return {"status": "completed", "raw": result.stdout[-500:]}
        else:
            return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  ERROR [{case_id}]: {e}", file=sys.stderr)
        return None


def collect_sft_from_dataset(sft_dataset_root: Path) -> list[dict]:
    """Collect SFT samples from the GEAKflow SFT dataset output."""
    samples = []
    for jsonl_file in sorted(sft_dataset_root.rglob("*.jsonl")):
        for line in jsonl_file.read_text().splitlines():
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu-ids", default="2,3,4,5,6,7")
    parser.add_argument("--lang", default=None, choices=["triton", "flydsl"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    model = cfg.get("model") or detect_model(args.base_url)
    if not cfg.get("model"):
        cfg["model"] = model
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Model: {model}")

    cases_path = Path(cfg["cases_path"])
    cases_data = yaml.safe_load(cases_path.read_text())
    all_cases = cases_data.get("tasks", [])

    if args.lang:
        cases = [c for c in all_cases if f"-{args.lang}-" in c["id"]]
    else:
        cases = all_cases

    if args.max_cases:
        cases = cases[:args.max_cases]

    gpu_ids = args.gpu_ids.split(",")
    lang_label = args.lang or "all"
    output_path = args.output or str(OUTPUT_DIR / f"collected_geakflow_{lang_label}.jsonl")

    print(f"\nGEAKflow SFT Collection")
    print(f"  Cases: {len(cases)} ({lang_label})")
    print(f"  Workers: {args.workers}")
    print(f"  GPUs: {args.gpu_ids}")
    print(f"  Output: {output_path}")
    print()

    n = len(cases)
    completed = 0
    failed = 0
    start_all = time.monotonic()

    with open(output_path, "w") as out_file:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_case = {}
            for i, case in enumerate(cases):
                case_id = case["id"]
                gpu = gpu_ids[i % len(gpu_ids)]
                sft_task_type = SFT_TASK_TYPES[i % len(SFT_TASK_TYPES)]

                f = pool.submit(
                    run_single_case,
                    case_id, config_path,
                    sft_task_type, gpu, model,
                )
                future_to_case[f] = (case_id, sft_task_type)

            for i, f in enumerate(as_completed(future_to_case), 1):
                case_id, sft_task_type = future_to_case[f]
                try:
                    result = f.result()
                except Exception as e:
                    result = None
                    print(f"  [{i:4d}/{n}] {case_id}: EXCEPTION {e}", flush=True)

                if result:
                    completed += 1
                    print(
                        f"  [{i:4d}/{n}] {case_id} ({sft_task_type}) OK",
                        flush=True,
                    )
                else:
                    failed += 1
                    print(
                        f"  [{i:4d}/{n}] {case_id} ({sft_task_type}) FAIL",
                        flush=True,
                    )

                if i % 50 == 0:
                    elapsed = time.monotonic() - start_all
                    rate = i / elapsed * 3600
                    print(
                        f"  --- Progress: {i}/{n} | OK: {completed} FAIL: {failed} "
                        f"| Rate: {rate:.0f}/hr | ETA: {(n-i)/max(rate/3600,0.01)/60:.0f}min ---",
                        flush=True,
                    )

    elapsed = time.monotonic() - start_all

    sft_dataset_root = Path(cfg.get("sft_dataset_root", str(CASES_ROOT / "sft_dataset")))
    sft_samples = collect_sft_from_dataset(sft_dataset_root)

    if sft_samples:
        with open(output_path, "w") as f:
            for sample in sft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\nCollected {len(sft_samples)} SFT samples from GEAKflow dataset")
    else:
        traj_root = Path(cfg.get("trajectory_root", str(CASES_ROOT / "trajectories")))
        traj_samples = collect_sft_from_dataset(traj_root)
        if traj_samples:
            with open(output_path, "w") as f:
                for sample in traj_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"\nCollected {len(traj_samples)} trajectory samples")

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total cases: {n}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
