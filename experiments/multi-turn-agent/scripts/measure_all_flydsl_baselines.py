#!/usr/bin/env python3
"""Measure real FlyDSL baseline kernel performance using each task's perf_runner.

Runs inside aiter-baseline-bench container which has:
  - Full aiter package (flydsl, MLIR, etc.)
  - /home/danyzhan mount for accessing artifacts

For each of the 50 flydsl tasks:
  1. cd into the task's initial/ directory
  2. Run: python3 scripts/perf_runner.py kernel.py <operator>
  3. Parse "Perf: X.XXXX ms" from output
  4. Record measured baseline

Usage:
  docker exec -e HIP_VISIBLE_DEVICES=2 aiter-baseline-bench \
    python3 /home/danyzhan/Lumen-RL/experiments/multi-tune-agent/scripts/measure_all_flydsl_baselines.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ARTIFACTS = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")
OUTPUT = Path("/home/danyzhan/agent-phase2-held-out-aiter/receipts/flydsl_real_baselines.json")
OVERRIDE_OUTPUT = Path("/home/danyzhan/agent-phase2-held-out-aiter/receipts/baseline_overrides.json")

TIMEOUT = 600  # 10 minutes per task


def get_flydsl_tasks():
    tasks = []
    for task_dir in sorted(ARTIFACTS.iterdir()):
        if not task_dir.is_dir():
            continue
        cfg_path = task_dir / "initial" / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("language") == "flydsl":
            tasks.append((task_dir.name, task_dir, cfg))
    return tasks


def measure_task(task_id, task_dir, cfg):
    initial = task_dir / "initial"
    kernel_py = initial / "kernel.py"
    perf_runner = initial / "scripts" / "perf_runner.py"
    operator = cfg.get("operator", "unknown")

    if not kernel_py.exists():
        return 0.0, "kernel.py missing"
    if not perf_runner.exists():
        return 0.0, "perf_runner.py missing"

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        r = subprocess.run(
            [sys.executable, str(perf_runner), str(kernel_py), operator],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=str(initial),
            env=env,
        )
        output = r.stdout + "\n" + r.stderr
        perf_re = re.compile(r'Perf:\s*([0-9]+(?:\.[0-9]+)?)\s*ms')
        match = perf_re.search(output)
        if match:
            ms = float(match.group(1))
            return ms, None
        return 0.0, f"no Perf line in output (rc={r.returncode}): {output[-500:]}"
    except subprocess.TimeoutExpired:
        return 0.0, f"timeout ({TIMEOUT}s)"
    except Exception as e:
        return 0.0, f"exception: {e}"


def main():
    tasks = get_flydsl_tasks()
    print(f"Found {len(tasks)} flydsl tasks\n")

    results = {}
    success_count = 0

    for i, (task_id, task_dir, cfg) in enumerate(tasks, 1):
        operator = cfg.get("operator", "unknown")
        cases = cfg.get("benchmark_shapes", [])
        case_id = cases[0].get("case_id", task_id) if cases else task_id
        old_baseline = cfg.get("baseline_ms", {})
        old_ms = list(old_baseline.values())[0] if old_baseline else 0.0

        print(f"[{i}/{len(tasks)}] {task_id}")
        print(f"  operator={operator}, case_id={case_id}, old_baseline={old_ms:.6f} ms")

        ms, err = measure_task(task_id, task_dir, cfg)

        if ms > 0:
            success_count += 1
            ratio = old_ms / ms if old_ms > 0 and ms > 0 else 0
            print(f"  MEASURED: {ms:.6f} ms (old={old_ms:.6f}, ratio={ratio:.2f}x)")
        else:
            print(f"  FAILED: {err}")

        results[task_id] = {
            "task_id": task_id,
            "case_id": case_id,
            "operator": operator,
            "old_baseline_ms": old_ms,
            "measured_ms": ms,
            "error": err,
            "ratio_old_new": old_ms / ms if ms > 0 and old_ms > 0 else None,
        }

    # Write detailed results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to {OUTPUT}")

    # Build baseline_overrides update
    # Load existing overrides
    existing = {}
    if OVERRIDE_OUTPUT.exists():
        existing = json.loads(OVERRIDE_OUTPUT.read_text())

    # Update only flydsl tasks that got valid measurements
    updated = 0
    for r in results.values():
        if r["measured_ms"] > 0:
            case_id = r["case_id"]
            existing[case_id] = r["measured_ms"]
            updated += 1

    with open(OVERRIDE_OUTPUT, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Updated {updated} entries in baseline_overrides.json")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary: {success_count}/{len(tasks)} measured successfully")
    print(f"\nSuccessful measurements:")
    for r in sorted(results.values(), key=lambda x: x["operator"]):
        if r["measured_ms"] > 0:
            ratio = r["ratio_old_new"]
            flag = " *** SUSPICIOUS" if ratio and (ratio > 5 or ratio < 0.2) else ""
            print(f"  {r['task_id']}: {r['measured_ms']:.6f} ms (old={r['old_baseline_ms']:.6f}, ratio={ratio:.2f}x){flag}")

    print(f"\nFailed measurements:")
    for r in sorted(results.values(), key=lambda x: x["operator"]):
        if r["measured_ms"] == 0:
            print(f"  {r['task_id']}: {r['error'][:100]}")


if __name__ == "__main__":
    main()
