"""Benchmark: AITER baseline + 20-turn multi-tune optimization.

Runs held-out-benchmark-aiter tasks with multi-turn agent loop.
Compares base, SFT-2e, SFT-4e models.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

AITER_HELD_OUT = Path("/home/danyzhan/held-out-benchmark-aiter")
VLLM_PORT = 8000
MAX_TURNS = 20


def verify_baselines(gpu_ids):
    """Verify AITER baselines compile+correct+perf on all tasks."""
    tasks_path = AITER_HELD_OUT / "tasks" / "kernel.jsonl"
    with open(tasks_path) as f:
        tasks = [json.loads(l) for l in f]

    results = []
    for i, task in enumerate(tasks):
        tid = task["task_id"]
        td = AITER_HELD_OUT / "artifacts" / "kernel" / tid / "initial"
        if not (td / "kernel.py").exists():
            continue

        gpu_id = gpu_ids[i % len(gpu_ids)]
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        # HIP tasks need longer timeout for JIT compilation
        timeout = 300 if "hip" in tid else 120

        passed = True
        perf_ms = 0.0
        for mode in ["compile", "correctness", "performance"]:
            try:
                r = subprocess.run(
                    ["python3", "scripts/task_runner.py", mode],
                    cwd=str(td), capture_output=True, text=True, timeout=timeout, env=env,
                )
                if r.returncode != 0:
                    passed = False
                    break
                if mode == "performance":
                    m = re.search(r"Perf:\s+([\d.]+)\s+ms", r.stdout)
                    perf_ms = float(m.group(1)) if m else 0
            except Exception:
                passed = False
                break

        results.append({"task": tid, "passed": passed, "perf_ms": perf_ms})
        status = f"OK {perf_ms:.4f}ms" if passed else "FAIL"
        print(f"  [{i+1}/{len(tasks)}] {tid[:50]} {status}")

    return results


def run_benchmark(model_name, model_label, baseline_results, gpu_ids, max_turns=20):
    """Run multi-turn benchmark for one model."""
    from openai import OpenAI
    from sandbox import get_backend
    from sandbox.reward import eval_single

    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy", timeout=300.0)
    backend = get_backend("geak")

    passing = [r for r in baseline_results if r["passed"]]
    results = []

    def _run_one(i, task_info):
        tid = task_info["task"]
        bl_ms = task_info["perf_ms"]
        td = AITER_HELD_OUT / "artifacts" / "kernel" / tid / "initial"
        family = tid.split("-")[2] if len(tid.split("-")) > 2 else ""
        gpu_id = gpu_ids[i % len(gpu_ids)]

        src = (td / "kernel.py").read_text()
        frozen_input = json.dumps({
            "input": {
                "contract": {"architecture": "gfx942", "operator": family},
                "parent_source": {"kernel.py": src},
                "baseline": {"commands": {}},
                "direction": None, "profile": None, "error_feedback": None,
            },
            "task_type": "cold_start",
        })

        messages = [
            {"role": "system", "content": "You are an expert GPU kernel engineer. Optimize the given kernel for AMD MI300X (gfx942). Return only a unified diff patch."},
            {"role": "user", "content": frozen_input},
        ]

        best_speedup = 0.0
        best_turn = -1
        turns_used = 0

        for turn in range(max_turns):
            turns_used = turn + 1
            try:
                resp = client.chat.completions.create(
                    model=model_name, messages=messages,
                    max_tokens=8192, temperature=0.0,
                )
                response = resp.choices[0].message.content or ""
            except Exception:
                break

            result = eval_single(backend, response, tid, td, bl_ms, family, gpu_id)
            speedup = result.get("speedup", 0)
            stage = result.get("stage", "fail")

            if speedup > best_speedup:
                best_speedup = speedup
                best_turn = turn

            if speedup >= 1.2:
                break

            if result["reward"] > -0.5:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Correct but speedup={speedup:.2f}x. Optimize further."})
            elif result["reward"] > -1.0:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Compiled but correctness failed. Fix it."})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Compile error: {result.get('error','')[:300]}\nFix and retry."})

        return {
            "task_id": tid, "family": family,
            "baseline_ms": bl_ms,
            "best_speedup": best_speedup,
            "best_turn": best_turn,
            "turns_used": turns_used,
            "compiled": best_speedup > 0 or best_turn >= 0,
        }

    print(f"\n=== Benchmarking {model_label} on {len(passing)} tasks (max {max_turns} turns) ===")
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {pool.submit(_run_one, i, t): i for i, t in enumerate(passing)}
        for f in as_completed(futures):
            r = f.result()
            sp = r["best_speedup"]
            tag = "FAST" if sp >= 1.2 else ("OK" if sp > 0 else "FAIL")
            print(f"  [{futures[f]+1}/{len(passing)}] {r['task_id'][:45]} sp={sp:.2f}x turns={r['turns_used']} {tag}")
            results.append(r)

    # Summary
    n = max(1, len(results))
    compiled = sum(1 for r in results if r["compiled"])
    correct = sum(1 for r in results if r["best_speedup"] > 0)
    fast12 = sum(1 for r in results if r["best_speedup"] >= 1.2)
    fast15 = sum(1 for r in results if r["best_speedup"] >= 1.5)
    avg_turns = sum(r["turns_used"] for r in results) / n
    speedups = [r["best_speedup"] for r in results if r["best_speedup"] > 0]
    from statistics import geometric_mean
    sp_geomean = geometric_mean(speedups) if speedups else 0

    summary = {
        "model": model_label, "tasks": len(results),
        "compiled": compiled, "correct": correct,
        "fast_1.2": fast12, "fast_1.5": fast15,
        "speedup_geomean": round(sp_geomean, 3),
        "avg_turns": round(avg_turns, 1),
    }
    print(f"\n{model_label} Summary:")
    print(f"  Compiled: {compiled}/{n} ({100*compiled//n}%)")
    print(f"  Correct: {correct}/{n} ({100*correct//n}%)")
    print(f"  Fast@1.2: {fast12}/{n} ({100*fast12//n}%)")
    print(f"  Speedup geomean: {sp_geomean:.3f}x")
    print(f"  Avg turns: {avg_turns:.1f}")

    return results, summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", default="sft2")
    parser.add_argument("--model-name", default="sft2")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "2,3,4,5,6,7").split(",")]

    # Step 1: Verify baselines
    print("=== Verifying AITER baselines ===")
    baseline_path = AITER_HELD_OUT / "receipts" / "aiter_baseline_results.json"
    if baseline_path.exists():
        baseline_results = json.load(open(baseline_path))
        passing = sum(1 for r in baseline_results if r["passed"])
        print(f"Loaded cached baselines: {passing} passing")
    else:
        baseline_results = verify_baselines(gpu_ids)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        passing = sum(1 for r in baseline_results if r["passed"])
        print(f"Verified: {passing}/{len(baseline_results)} passing")

    if args.verify_only:
        return

    # Step 2: Run benchmark
    results, summary = run_benchmark(
        args.model_name, args.model_label, baseline_results, gpu_ids, args.max_turns,
    )

    # Save results
    out_dir = AITER_HELD_OUT / "receipts" / f"benchmark-{args.model_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
