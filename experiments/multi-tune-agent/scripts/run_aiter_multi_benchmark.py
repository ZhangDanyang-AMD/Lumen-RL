"""AITER Multi-Turn × Multi-Model Benchmark.

Tests 10/20/30/40/50 turns × base/SFT-2e/SFT-4e.
For each (model, turns) pair, runs all 100 AITER tasks through GEAK agent flow.
Saves per-kernel results and summary comparison table.

Usage:
    python scripts/run_aiter_multi_benchmark.py --models sft2,sft4,base --turns 10,20,30,40,50
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import geometric_mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

AITER_HELD_OUT = Path("/home/danyzhan/held-out-benchmark-aiter")
VLLM_PORT = 8000
RESULTS_DIR = AITER_HELD_OUT / "receipts" / "multi-benchmark"


def load_baselines():
    path = AITER_HELD_OUT / "receipts" / "aiter_baseline_results.json"
    if not path.exists():
        raise FileNotFoundError("Run baseline verification first")
    return json.load(open(path))


def run_one_task(client, backend, task_info, model_name, max_turns, gpu_id):
    from sandbox.reward import eval_single

    tid = task_info["task"]
    bl_ms = task_info["perf_ms"]
    td = AITER_HELD_OUT / "artifacts" / "kernel" / tid / "initial"
    family = tid.split("-")[2] if len(tid.split("-")) > 2 else ""

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
    compile_ok = False
    correct_ok = False
    turn_results = []

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages,
                max_tokens=32768, temperature=0.0,
            )
            response = resp.choices[0].message.content or ""
        except Exception:
            break

        result = eval_single(backend, response, tid, td, bl_ms, family, gpu_id)
        speedup = result.get("speedup", 0)
        stage = result.get("stage", "fail")

        if result["reward"] > -1.0:
            compile_ok = True
        if result["reward"] > -0.5:
            correct_ok = True
        if speedup > best_speedup:
            best_speedup = speedup
            best_turn = turn

        turn_results.append({
            "turn": turn, "stage": stage, "speedup": speedup,
            "reward": result["reward"],
        })

        if speedup >= 1.5:
            break

        if result["reward"] > -0.5:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Correct but speedup={speedup:.2f}x. Optimize further for higher performance."})
        elif result["reward"] > -1.0:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Compiled but correctness failed. Fix the kernel."})
        else:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Compile error: {result.get('error','')[:500]}\nFix and retry."})

    return {
        "task_id": tid, "family": family, "baseline_ms": bl_ms,
        "best_speedup": best_speedup, "best_turn": best_turn,
        "turns_used": len(turn_results),
        "compiled": compile_ok, "correct": correct_ok,
        "turn_results": turn_results,
    }


def run_benchmark(model_name, model_label, baselines, max_turns, gpu_ids):
    from openai import OpenAI
    from sandbox import get_backend

    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy", timeout=300.0)
    backend = get_backend("geak")
    passing = [r for r in baselines if r["passed"]]

    print(f"\n=== {model_label} × {max_turns} turns ({len(passing)} tasks) ===")
    results = []

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {}
        for i, t in enumerate(passing):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            f = pool.submit(run_one_task, client, backend, t, model_name, max_turns, gpu_id)
            futures[f] = i

        for f in as_completed(futures):
            r = f.result()
            sp = r["best_speedup"]
            tag = "FAST" if sp >= 1.2 else ("OK" if sp > 0 else "FAIL")
            print(f"  [{futures[f]+1}/{len(passing)}] {r['task_id'][:40]} sp={sp:.2f}x t={r['turns_used']} {tag}")
            results.append(r)

    # Summary
    n = max(1, len(results))
    compiled = sum(1 for r in results if r["compiled"])
    correct = sum(1 for r in results if r["correct"])
    fast12 = sum(1 for r in results if r["best_speedup"] >= 1.2)
    fast15 = sum(1 for r in results if r["best_speedup"] >= 1.5)
    speedups = [r["best_speedup"] for r in results if r["best_speedup"] > 0]
    sp_geo = geometric_mean(speedups) if speedups else 0.0
    avg_turns = sum(r["turns_used"] for r in results) / n

    summary = {
        "model": model_label, "max_turns": max_turns,
        "tasks": n, "compiled": compiled, "correct": correct,
        "fast_1.2": fast12, "fast_1.5": fast15,
        "speedup_geomean": round(sp_geo, 4),
        "avg_turns": round(avg_turns, 1),
        "compile_rate": round(compiled / n, 3),
        "correct_rate": round(correct / n, 3),
        "fast12_rate": round(fast12 / n, 3),
    }

    # Per-operator breakdown
    by_op = defaultdict(lambda: {"total": 0, "compiled": 0, "correct": 0, "fast12": 0, "speedups": []})
    for r in results:
        op = by_op[r["family"]]
        op["total"] += 1
        if r["compiled"]: op["compiled"] += 1
        if r["correct"]: op["correct"] += 1
        if r["best_speedup"] >= 1.2: op["fast12"] += 1
        if r["best_speedup"] > 0: op["speedups"].append(r["best_speedup"])
    summary["per_operator"] = {
        k: {**{kk: vv for kk, vv in v.items() if kk != "speedups"},
            "speedup_geo": round(geometric_mean(v["speedups"]), 3) if v["speedups"] else 0}
        for k, v in by_op.items()
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="sft2", help="Comma-separated model names")
    parser.add_argument("--turns", default="10,20,30,40,50", help="Comma-separated turn counts")
    args = parser.parse_args()

    models = args.models.split(",")
    turn_counts = [int(t) for t in args.turns.split(",")]
    gpu_ids = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "1,2,3,4,5,6,7").split(",")]

    baselines = load_baselines()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model in models:
        for turns in turn_counts:
            results, summary = run_benchmark(model, model, baselines, turns, gpu_ids)

            # Save per-kernel results
            out_dir = RESULTS_DIR / f"{model}-{turns}turns"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            all_summaries.append(summary)
            print(f"\n{model} × {turns}t: compile={summary['compile_rate']:.0%} correct={summary['correct_rate']:.0%} Fast@1.2={summary['fast12_rate']:.0%} geomean={summary['speedup_geomean']:.3f}x")

    # Write comparison table
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print markdown table
    print("\n\n=== COMPARISON TABLE ===")
    header = "| Model |"
    for t in turn_counts:
        header += f" {t}t Fast@1.2 |"
    print(header)
    print("|" + "---|" * (len(turn_counts) + 1))
    for model in models:
        row = f"| {model} |"
        for t in turn_counts:
            s = next((s for s in all_summaries if s["model"] == model and s["max_turns"] == t), None)
            if s:
                row += f" {s['fast12_rate']:.0%} (geo={s['speedup_geomean']:.2f}x) |"
            else:
                row += " — |"
        print(row)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
