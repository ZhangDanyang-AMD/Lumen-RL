"""AITER Benchmark using GEAK agent flow (tool-calling multi-turn).

Uses GEAKToolEnvironment + OpenAIModelBackend for proper agent optimization.
The model uses tools (list_files, read_file, write_file, evaluate) rather
than simple chat-based patch generation.

Usage:
    python scripts/run_aiter_benchmark_geak.py --model-label sft2 --model-name sft2
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import geometric_mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AITER_HELD_OUT = Path("/home/danyzhan/held-out-benchmark-aiter")
VLLM_PORT = 8000

SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer optimizing Triton kernels for AMD MI300X (gfx942). "
    "You have access to a GEAK sandbox tool with actions: list_files, read_file, write_file, evaluate. "
    "Your workflow: 1) Read the current kernel.py, 2) Understand the algorithm, "
    "3) Write an optimized version, 4) Evaluate (compile -> correctness -> performance). "
    "If evaluation fails, read the error, fix, and retry. "
    "Goal: maximize speedup over the baseline while maintaining correctness.\n\n"
    "RULES:\n"
    "- You MUST keep the kernel as a Triton kernel (@triton.jit). Do NOT replace it with PyTorch ops.\n"
    "- You MUST preserve @triton.autotune if present. You may modify or add autotune configs.\n"
    "- Focus on algorithmic optimizations: memory access patterns, tiling, vectorization, "
    "shared memory usage, loop unrolling, instruction-level parallelism.\n"
    "- Do NOT simply change BLOCK_SIZE or num_warps — the baseline is already autotuned for these."
)


def build_cases_yaml(baseline_results: list[dict]) -> Path:
    """Build a YAML task catalog from held-out baseline results for GEAKToolEnvironment."""
    import yaml

    passing = [r for r in baseline_results if r["passed"]]
    tasks = []
    for r in passing:
        tid = r["task"]
        td = AITER_HELD_OUT / "artifacts" / "kernel" / tid / "initial"
        if not td.exists():
            continue
        tasks.append({
            "id": tid,
            "type": "aiter_generated",
            "kernel_path": str(td),
            "direction": f"Optimize the {tid.split('-')[2]} kernel for maximum speedup on MI300X.",
        })

    out = AITER_HELD_OUT / "receipts" / "benchmark_cases.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump({"tasks": tasks}, open(out, "w"), default_flow_style=False)
    print(f"Built {len(tasks)} benchmark cases -> {out}")
    return out


def run_single_task(
    env, backend, task_id: str, max_turns: int = 20,
) -> dict:
    """Run GEAK agent flow on a single task."""
    from multi_tune_agent.runtime import ToolAgentLoop, OpenAIModelBackend
    from multi_tune_agent.geak_tool import GEAKStatefulTool

    tool = GEAKStatefulTool(env)
    loop = ToolAgentLoop(backend, tool, max_assistant_turns=max_turns)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Optimize the kernel in task {task_id}. "
         f"Start by reading kernel.py, then write an optimized version and evaluate it."},
    ]

    try:
        output = asyncio.run(loop.run(
            messages,
            create_kwargs={
                "case_id": task_id,
                "role": "benchmark_engineer",
                "establish_baseline": True,
            },
        ))

        # Extract best speedup from events
        best_speedup = 0.0
        compiled = False
        correct = False
        for event in output.events:
            if event.get("type") == "tool":
                result = event.get("result", {})
                eval_data = result.get("evaluation", {})
                if eval_data.get("compiled"):
                    compiled = True
                if eval_data.get("correct"):
                    correct = True
                    sp = float(eval_data.get("speedup_geomean", 0))
                    best_speedup = max(best_speedup, sp)

        return {
            "task_id": task_id,
            "family": task_id.split("-")[2] if len(task_id.split("-")) > 2 else "",
            "lane": "triton" if "triton" in task_id else "hip",
            "compiled": compiled,
            "correct": correct,
            "best_speedup": best_speedup,
            "turns_used": output.metrics.assistant_turns,
            "tool_calls": output.metrics.tool_calls,
            "model_seconds": output.metrics.model_seconds,
            "tool_seconds": output.metrics.tool_seconds,
            "reward": output.reward_score,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "family": task_id.split("-")[2] if len(task_id.split("-")) > 2 else "",
            "lane": "triton" if "triton" in task_id else "hip",
            "compiled": False,
            "correct": False,
            "best_speedup": 0.0,
            "turns_used": 0,
            "tool_calls": 0,
            "error": str(e)[:200],
        }


def run_benchmark(model_label, model_name, baseline_results, max_turns=20):
    """Run full GEAK-flow benchmark for one model."""
    from multi_tune_agent.config import MultiTuneConfig
    from multi_tune_agent.geak_tool import GEAKToolEnvironment
    from multi_tune_agent.runtime import OpenAIModelBackend

    cases_yaml = build_cases_yaml(baseline_results)
    gpu_ids = os.environ.get("EVAL_GPU_IDS", "1,2,3,4,5,6,7")

    config = MultiTuneConfig(
        geak_root=Path("/home/danyzhan/GEAK"),
        cases_path=cases_yaml,
        trajectory_root=AITER_HELD_OUT / "receipts" / f"geak-flow-{model_label}",
        base_url=f"http://127.0.0.1:{VLLM_PORT}/v1",
        model=model_name,
        gpu_ids=gpu_ids,
        command_timeout=300,
        baseline_repeats=3,
        engineer_tool_rounds=max_turns,
        request_max_tokens=32768,
        keep_sessions=True,
    )

    env = GEAKToolEnvironment(config)
    backend = OpenAIModelBackend(
        base_url=config.base_url,
        model=model_name,
        max_tokens=config.request_max_tokens,
        temperature=0.0,
        timeout=900.0,
    )

    passing = [r for r in baseline_results if r["passed"]]
    results = []
    num_gpus = len(gpu_ids.split(","))

    print(f"\n=== GEAK-flow benchmark: {model_label} | {len(passing)} tasks | max {max_turns} turns | {num_gpus} parallel workers ===", flush=True)

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {}
        for i, task_info in enumerate(passing):
            tid = task_info["task"]
            f = pool.submit(run_single_task, env, backend, tid, max_turns)
            futures[f] = (i, tid)

        for f in as_completed(futures):
            i, tid = futures[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"task_id": tid, "family": tid.split("-")[2] if len(tid.split("-")) > 2 else "",
                     "compiled": False, "correct": False, "best_speedup": 0.0,
                     "turns_used": 0, "tool_calls": 0, "error": str(e)[:200]}
            sp = r["best_speedup"]
            tag = "FAST" if sp >= 1.2 else ("OK" if sp > 0 else "FAIL")
            turns = r["turns_used"]
            tools = r.get("tool_calls", 0)
            err = r.get("error", "")
            err_msg = f" err={err[:80]}" if err else ""
            print(f"  [{i+1}/{len(passing)}] {tid[:45]}... sp={sp:.2f}x turns={turns} tools={tools} {tag}{err_msg}", flush=True)
            results.append(r)

    # Summary
    n = max(1, len(results))
    compiled = sum(1 for r in results if r["compiled"])
    correct = sum(1 for r in results if r["correct"])
    fast12 = sum(1 for r in results if r["best_speedup"] >= 1.2)
    fast15 = sum(1 for r in results if r["best_speedup"] >= 1.5)
    speedups = [r["best_speedup"] for r in results if r["best_speedup"] > 0]
    sp_geomean = geometric_mean(speedups) if speedups else 0
    avg_turns = sum(r["turns_used"] for r in results) / n
    avg_tools = sum(r.get("tool_calls", 0) for r in results) / n

    # Per-operator breakdown
    op_stats = {}
    for r in results:
        fam = r["family"]
        if fam not in op_stats:
            op_stats[fam] = {"total": 0, "compiled": 0, "correct": 0, "fast12": 0, "speedups": []}
        op_stats[fam]["total"] += 1
        if r["compiled"]: op_stats[fam]["compiled"] += 1
        if r["correct"]: op_stats[fam]["correct"] += 1
        if r["best_speedup"] >= 1.2: op_stats[fam]["fast12"] += 1
        if r["best_speedup"] > 0: op_stats[fam]["speedups"].append(r["best_speedup"])

    summary = {
        "model": model_label,
        "tasks": len(results),
        "compiled": compiled,
        "correct": correct,
        "fast_1.2": fast12,
        "fast_1.5": fast15,
        "speedup_geomean": round(sp_geomean, 3),
        "avg_turns": round(avg_turns, 1),
        "avg_tool_calls": round(avg_tools, 1),
        "per_operator": {
            op: {
                "total": s["total"],
                "compiled": s["compiled"],
                "correct": s["correct"],
                "fast12": s["fast12"],
                "speedup_geomean": round(geometric_mean(s["speedups"]), 3) if s["speedups"] else 0,
            }
            for op, s in op_stats.items()
        },
    }

    print(f"\n{model_label} Summary:")
    print(f"  Compiled: {compiled}/{n} ({100*compiled//n}%)")
    print(f"  Correct: {correct}/{n} ({100*correct//n}%)")
    print(f"  Fast@1.2: {fast12}/{n} ({100*fast12//n}%)")
    print(f"  Speedup geomean: {sp_geomean:.3f}x")
    print(f"  Avg turns: {avg_turns:.1f}, Avg tool calls: {avg_tools:.1f}")
    print(f"\n  Per-operator:")
    for op, s in sorted(op_stats.items()):
        sp = geometric_mean(s["speedups"]) if s["speedups"] else 0
        print(f"    {op:20s} compile={s['compiled']}/{s['total']} correct={s['correct']}/{s['total']} fast12={s['fast12']} geomean={sp:.3f}x")

    # Save results
    out_dir = AITER_HELD_OUT / "receipts" / f"geak-benchmark-{model_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}")
    return results, summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", default="sft2")
    parser.add_argument("--model-name", default="sft2")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    # Load baseline results
    baseline_path = AITER_HELD_OUT / "receipts" / "aiter_baseline_results.json"
    if not baseline_path.exists():
        print("ERROR: Run baseline verification first")
        return
    baseline_results = json.load(open(baseline_path))
    passing = sum(1 for r in baseline_results if r["passed"])
    print(f"Loaded baselines: {passing} passing tasks")

    if args.verify_only:
        return

    run_benchmark(args.model_label, args.model_name, baseline_results, args.max_turns)


if __name__ == "__main__":
    main()
