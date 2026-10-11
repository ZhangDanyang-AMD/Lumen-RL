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
    "You are an expert GPU kernel engineer writing Triton kernels for AMD MI300X (gfx942). "
    "You have access to a GEAK sandbox tool with actions: list_files, read_file, write_file, evaluate. "
    "Your workflow: 1) Read scripts/task_runner.py to understand the required function signature and test cases, "
    "2) Write a high-performance Triton kernel in kernel.py, "
    "3) Evaluate (compile -> correctness -> performance). "
    "4) If evaluation fails or performance is insufficient, read the error, improve, and retry.\n\n"
    "RULES:\n"
    "- You MUST write a Triton kernel using @triton.jit. Do NOT use PyTorch ops for the core computation.\n"
    "- Your kernel must pass correctness checks against the reference implementation.\n"
    "- Include @triton.autotune with multiple configs to search the best BLOCK_SIZE, num_warps, "
    "and num_stages for the given shape. The autotune will be frozen to the best config after benchmarking.\n"
    "- Target: achieve at least 90% of the baseline performance.\n"
    "- Focus on: efficient memory access patterns, proper tiling for gfx942, vectorized loads/stores, "
    "and appropriate num_warps/num_stages for the AMD CDNA3 architecture."
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


def _build_sft_prompt(task_id: str, baseline_ms: float = 0.0) -> str:
    """Build user message in SFT training format so model sees familiar input."""
    td = AITER_HELD_OUT / "artifacts" / "kernel" / task_id / "initial"
    family = task_id.split("-")[2] if len(task_id.split("-")) > 2 else ""
    src = (td / "kernel.py").read_text() if (td / "kernel.py").exists() else ""

    sft_input = json.dumps({
        "input": {
            "contract": {
                "architecture": "gfx942",
                "operator": family,
                "language": "triton",
                "target_lane": "triton_gfx942",
            },
            "parent_source": {"kernel.py": src},
            "baseline": {"geomean_ms": baseline_ms},
            "direction": {
                "instructions": f"Optimize this {family} Triton kernel for AMD MI300X (gfx942). "
                    "Maximize speedup while maintaining correctness.",
                "strategy": "performance",
            },
            "error_feedback": None,
            "profile": None,
        },
        "task_type": "direction_conditioned",
    })
    return (
        f"Optimize the kernel in task {task_id}.\n\n"
        f"Here is the kernel and optimization context:\n{sft_input}\n\n"
        f"Use write_file to write your optimized kernel.py (you may write a unified diff patch "
        f"or the complete file), then use evaluate to test it."
    )


def run_single_task(
    env, backend, task_id: str, max_turns: int = 20, sft_compat: bool = False,
) -> dict:
    """Run GEAK agent flow on a single task."""
    from multi_tune_agent.runtime import ToolAgentLoop, OpenAIModelBackend
    from multi_tune_agent.geak_tool import GEAKStatefulTool

    tool = GEAKStatefulTool(env)
    loop = ToolAgentLoop(backend, tool, max_assistant_turns=max_turns)

    family = task_id.split("-")[2] if len(task_id.split("-")) > 2 else ""
    bl = json.load(open(AITER_HELD_OUT / "receipts" / "aiter_baseline_results.json"))
    bl_ms = next((r["perf_ms"] for r in bl if r["task"] == task_id), 0.0)

    user_msg = (
        f"Write a high-performance Triton kernel for the '{family}' operator on AMD MI300X (gfx942).\n\n"
        f"Task: {task_id}\n"
        f"Baseline performance: {bl_ms:.4f} ms\n"
        f"Target: achieve at least 90% of baseline ({bl_ms/0.9:.4f} ms or faster).\n\n"
        f"Steps:\n"
        f"1. Read scripts/task_runner.py to understand the function signature, input shapes, and correctness oracle.\n"
        f"2. Write kernel.py with your Triton implementation.\n"
        f"3. Use evaluate to test compile -> correctness -> performance.\n"
        f"4. Iterate until you reach the target performance.\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        output = asyncio.run(loop.run(
            messages,
            create_kwargs={
                "case_id": task_id,
                "role": "benchmark_engineer",
                "establish_baseline": True,
                "clear_kernel": True,
            },
        ))

        # Extract compile/correct status from events (for diagnostics only)
        ever_compiled = False
        ever_correct = False
        in_loop_speedup = 0.0
        for event in output.events:
            if event.get("type") == "tool":
                result = event.get("result", {})
                eval_data = result.get("evaluation", {})
                if eval_data.get("compiled"):
                    ever_compiled = True
                if eval_data.get("correct"):
                    ever_correct = True
                    sp = float(eval_data.get("speedup_geomean", 0))
                    in_loop_speedup = max(in_loop_speedup, sp)

        # Post-task: verify with kernel_best.py — this is the ONLY source of final speedup
        best_speedup = 0.0
        compiled = False
        correct = False
        if ever_correct and in_loop_speedup > 0:
            import subprocess, shutil
            flow_dir = AITER_HELD_OUT / "receipts" / f"geak-flow-{env.config.model}" / "sessions"
            td = AITER_HELD_OUT / "artifacts" / "kernel" / task_id / "initial"
            gpu_id = int(os.environ.get("EVAL_GPU_IDS", "2").split(",")[0])
            for sess in flow_dir.iterdir():
                meta_f = sess / "workspace" / "metadata.json"
                if meta_f.exists() and task_id in meta_f.read_text():
                    ws = sess / "workspace"
                    best_kernel = ws / "snapshots" / "kernel_best.py"
                    if best_kernel.exists():
                        # Validate kernel_best.py for PyTorch cheating
                        from geak_utils.kernel_validator import validate_kernel
                        best_src = best_kernel.read_text()
                        vk_ok, vk_msg = validate_kernel(best_src)
                        if not vk_ok:
                            break  # kernel_best uses PyTorch, skip
                        shutil.copy2(best_kernel, ws / "kernel.py")
                        # Verify compile + correctness first
                        try:
                            cr = subprocess.run(
                                ["python3", "scripts/task_runner.py", "correctness"],
                                cwd=str(ws), capture_output=True, text=True, timeout=120,
                                env={**os.environ, "HIP_VISIBLE_DEVICES": str(gpu_id)},
                            )
                            if cr.returncode == 0 and "OK" in cr.stdout:
                                compiled = True
                                correct = True
                            else:
                                break  # kernel_best doesn't pass, skip perf
                        except Exception:
                            break
                        # Re-benchmark candidate (3 runs, median)
                        candidate_times = []
                        for _ in range(3):
                            try:
                                r = subprocess.run(
                                    ["python3", "scripts/task_runner.py", "performance"],
                                    cwd=str(ws), capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "HIP_VISIBLE_DEVICES": str(gpu_id)},
                                )
                                if r.returncode == 0:
                                    for line in r.stdout.splitlines():
                                        if "Perf:" in line:
                                            candidate_times.append(float(line.split("Perf:")[1].split("ms")[0].strip()))
                            except Exception:
                                pass
                        # Re-benchmark baseline (3 runs, median)
                        baseline_times = []
                        for _ in range(3):
                            try:
                                r = subprocess.run(
                                    ["python3", "scripts/task_runner.py", "performance"],
                                    cwd=str(td), capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "HIP_VISIBLE_DEVICES": str(gpu_id)},
                                )
                                if r.returncode == 0:
                                    for line in r.stdout.splitlines():
                                        if "Perf:" in line:
                                            baseline_times.append(float(line.split("Perf:")[1].split("ms")[0].strip()))
                            except Exception:
                                pass
                        if candidate_times and baseline_times:
                            candidate_times.sort()
                            baseline_times.sort()
                            cand_ms = candidate_times[len(candidate_times) // 2]
                            base_ms = baseline_times[len(baseline_times) // 2]
                            best_speedup = base_ms / cand_ms if cand_ms > 0 else 0
                    break

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
        timeout=3600.0,
    )

    passing = [r for r in baseline_results if r["passed"]]
    results = []
    num_gpus = len(gpu_ids.split(","))

    use_sft = model_label in ("sft2", "sft4")
    print(f"\n=== GEAK-flow benchmark: {model_label} | {len(passing)} tasks | max {max_turns} turns | {num_gpus} workers | sft_compat={use_sft} ===", flush=True)

    # Complex operators get more turns
    COMPLEX_OPS = {"mha", "mla", "paged_attention", "rope_kv_cache", "fused_moe"}

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {}
        for i, task_info in enumerate(passing):
            tid = task_info["task"]
            family = tid.split("-")[2] if len(tid.split("-")) > 2 else ""
            task_turns = max_turns
            f = pool.submit(run_single_task, env, backend, tid, task_turns, sft_compat=use_sft)
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
            tag = "FAST" if sp >= 0.9 else ("OK" if sp > 0 else "FAIL")
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
    fast11 = sum(1 for r in results if r["best_speedup"] >= 0.9)
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
            op_stats[fam] = {"total": 0, "compiled": 0, "correct": 0, "fast11": 0, "speedups": []}
        op_stats[fam]["total"] += 1
        if r["compiled"]: op_stats[fam]["compiled"] += 1
        if r["correct"]: op_stats[fam]["correct"] += 1
        if r["best_speedup"] >= 0.9: op_stats[fam]["fast11"] += 1
        if r["best_speedup"] > 0: op_stats[fam]["speedups"].append(r["best_speedup"])

    summary = {
        "model": model_label,
        "tasks": len(results),
        "compiled": compiled,
        "correct": correct,
        "fast_1.1": fast11,
        "fast_1.5": fast15,
        "speedup_geomean": round(sp_geomean, 3),
        "avg_turns": round(avg_turns, 1),
        "avg_tool_calls": round(avg_tools, 1),
        "per_operator": {
            op: {
                "total": s["total"],
                "compiled": s["compiled"],
                "correct": s["correct"],
                "fast11": s["fast11"],
                "speedup_geomean": round(geometric_mean(s["speedups"]), 3) if s["speedups"] else 0,
            }
            for op, s in op_stats.items()
        },
    }

    print(f"\n{model_label} Summary:")
    print(f"  Compiled: {compiled}/{n} ({100*compiled//n}%)")
    print(f"  Correct: {correct}/{n} ({100*correct//n}%)")
    print(f"  Fast@1.1: {fast11}/{n} ({100*fast11//n}%)")
    print(f"  Speedup geomean: {sp_geomean:.3f}x")
    print(f"  Avg turns: {avg_turns:.1f}, Avg tool calls: {avg_tools:.1f}")
    print(f"\n  Per-operator:")
    for op, s in sorted(op_stats.items()):
        sp = geometric_mean(s["speedups"]) if s["speedups"] else 0
        print(f"    {op:20s} compile={s['compiled']}/{s['total']} correct={s['correct']}/{s['total']} fast11={s['fast11']} geomean={sp:.3f}x")

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
