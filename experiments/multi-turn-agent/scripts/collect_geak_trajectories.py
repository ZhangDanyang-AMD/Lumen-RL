#!/usr/bin/env python3
"""Collect SFT training trajectories by running GEAK agent on training-split tasks.

Runs ToolAgentLoop on cloned training cases, captures successful trajectories
(compile + correct) as multi-turn SFT data.

Usage:
    # Collect FlyDSL trajectories
    python3 scripts/collect_geak_trajectories.py --lang flydsl --workers 6

    # Collect Triton trajectories
    python3 scripts/collect_geak_trajectories.py --lang triton --workers 6

    # Collect all (triton + flydsl)
    python3 scripts/collect_geak_trajectories.py --workers 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.geak_tool import GEAKStatefulTool, GEAKToolEnvironment
from multi_tune_agent.runtime import OpenAIModelBackend, ToolAgentLoop

HELD_OUT = Path("/home/danyzhan/agent-phase2-held-out-aiter")
TRAINING_CASES = ROOT / "sft-training-cases"
CASES_YAML = TRAINING_CASES / "sft_training_cases.yaml"
BASELINES = HELD_OUT / "receipts" / "baseline_overrides.json"
OUTPUT_DIR = Path("/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP")

SYSTEM_PROMPT_TRITON = """\
You are an expert GPU kernel engineer writing Triton kernels for AMD gfx950 (MI350X, 256 CUs).

You have access to a GEAK sandbox tool. Workflow:
1. read_file scripts/task_runner.py → understand required function signature and test cases
2. read_file config.yaml → see benchmark shape and best_config if available
3. write_file kernel.py → write complete Triton kernel with @triton.autotune configs
4. evaluate → compile, correctness check, performance measurement
5. If not passing or not fast enough, iterate

RULES:
- Write a complete Triton kernel using @triton.jit and @triton.autotune
- Target: speedup >= 1.0x vs baseline
- Use gfx950-appropriate tile sizes: BLOCK_M/N typically 64-256, BLOCK_K 32-128"""

SYSTEM_PROMPT_OPUS = """\
You are an expert GPU kernel engineer writing HIP C++ kernels for AMD gfx950 (MI350X, 256 CUs).

You have access to a GEAK sandbox tool. Workflow:
1. read_file scripts/task_runner.py → understand required function signature
2. read_file config.yaml → see benchmark shape
3. write_file kernel.cu → write complete HIP C++ kernel
4. evaluate → compile with hipcc, correctness check, performance measurement
5. Iterate if needed

RULES:
- Write complete HIP C++ kernel with __global__ function
- Target: speedup >= 1.0x vs baseline"""

SYSTEM_PROMPT_FLYDSL = """\
You are an expert FlyDSL kernel engineer for AMD gfx950 (MI350X, 256 CUs, 64KB LDS/CU).

You have access to a GEAK sandbox tool. Workflow:
1. read_file scripts/task_runner.py → understand build_*/launch_* function interface
2. read_file config.yaml → see operator, benchmark shape
3. write_file kernel.py → write FlyDSL kernel
4. evaluate → compile, correctness, performance
5. Iterate if needed

RULES:
- Use FlyDSL DSL: import flydsl.compiler as flyc, import flydsl.expr as fx
- @flyc.kernel / @flyc.jit / @flyc.amdgpu_kernel decorators
- ABSOLUTELY FORBIDDEN: @triton.jit, import triton, __global__ keyword
- Target: speedup >= 1.0x vs baseline"""


def get_system_prompt(language: str) -> str:
    if language == "opus":
        return SYSTEM_PROMPT_OPUS
    if language == "flydsl":
        return SYSTEM_PROMPT_FLYDSL
    return SYSTEM_PROMPT_TRITON


def run_and_collect(
    task_id: str,
    config: MultiTuneConfig,
    baseline: dict,
    max_turns: int,
    gpu_id: str,
    repeat_idx: int,
    kernel_path: str | None = None,
) -> dict | None:
    """Run one task and return the trajectory if successful."""
    import yaml

    backend = OpenAIModelBackend(
        base_url=config.base_url,
        model=config.model,
        timeout=config.request_timeout,
        max_tokens=config.request_max_tokens,
    )

    environment = GEAKToolEnvironment(config)
    tool = GEAKStatefulTool(environment)

    language = "triton"
    operator = "unknown"
    kernel_prompt = ""
    task_dir = Path(kernel_path) if kernel_path else TRAINING_CASES / "artifacts" / "kernel" / task_id / "initial"
    cfg_path = task_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        language = cfg.get("language", "triton")
        operator = cfg.get("operator", "unknown")
        shapes = cfg.get("benchmark_shapes", [])
        if shapes and isinstance(shapes[0], dict):
            kernel_prompt = shapes[0].get("kernel_prompt", "")

    system_prompt = get_system_prompt(language)

    bl_data = baseline.get(task_id, {})
    baseline_override = {
        "per_case_ms": bl_data.get("per_case_ms", {"default": 1.0}),
        "geomean_ms": bl_data.get("geomean_ms", 1.0),
    }

    kernel_name = ""
    if kernel_prompt and kernel_prompt.startswith("Kernel:"):
        kernel_name = kernel_prompt.split("(")[0].replace("Kernel:", "").strip()

    task_desc = f"You must write a **{operator}** kernel in {language} for AMD gfx950."
    if kernel_name:
        task_desc += f"\nThe specific kernel function to implement is: **{kernel_name}**"
    if kernel_prompt:
        task_desc += f"\n{kernel_prompt}"
    task_desc += (
        f"\nDo NOT write a different operator — the correctness test will check "
        f"that your kernel implements {operator} correctly."
        f"\nIMPORTANT: Read scripts/task_runner.py FIRST to understand the exact function signature, "
        f"then write_file your complete kernel implementation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_desc},
        {"role": "assistant", "content": f"I will write the {kernel_name or operator} kernel in {language}. Let me start by reading scripts/task_runner.py to understand the exact function signature required."},
    ]
    create_kwargs = {
        "case_id":            task_id,
        "role":               "engineer",
        "establish_baseline": False,
        "baseline_override":  baseline_override,
        "clear_kernel":       True,
    }

    loop = ToolAgentLoop(
        backend=backend,
        tool=tool,
        max_assistant_turns=max_turns,
        retain_session=True,
    )

    try:
        output = asyncio.run(loop.run(messages, create_kwargs=create_kwargs))
    except Exception as e:
        print(f"    ERROR [{task_id} r{repeat_idx}]: {e}", file=sys.stderr)
        return None

    sid = output.session_id if output else ""
    compiled = False
    correct = False
    best_speedup = 0.0

    try:
        state = environment.get(sid)
        best_speedup = state.best_speedup if state.best_speedup > 0 else 0.0
        last_eval = state.last_evaluation or {}
        compiled = last_eval.get("compiled", False)
        correct = last_eval.get("correct", False)
        if not correct:
            best_speedup = 0.0
    except Exception:
        pass

    if sid:
        try:
            tool.release(sid)
        except Exception:
            pass

    if not compiled:
        return None

    trajectory_messages = output.messages if output else []
    if len(trajectory_messages) < 4:
        return None

    has_tool_calls = any(
        m.get("tool_calls") for m in trajectory_messages if m.get("role") == "assistant"
    )
    if not has_tool_calls:
        return None

    return {
        "messages": trajectory_messages,
        "task_type": "complete_implementation",
        "operator": operator,
        "data_source": "geak_agent_collected",
        "schema_version": "messages_multiturn_v1",
        "language": language,
        "task_id": task_id,
        "repeat_idx": repeat_idx,
        "compiled": compiled,
        "correct": correct,
        "best_speedup": best_speedup,
        "turns_used": output.metrics.assistant_turns if output and output.metrics else 0,
        "tool_calls_count": output.metrics.tool_calls if output and output.metrics else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=500,
                        help="Max turns per task")
    parser.add_argument("--gpu-ids", default="2,3,4,5,6,7")
    parser.add_argument("--lang", default=None, help="triton|opus|flydsl or None for all")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases to run (default: all)")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--docker", default="aiter-mi400-regress",
                        help="Docker container for GPU commands (empty=no docker)")
    args = parser.parse_args()

    if not args.model:
        try:
            import requests as _req
            _models = _req.get(f"{args.base_url}/models", timeout=10).json()
            args.model = _models["data"][0]["id"]
            print(f"  Model: {args.model}")
        except Exception:
            args.model = "model"

    import yaml
    cases_data = yaml.safe_load(CASES_YAML.read_text())
    all_tasks = cases_data.get("tasks", [])
    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}

    tasks = all_tasks
    if args.lang:
        tasks = [t for t in tasks if f"-{args.lang}-" in t["id"]]

    task_list = []
    for t in tasks:
        task_list.append((t["id"], 0, t.get("kernel_path", "")))

    if args.max_cases:
        task_list = task_list[:args.max_cases]

    n = len(task_list)
    lang_label = args.lang or "all"
    output_path = args.output or str(OUTPUT_DIR / f"collected_{lang_label}.jsonl")

    print(f"GEAK SFT Collection: {lang_label}")
    print(f"  Tasks: {n} total runs (training-split cases)")
    print(f"  Workers: {args.workers} | Max turns: {args.max_turns}")
    print(f"  Output: {output_path}")
    print()

    config = MultiTuneConfig(
        geak_root=Path("/home/danyzhan/GEAK"),
        cases_path=CASES_YAML,
        trajectory_root=TRAINING_CASES / "trajectories",
        base_url=args.base_url,
        model=args.model,
        gpu_ids=args.gpu_ids,
        request_timeout=600,
        request_max_tokens=8192,
        command_timeout=300,
        baseline_repeats=1,
        max_rounds=1,
        engineers_per_round=1,
        engineer_tool_rounds=args.max_turns,
        integrator_tool_rounds=10,
        candidate_floor=0.8,
        min_improvement=0.01,
        target_speedup=1.1,
        keep_sessions=True,
        bootstrap_enabled=False,
        bootstrap_auto_promote=False,
        aiter_root=Path("/home/danyzhan/aiter"),
        sft_enabled=False,
        docker_container=args.docker,
    )

    gpu_ids = args.gpu_ids.split(",")
    collected = []
    n_compiled = 0
    n_correct = 0
    start_all = time.monotonic()

    out_file = open(output_path, "w")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_task = {}
        for i, (task_id, repeat_idx, kpath) in enumerate(task_list):
            cloned_from = task_id.replace("train-gfx950-", "heldout-gfx950-").rsplit("-", 1)[0]
            bl = baselines.get(cloned_from, baselines.get(task_id, {}))
            f = pool.submit(
                run_and_collect,
                task_id, config,
                bl,
                args.max_turns,
                gpu_ids[i % len(gpu_ids)],
                repeat_idx,
                kpath or None,
            )
            future_to_task[f] = (task_id, repeat_idx)

        for i, f in enumerate(as_completed(future_to_task), 1):
            task_id, repeat_idx = future_to_task[f]
            try:
                result = f.result()
            except Exception as e:
                result = None
                print(f"  [{i:4d}/{n}] {task_id} r{repeat_idx}: EXCEPTION {e}", flush=True)

            if result:
                compiled = result.get("compiled", False)
                correct = result.get("correct", False)
                speedup = result.get("best_speedup", 0)
                turns = result.get("turns_used", 0)
                lang = result.get("language", "?")
                n_compiled += 1 if compiled else 0
                n_correct += 1 if correct else 0

                sft_sample = {
                    "messages": result["messages"],
                    "task_type": result["task_type"],
                    "operator": result["operator"],
                    "data_source": result["data_source"],
                    "schema_version": result["schema_version"],
                }
                out_file.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                out_file.flush()
                collected.append(result)

                print(
                    f"  [{i:4d}/{n}] [{lang:6s}] {task_id[:40]} r{repeat_idx} "
                    f"compile={compiled} correct={correct} speedup={speedup:.2f}x "
                    f"turns={turns} msgs={len(result['messages'])} ✓",
                    flush=True,
                )
            else:
                print(
                    f"  [{i:4d}/{n}] {task_id[:40]} r{repeat_idx} SKIP (no compile)",
                    flush=True,
                )

            if i % 100 == 0:
                elapsed = time.monotonic() - start_all
                rate = i / elapsed * 3600
                print(
                    f"  --- Progress: {i}/{n} | Collected: {len(collected)} "
                    f"({len(collected)/max(i,1)*100:.0f}%) | "
                    f"Rate: {rate:.0f}/hr | ETA: {(n-i)/max(rate/3600,0.01)/60:.0f}min ---",
                    flush=True,
                )

    out_file.close()
    elapsed = time.monotonic() - start_all

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total runs: {n}")
    print(f"  Collected: {len(collected)} ({len(collected)/max(n,1)*100:.0f}%)")
    print(f"  Compiled: {n_compiled}")
    print(f"  Correct: {n_correct}")
    print(f"  Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"  Output: {output_path}")

    by_lang = Counter()
    by_op = Counter()
    for r in collected:
        by_lang[r.get("language", "?")] += 1
        by_op[r.get("operator", "?")] += 1
    print(f"\n  By language:")
    for l, c in by_lang.most_common():
        print(f"    {l}: {c}")
    print(f"\n  By operator (top 10):")
    for o, c in by_op.most_common(10):
        print(f"    {o}: {c}")


if __name__ == "__main__":
    main()
