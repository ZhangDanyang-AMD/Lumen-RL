"""AITER Benchmark: run 50 turns once, extract 10/20/30/40/50 checkpoints.

5x faster than running separate 10/20/30/40/50 turn benchmarks.
"""

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import geometric_mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

AITER_HELD_OUT = Path("/home/danyzhan/held-out-benchmark-aiter")
VLLM_PORT = 8000
RESULTS_DIR = AITER_HELD_OUT / "receipts" / "multi-benchmark"
CHECKPOINTS = [10, 20, 30, 40, 50]
KEEP_RECENT = 4  # keep last N turn-pairs verbatim, compress the rest
MAX_CONTEXT_CHARS = 200_000  # compress when total message chars exceed this


def compress_context(messages, turn_history):
    """Compress old turns, keeping system + original kernel + summary + recent turns.

    messages: full message list [system, user(kernel), asst, user, asst, user, ...]
    turn_history: list of dicts with turn/stage/speedup/reward for each completed turn

    Returns compressed message list.
    """
    total_chars = sum(len(m["content"]) for m in messages)
    # Count turn pairs after the initial 2 messages (system + kernel)
    turn_pairs = (len(messages) - 2) // 2
    if total_chars < MAX_CONTEXT_CHARS and turn_pairs <= KEEP_RECENT:
        return messages

    # Always keep: messages[0] (system), messages[1] (original kernel)
    head = messages[:2]

    # Turn pairs: messages[2:] in (assistant, user) pairs
    pairs = []
    for i in range(2, len(messages) - 1, 2):
        pairs.append((messages[i], messages[i + 1] if i + 1 < len(messages) else None))

    if len(pairs) <= KEEP_RECENT:
        return messages

    # Compress old pairs into a summary
    old_pairs = pairs[:-KEEP_RECENT]
    recent_pairs = pairs[-KEEP_RECENT:]

    summary_lines = [f"Previous {len(old_pairs)} optimization attempts summary:"]
    for idx, (asst_msg, user_msg) in enumerate(old_pairs):
        h = turn_history[idx] if idx < len(turn_history) else {}
        stage = h.get("stage", "?")
        sp = h.get("speedup", 0)
        reward = h.get("reward", -1)
        if reward > -0.5:
            summary_lines.append(f"  Turn {idx}: correct, speedup={sp:.2f}x")
        elif reward > -1.0:
            summary_lines.append(f"  Turn {idx}: compiled but incorrect")
        else:
            summary_lines.append(f"  Turn {idx}: {stage} error")

    best_idx = -1
    best_sp = 0
    for idx, h in enumerate(turn_history[:len(old_pairs)]):
        if h.get("speedup", 0) > best_sp:
            best_sp = h["speedup"]
            best_idx = idx
    if best_sp > 0:
        summary_lines.append(f"Best so far: turn {best_idx} with {best_sp:.2f}x speedup")

    compressed = head + [{"role": "user", "content": "\n".join(summary_lines)}]
    # Re-add recent pairs
    for asst_msg, user_msg in recent_pairs:
        compressed.append(asst_msg)
        if user_msg:
            compressed.append(user_msg)

    return compressed


def strip_autotune(src: str) -> str:
    """Remove @triton.autotune(...) decorators from kernel source for prompt.

    The model sees clean kernel code without autotune configs.
    Eval uses the full kernel with autotune for fair benchmarking.
    """
    import re
    return re.sub(
        r"@triton\.autotune\(\n(?:.*\n)*?\)\n(?=@triton\.jit)",
        "",
        src
    )


def load_baselines():
    path = AITER_HELD_OUT / "receipts" / "aiter_baseline_results.json"
    return json.load(open(path))


def run_one_task(client, backend, task_info, model_name, gpu_id):
    from sandbox.reward import eval_single

    tid = task_info["task"]
    bl_ms = task_info["perf_ms"]
    td = AITER_HELD_OUT / "artifacts" / "kernel" / tid / "initial"
    family = tid.split("-")[2] if len(tid.split("-")) > 2 else ""

    src = (td / "kernel.py").read_text()
    meta_path = td / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    frozen_input = json.dumps({
        "input": {
            "contract": {
                "architecture": "gfx942",
                "operator": family,
                "language": "triton",
                "target_lane": "triton_gfx942",
            },
            "parent_source": {"kernel.py": src},
            "baseline": {"commands": {}, "geomean_ms": bl_ms},
            "direction": {
                "instructions": f"Optimize this {family} Triton kernel for AMD MI300X (gfx942). Maximize speedup while maintaining correctness.",
                "strategy": "performance",
            },
            "profile": None,
            "error_feedback": None,
        },
        "task_type": "direction_conditioned",
    })

    messages = [
        {"role": "system", "content": "You are an expert GPU kernel engineer. Given a kernel and optimization direction, produce a unified diff patch that improves performance on the target architecture."},
        {"role": "user", "content": frozen_input},
    ]

    turn_results = []
    best_speedup = 0.0
    compile_ok = False
    correct_ok = False

    for turn in range(50):
        ctx = compress_context(messages, turn_results)
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=ctx,
                max_tokens=32768, temperature=0.0,
            )
            response = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    {tid} turn {turn} API error: {type(e).__name__}: {str(e)[:200]}", flush=True)
            break

        try:
            result = eval_single(backend, response, tid, td, bl_ms, family, gpu_id)
        except Exception as e:
            print(f"    {tid} turn {turn} eval error: {type(e).__name__}: {str(e)[:200]}", flush=True)
            result = {"reward": -1.0, "stage": "eval_error", "error": str(e)[:500]}
        speedup = result.get("speedup", 0)
        stage = result.get("stage", "fail")

        if result["reward"] > -1.0:
            compile_ok = True
        if result["reward"] > -0.5:
            correct_ok = True
        if speedup > best_speedup:
            best_speedup = speedup

        turn_results.append({
            "turn": turn, "stage": stage, "speedup": speedup,
            "reward": result["reward"], "compiled": result["reward"] > -1.0,
            "correct": result["reward"] > -0.5,
            "best_speedup_so_far": best_speedup,
        })

        if speedup >= 1.5:
            break

        if result["reward"] > -0.5:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Correct but speedup={speedup:.2f}x vs autotuned baseline. Optimize further."})
        elif result["reward"] > -1.0:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Compiled but correctness failed. Fix it."})
        else:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Compile error: {result.get('error','')[:500]}\nFix and retry."})

    return {
        "task_id": tid, "family": family, "baseline_ms": bl_ms,
        "total_turns": len(turn_results),
        "best_speedup": best_speedup,
        "compiled": compile_ok, "correct": correct_ok,
        "turn_results": turn_results,
    }


def extract_checkpoint(results, max_turns):
    """Extract metrics as if we stopped at max_turns."""
    n = max(1, len(results))
    compiled = 0
    correct = 0
    fast12 = 0
    speedups = []

    for r in results:
        turns = r["turn_results"][:max_turns]
        if not turns:
            continue
        best_sp = max((t["best_speedup_so_far"] for t in turns), default=0)
        any_compiled = any(t["compiled"] for t in turns)
        any_correct = any(t["correct"] for t in turns)

        if any_compiled:
            compiled += 1
        if any_correct:
            correct += 1
        if best_sp >= 1.2:
            fast12 += 1
        if best_sp > 0:
            speedups.append(best_sp)

    geo = geometric_mean(speedups) if speedups else 0.0
    return {
        "max_turns": max_turns, "tasks": n,
        "compiled": compiled, "correct": correct, "fast_1.2": fast12,
        "compile_rate": round(compiled / n, 3),
        "correct_rate": round(correct / n, 3),
        "fast12_rate": round(fast12 / n, 3),
        "speedup_geomean": round(geo, 4),
        "avg_turns": round(sum(min(len(r["turn_results"]), max_turns) for r in results) / n, 1),
    }


def run_model(model_name, baselines, gpu_ids):
    from openai import OpenAI
    from sandbox import get_backend

    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy", timeout=900.0)
    backend = get_backend("geak")
    passing = [r for r in baselines if r["passed"]]

    print(f"\n=== {model_name} × 50 turns ({len(passing)} tasks) ===", flush=True)
    results = []

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {}
        for i, t in enumerate(passing):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            f = pool.submit(run_one_task, client, backend, t, model_name, gpu_id)
            futures[f] = i

        for f in as_completed(futures):
            r = f.result()
            sp = r["best_speedup"]
            tag = "FAST" if sp >= 1.2 else ("OK" if sp > 0 else "FAIL")
            print(f"  [{futures[f]+1}/{len(passing)}] {r['task_id'][:40]} sp={sp:.2f}x t={r['total_turns']} {tag}", flush=True)
            results.append(r)

    # Save full 50-turn results
    out_dir = RESULTS_DIR / f"{model_name}-50turns"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Extract checkpoints
    for cp in CHECKPOINTS:
        summary = extract_checkpoint(results, cp)
        summary["model"] = model_name
        cp_dir = RESULTS_DIR / f"{model_name}-{cp}turns"
        cp_dir.mkdir(parents=True, exist_ok=True)
        with open(cp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  {model_name} @{cp}t: compile={summary['compile_rate']:.0%} correct={summary['correct_rate']:.0%} Fast@1.2={summary['fast12_rate']:.0%} geo={summary['speedup_geomean']:.3f}x", flush=True)

    return results


MODEL_PATHS = {
    "base": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "sft2": "/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-full2000-2epoch-merged",
    "sft4": "/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-full2000-4epoch-merged",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model label: base, sft2, or sft4")
    args = parser.parse_args()
    print(f"Model: {args.model} -> {MODEL_PATHS.get(args.model, args.model)}", flush=True)

    gpu_ids = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "1,2,3,4,5,6,7").split(",")]
    baselines = load_baselines()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_model(args.model, baselines, gpu_ids)


if __name__ == "__main__":
    main()
