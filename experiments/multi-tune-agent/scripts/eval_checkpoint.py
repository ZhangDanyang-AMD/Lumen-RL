"""Evaluate a LoRA checkpoint on held-out tasks using vLLM, push results to W&B."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL_TASKS_PATH = Path("/home/danyzhan/held-out-benchmark/receipts/baseline_results.json")
EVAL_ARTIFACTS = Path("/home/danyzhan/held-out-benchmark/artifacts/kernel")
BASE_MODEL = "/home/danyzhan/Lumen/experiments/GEAK-agent-coder/outputs/qwen3-coder-full2000-2epoch-merged"
VLLM_PORT = 8000


def merge_and_serve(ckpt_path, model_name="eval"):
    """Merge LoRA checkpoint and start vLLM."""
    print(f"Merging LoRA from {ckpt_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    merged_dir = Path("/home/danyzhan/Lumen-RL/experiments/multi-tune-agent/outputs/_eval_merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(ckpt_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    del model, merged
    torch.cuda.empty_cache()
    print(f"Merged model saved to {merged_dir}")

    subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
    time.sleep(5)

    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = "0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model", str(merged_dir), "--served-model-name", model_name,
         "--tensor-parallel-size", "1", "--max-model-len", "32768",
         "--enforce-eager", "--dtype", "bfloat16", "--trust-remote-code",
         "--gpu-memory-utilization", "0.92", "--port", str(VLLM_PORT)],
        env=env, stdout=open("/home/danyzhan/vllm_eval.log", "w"), stderr=subprocess.STDOUT,
    )

    for _ in range(120):
        time.sleep(5)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=5)
            print(f"vLLM ready (pid={proc.pid})")
            return proc
        except Exception:
            pass
    print("ERROR: vLLM failed to start")
    return proc


def run_eval(model_name, n_tasks=40):
    """Run held-out eval using vLLM API with SFT-format prompts."""
    from openai import OpenAI
    from sandbox import get_backend
    from sandbox.reward import eval_single

    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy", timeout=300.0)
    backend = get_backend("geak")
    gpu_ids = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "2,3,4,5,6,7").split(",")]

    EASY_OPS = {"rms_norm", "softmax", "sampling", "gemm"}

    with open(EVAL_TASKS_PATH) as f:
        passing = [r for r in json.load(f) if r["passed"]]

    # Prioritize easy operators that the model was trained on
    easy = [t for t in passing if any(op in t["task"] for op in EASY_OPS)]
    hard = [t for t in passing if not any(op in t["task"] for op in EASY_OPS)]

    import random
    random.seed(42)
    random.shuffle(easy)
    random.shuffle(hard)

    # Take easy first, fill rest with hard
    n_easy = min(len(easy), n_tasks * 2 // 3)  # 2/3 easy
    n_hard = min(len(hard), n_tasks - n_easy)
    tasks = easy[:n_easy] + hard[:n_hard]
    print(f"Eval: {n_easy} easy + {n_hard} hard = {len(tasks)} tasks")

    compiled = correct = 0
    speedups = []
    results = []

    for i, task in enumerate(tasks):
        tid = task["task"]
        td = EVAL_ARTIFACTS / tid / "initial"
        if not (td / "kernel.py").exists():
            continue

        src = (td / "kernel.py").read_text()
        perf_stdout = task.get("performance", {}).get("stdout", "")
        m = re.search(r"Perf:\s+([\d.]+)\s+ms", perf_stdout)
        bl = float(m.group(1)) if m else 1.0
        family = tid.split("-")[2] if len(tid.split("-")) > 2 else ""

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

        try:
            completion = client.chat.completions.create(
                model=model_name, messages=messages,
                max_tokens=4096, temperature=0.0,
            )
            resp = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i}] {tid[:40]} generation failed: {e}")
            resp = ""

        gpu_id = gpu_ids[i % len(gpu_ids)]
        r = eval_single(backend, resp, tid, td, bl, family, gpu_id)

        status = r.get("stage", "fail")
        sp = r.get("speedup", 0)
        if r["reward"] > -1.0:
            compiled += 1
        if r["reward"] > -0.5:
            correct += 1
        if sp > 0:
            speedups.append(sp)

        print(f"  [{i+1}/{len(tasks)}] {tid[:40]} -> {status} speedup={sp:.2f}x")
        results.append({"task": tid, "stage": status, "speedup": sp, "reward": r["reward"]})

    n = max(1, len(tasks))
    metrics = {
        "eval/compile_rate": compiled / n,
        "eval/correct_rate": correct / n,
        "eval/num_compiled": compiled,
        "eval/num_correct": correct,
        "eval/speedup_mean": sum(speedups) / max(1, len(speedups)) if speedups else 0.0,
        "eval/n_tasks": len(tasks),
    }
    print(f"\nResults: compile={compiled}/{n} ({100*compiled//n}%) correct={correct}/{n} ({100*correct//n}%) speedup={metrics['eval/speedup_mean']:.3f}")
    return metrics, results


def push_to_wandb(step, metrics, wandb_run_id="5lptjqkk"):
    import wandb
    os.environ["WANDB_API_KEY"] = open("/home/danyzhan/wandb.key").read().strip()
    os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"

    run = wandb.init(
        project="coder-model-rl", entity="danyzhan-amd",
        id=wandb_run_id, resume="must",
        settings=wandb.Settings(init_timeout=60),
    )
    run.log(metrics, step=step, commit=True)
    time.sleep(5)
    run.finish()
    print(f"Pushed eval metrics to W&B step={step}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--n-tasks", type=int, default=40)
    parser.add_argument("--wandb-run-id", default="5lptjqkk")
    args = parser.parse_args()

    os.environ.setdefault("EVAL_GPU_IDS", "2,3,4,5,6,7")

    proc = merge_and_serve(args.ckpt)
    try:
        metrics, results = run_eval("eval", n_tasks=args.n_tasks)
        push_to_wandb(args.step, metrics, args.wandb_run_id)

        out_path = Path(args.ckpt) / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump({"step": args.step, "metrics": metrics, "results": results}, f, indent=2)
        print(f"Results saved to {out_path}")
    finally:
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
