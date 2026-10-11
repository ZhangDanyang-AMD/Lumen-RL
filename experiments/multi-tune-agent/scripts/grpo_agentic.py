"""Agentic multi-turn GRPO training for kernel optimization.

Improvements over grpo_multiturn.py:
  1. KL penalty against reference model (prevents catastrophic forgetting)
  2. Mixed sampling from all operators (no sequential curriculum)
  3. Replay buffer of best trajectories (reinforces successes)
  4. Profiling reward via rocprof hardware counters (prevents lazy optimisation)
  5. Robust reward hacking detection (Dr. Kernel patterns)

Architecture:
    GPU 0: vLLM server for generation (on-policy sync every step)
    GPU 1: LoRA training model
    GPU 2-7: sandbox kernel evaluation + rocprof profiling
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grpo_ag")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

EVAL_TASKS_PATH = Path("/home/danyzhan/held-out-benchmark/receipts/baseline_results.json")
EVAL_ARTIFACTS = Path("/home/danyzhan/held-out-benchmark/artifacts/kernel")
SYNC_MODEL_DIR = Path("/home/danyzhan/Lumen-RL/experiments/multi-tune-agent/outputs/_vllm_live")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            gt = json.loads(row["reward_model"]["ground_truth"])
            row["_family"] = gt.get("family", "")
            row["_gt"] = gt
            prompts.append(row)
    logger.info("Loaded %d prompts", len(prompts))
    return prompts

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Keeps the top-K trajectories by best turn reward."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.buffer: list[tuple[float, tuple]] = []

    def add(self, prompt, turns, discounted_r):
        best = max((t["reward"] for t in turns), default=-2.0)
        if best <= -1.0:
            return
        entry = (best, (prompt, turns, discounted_r))
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, entry)
        elif best > self.buffer[0][0]:
            heapq.heapreplace(self.buffer, entry)

    def sample(self, n: int) -> list[tuple]:
        if not self.buffer:
            return []
        k = min(n, len(self.buffer))
        chosen = random.sample(self.buffer, k)
        return [entry[1] for entry in chosen]

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------
# vLLM interaction
# ---------------------------------------------------------------------------

def call_vllm(client, model_name, messages, temperature=1.0, max_tokens=4096):
    try:
        resp = client.chat.completions.create(
            model=model_name, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("vLLM call failed: %s", str(e)[:100])
        return ""

# ---------------------------------------------------------------------------
# Kernel evaluation with profiling
# ---------------------------------------------------------------------------

def eval_kernel(backend, response, task_dir, baseline_ms, family, gpu_id):
    try:
        from sandbox.reward import eval_single
        return eval_single(backend, response, "eval", task_dir, baseline_ms, family, gpu_id)
    except Exception as e:
        return {"reward": -1.0, "stage": "error", "error": str(e)[:100]}


def run_rocprof(workspace: Path, gpu_id: int, timeout: int = 120) -> str:
    """Run rocprof on the performance benchmark and return raw output."""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        proc = subprocess.run(
            ["rocprof", "--stats",
             "python3", "scripts/task_runner.py", "performance"],
            cwd=str(workspace),
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return proc.stdout + "\n" + proc.stderr
    except Exception:
        return ""


def eval_kernel_with_profiling(
    backend, response, task_dir, baseline_ms, family, gpu_id,
    baseline_prof_text: str = "",
):
    """Evaluate kernel and optionally collect rocprof profiling data."""
    result = eval_kernel(backend, response, task_dir, baseline_ms, family, gpu_id)

    if result.get("stage") == "success" and result.get("speedup", 0) > 0:
        try:
            from lumenrl.rewards.profiling_reward import (
                parse_rocprof_output, compute_profiling_reward,
            )
            candidate_text = run_rocprof(task_dir, gpu_id)
            baseline_metrics = parse_rocprof_output(baseline_prof_text)
            candidate_metrics = parse_rocprof_output(candidate_text)
            prof_reward = compute_profiling_reward(baseline_metrics, candidate_metrics)
            result["profiling_reward"] = prof_reward["total"]
            result["profiling_detail"] = prof_reward
        except Exception as e:
            result["profiling_reward"] = 0.0
            logger.debug("Profiling failed: %s", e)

    return result

# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------

def _format_feedback(result):
    stage = result.get("stage", "")
    error = result.get("error", "")
    speedup = result.get("speedup", 0.0)

    if stage == "success" or result["reward"] > 0:
        return (
            f"Evaluation result: PASSED\n"
            f"- Compile: OK\n- Correctness: OK\n"
            f"- Speedup: {speedup:.3f}x vs baseline\n"
            f"The kernel is correct. To improve further, focus on optimizing "
            f"memory access patterns, occupancy, and instruction throughput."
        )
    elif stage == "correctness":
        return (
            f"Evaluation result: CORRECTNESS FAILED\n"
            f"- Compile: OK\n- Correctness: FAILED\n"
            f"Error output:\n```\n{error}\n```\n"
            f"The kernel compiles but produces incorrect results. "
            f"Check numerical precision, boundary conditions, and data layout."
        )
    elif stage == "compile":
        return (
            f"Evaluation result: COMPILE FAILED\n"
            f"Error output:\n```\n{error}\n```\n"
            f"Fix the compilation error and return the corrected kernel."
        )
    elif stage == "patch_apply" or stage == "patch_apply_syntax":
        return (
            f"Evaluation result: PATCH APPLY FAILED\n"
            f"The patch could not be applied to the source file. "
            f"Return a complete kernel implementation instead of a patch."
        )
    else:
        return f"Evaluation result: FAILED at stage '{stage}'\nError: {error}"

# ---------------------------------------------------------------------------
# Reward hacking detection (robust version)
# ---------------------------------------------------------------------------

def _detect_reward_hacking(response, result):
    if result["reward"] <= 0:
        return False
    code = response.lower()
    if "pass" in code and code.count("pass") > code.count("def "):
        return True
    if ("return input" in code or "return x" in code) and len(code) < 300:
        return True
    if result.get("speedup", 0) > 10.0:
        return True
    if "@triton.jit" in code and code.count("tl.") < 2:
        return True
    lines = [l.strip() for l in response.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 5 and result.get("speedup", 0) > 2.0:
        return True
    return False

# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------

def rollout_multiturn(
    client, model_name, prompt, backend, gpu_id,
    max_turns=8, temperature=1.0, use_profiling=False,
):
    gt = prompt["_gt"]
    task_dir = Path(gt["task_dir"])
    baseline_ms = gt.get("baseline_ms", 1.0)
    family = gt.get("family", "")
    messages = list(prompt["prompt"])

    turns = []
    for turn_idx in range(max_turns):
        response = call_vllm(client, model_name, messages, temperature)
        if not response:
            turns.append({"response": "", "reward": -1.0, "stage": "empty", "feedback": ""})
            break

        if use_profiling:
            result = eval_kernel_with_profiling(
                backend, response, task_dir, baseline_ms, family, gpu_id)
        else:
            result = eval_kernel(backend, response, task_dir, baseline_ms, family, gpu_id)

        reward = result["reward"]
        stage = result.get("stage", "")
        speedup = result.get("speedup", 0.0)
        prof_r = result.get("profiling_reward", 0.0)

        if use_profiling and prof_r != 0.0:
            reward = reward + 0.3 * prof_r

        if _detect_reward_hacking(response, result):
            reward = -1.0
            stage = "hacking"

        feedback = _format_feedback(result)
        turns.append({
            "response": response, "reward": reward, "stage": stage,
            "feedback": feedback, "speedup": speedup, "turn": turn_idx,
            "profiling_reward": prof_r,
        })

        if reward > 0 and speedup > 1.0:
            break

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": feedback})

    return turns

# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def discounted_rewards(turns, gamma=0.4):
    n = len(turns)
    rewards = [t["reward"] for t in turns]
    discounted = []
    for t in range(n):
        r = 0.0
        for i in range(t, n):
            r += (gamma ** (i - t)) * rewards[i]
        discounted.append(r)
    return discounted


def trloo_advantages(all_rewards, group_size):
    advantages = []
    for i in range(0, len(all_rewards), group_size):
        group = all_rewards[i:i + group_size]
        n = len(group)
        if n <= 1:
            advantages.extend([0.0] * n)
            continue
        group_sum = sum(group)
        for j, r in enumerate(group):
            baseline = (group_sum - r) / (n - 1)
            advantages.append(r - baseline)
    std = max(1e-8, (sum(a**2 for a in advantages) / max(1, len(advantages))) ** 0.5)
    return [a / std for a in advantages]

# ---------------------------------------------------------------------------
# Loss with KL penalty
# ---------------------------------------------------------------------------

def compute_reference_logprobs(
    ref_model, tokenizer, context: str, response: str, device: str,
    max_ctx=4096, max_resp=2048,
) -> torch.Tensor | None:
    """Compute per-token log-probs under the reference model (frozen)."""
    full_text = context + "\n" + response
    enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                    max_length=max_ctx + max_resp).to(device)
    input_ids = enc["input_ids"]
    ctx_enc = tokenizer(context, truncation=True, max_length=max_ctx)
    ctx_len = len(ctx_enc["input_ids"])
    if input_ids.shape[1] <= ctx_len:
        return None

    with torch.no_grad():
        outputs = ref_model(input_ids=input_ids)
        logits = outputs.logits[:, ctx_len - 1:-1, :]
        labels = input_ids[:, ctx_len:]
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    del outputs, logits, log_probs
    return token_lp.detach()


def compute_loss_with_kl(
    model, ref_model, tokenizer, trajectories, device,
    group_size=8, max_resp_tokens=2048, kl_coeff=0.02,
):
    """TRLOO loss with per-turn credit assignment + KL penalty."""
    model.gradient_checkpointing_enable()
    total_pg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_kl = torch.tensor(0.0, device=device)
    n_samples = 0

    all_discounted = [dr for _, _, drs in trajectories for dr in drs]
    advantages = trloo_advantages(all_discounted, group_size)

    adv_idx = 0
    for prompt, turns, _ in trajectories:
        system_text = prompt["prompt"][0].get("content", "")
        user_text = prompt["prompt"][1].get("content", "")
        context = system_text + "\n" + user_text

        for turn_info in turns:
            adv = advantages[adv_idx]
            adv_idx += 1

            if abs(adv) < 1e-8 or not turn_info["response"].strip():
                continue

            full_text = context + "\n" + turn_info["response"]
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=4096 + max_resp_tokens,
            ).to(device)
            input_ids = enc["input_ids"]

            ctx_enc = tokenizer(context, truncation=True, max_length=4096)
            ctx_len = len(ctx_enc["input_ids"])
            if input_ids.shape[1] <= ctx_len:
                continue

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, ctx_len - 1:-1, :]
            labels = input_ids[:, ctx_len:]
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            pg_loss = -(adv * token_lp.mean())
            total_pg_loss = total_pg_loss + pg_loss

            if kl_coeff > 0 and ref_model is not None:
                ref_lp = compute_reference_logprobs(
                    ref_model, tokenizer, context, turn_info["response"], device)
                if ref_lp is not None:
                    min_len = min(token_lp.shape[1], ref_lp.shape[1])
                    kl_div = (token_lp[:, :min_len] - ref_lp[:, :min_len]).mean()
                    total_kl = total_kl + kl_div

            n_samples += 1
            context = full_text + "\n" + turn_info.get("feedback", "")

            del outputs, logits, log_probs, token_lp
            torch.cuda.empty_cache()

    model.gradient_checkpointing_disable()

    if n_samples == 0:
        return total_pg_loss, 0.0

    avg_pg = total_pg_loss / n_samples
    avg_kl = total_kl / n_samples
    total_loss = avg_pg + kl_coeff * avg_kl
    return total_loss, float(avg_kl.item())

# ---------------------------------------------------------------------------
# Online eval
# ---------------------------------------------------------------------------

def run_online_eval(model, tokenizer, device, output_dir, step, wb_run, n_tasks=40):
    logger.info("Step %d: online eval on %d tasks...", step, n_tasks)
    if not EVAL_TASKS_PATH.exists():
        return

    EASY_EVAL = {"rms_norm", "softmax", "sampling", "gemm"}
    with open(EVAL_TASKS_PATH) as f:
        passing = [r for r in json.load(f) if r["passed"]]
    easy = [t for t in passing if any(op in t["task"] for op in EASY_EVAL)]
    hard = [t for t in passing if not any(op in t["task"] for op in EASY_EVAL)]
    random.shuffle(easy)
    random.shuffle(hard)
    n_easy = min(len(easy), n_tasks * 2 // 3)
    tasks = easy[:n_easy] + hard[:n_tasks - n_easy]

    try:
        from sandbox import get_backend
        from sandbox.reward import eval_single
        backend = get_backend("geak")
    except ImportError:
        return

    gpu_ids = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "2,3,4,5,6,7").split(",")]
    compiled = correct = 0
    speedups = []

    model.eval()
    from openai import OpenAI
    eval_client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy", timeout=120.0)

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
            completion = eval_client.chat.completions.create(
                model="sft2", messages=messages,
                max_tokens=4096, temperature=0.0,
            )
            resp = completion.choices[0].message.content or ""
        except Exception:
            resp = ""

        r = eval_single(backend, resp, tid, td, bl, family, gpu_ids[i % len(gpu_ids)])
        if r["reward"] > -1.0: compiled += 1
        if r["reward"] > -0.5: correct += 1
        if r.get("speedup", 0) > 0: speedups.append(r["speedup"])

    model.train()
    n = max(1, len(tasks))
    metrics = {"eval/compile": compiled/n, "eval/correct": correct/n,
               "eval/speedup": sum(speedups)/max(1,len(speedups)) if speedups else 0}
    logger.info("Eval: compile=%d/%d correct=%d/%d speedup=%.3f",
                compiled, n, correct, n, metrics["eval/speedup"])
    with open(output_dir / "eval.jsonl", "a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")
    log_wandb(wb_run, step, metrics)

# ---------------------------------------------------------------------------
# On-policy sync
# ---------------------------------------------------------------------------

def sync_policy_to_vllm(model, tokenizer, model_name, step, lora_rank=32):
    import subprocess as sp
    logger.info("Step %d: syncing LoRA weights to vLLM...", step)

    merged_dir = SYNC_MODEL_DIR
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(merged, LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.0, task_type="CAUSAL_LM",
    ))

    sp.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
    time.sleep(5)

    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = "0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = sp.Popen(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model", str(merged_dir), "--served-model-name", model_name,
         "--tensor-parallel-size", "1", "--max-model-len", "32768",
         "--enforce-eager", "--dtype", "bfloat16", "--trust-remote-code",
         "--gpu-memory-utilization", "0.92", "--port", "8000"],
        env=env,
        stdout=open("/home/danyzhan/vllm_live.log", "w"),
        stderr=subprocess.STDOUT,
    )

    for _ in range(120):
        time.sleep(5)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            logger.info("vLLM restarted with updated weights (pid=%d)", proc.pid)
            return model
        except Exception:
            pass

    logger.error("vLLM failed to restart after policy sync!")
    return model

# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(args, output_dir):
    try:
        import wandb
        key = Path("/home/danyzhan/wandb.key").read_text().strip()
        os.environ["WANDB_API_KEY"] = key
        os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"
        run = wandb.init(
            project="coder-model-rl", entity="danyzhan-amd",
            name=f"agentic-grpo-{args.tag}",
            config={
                "model": args.model_path,
                "algo": "GRPO-Agentic-OnPolicy",
                "advantage": "TRLOO",
                "credit_assignment": f"Kevin gamma={args.gamma}",
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "max_turns": args.max_turns,
                "gamma": args.gamma,
                "lr": args.lr,
                "lora_rank": args.lora_rank,
                "temperature": args.temperature,
                "kl_coeff": args.kl_coeff,
                "replay_frac": args.replay_frac,
                "profiling_reward": args.profiling_reward,
                "sampling": "mixed (all operators)",
                "eval_interval": 5,
                "eval_tasks": 40,
                "on_policy_sync": "merge_lora + restart_vllm every step",
                "hardware": "8x MI308X (gfx942)",
            },
            settings=wandb.Settings(init_timeout=120),
        )
        logger.info("W&B: %s", run.url)
        return run
    except Exception as e:
        logger.warning("W&B failed: %s", e)
        return None


def log_wandb(run, step, metrics, flush=False):
    if run is None: return
    try:
        run.log(metrics, step=step, commit=True)
        if flush:
            time.sleep(10)
    except Exception as e:
        logger.warning("W&B log failed: %s", e)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic GRPO kernel training")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", default="sft2")
    parser.add_argument("--dataset", default="/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/rl_prompts.jsonl")
    parser.add_argument("--output-dir", default="/home/danyzhan/Lumen-RL/experiments/multi-tune-agent/outputs")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--num-steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--kl-coeff", type=float, default=0.02)
    parser.add_argument("--replay-frac", type=float, default=0.25)
    parser.add_argument("--profiling-reward", action="store_true", default=True)
    parser.add_argument("--no-profiling-reward", dest="profiling_reward", action="store_false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"rl-ag-{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"

    start_step = 0
    resume_ckpt = None
    if args.resume and log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            start_step = last["step"] + 1
            for s in range(last["step"], -1, -1):
                c = output_dir / f"step-{s+1}"
                if c.exists() and ((c/"adapter_model.safetensors").exists() or (c/"adapter_model.bin").exists()):
                    resume_ckpt = c; break
            logger.info("Resume step %d, ckpt=%s", start_step, resume_ckpt)

    os.environ.setdefault("EVAL_GPU_IDS", "2,3,4,5,6,7")
    device = f"cuda:{args.gpu}"
    eval_gpu_ids = [int(g) for g in os.environ["EVAL_GPU_IDS"].split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from openai import OpenAI

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if resume_ckpt:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, str(resume_ckpt), is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank*2,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.0, task_type="CAUSAL_LM"))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LoRA: %d trainable (%.2f%%)", trainable, 100*trainable/sum(p.numel() for p in model.parameters()))

    # Reference model for KL (frozen copy of LoRA-merged weights, stays on same GPU)
    ref_model = None
    if args.kl_coeff > 0:
        logger.info("Loading reference model for KL penalty (kl_coeff=%.3f)", args.kl_coeff)
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        logger.info("Reference model loaded (frozen)")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.1)
    client = OpenAI(base_url=args.api_base, api_key="dummy", timeout=300.0)
    all_prompts = load_prompts(args.dataset)

    try:
        from sandbox import get_backend
        backend = get_backend("geak")
    except ImportError:
        from sandbox.backends.geak.backend import GEAKBackend
        backend = GEAKBackend()

    replay_buf = ReplayBuffer(max_size=200)
    wb_run = None if args.no_wandb else init_wandb(args, output_dir)

    logger.info(
        "Agentic GRPO: steps=%d batch=%d group=%d turns=%d gamma=%.1f "
        "kl=%.3f replay=%.0f%% profiling=%s mixed-sampling",
        args.num_steps, args.batch_size, args.group_size, args.max_turns,
        args.gamma, args.kl_coeff, args.replay_frac * 100, args.profiling_reward,
    )

    for step in range(start_step, args.num_steps):
        step_start = time.time()

        # --- Mixed sampling: random from ALL prompts ---
        batch_prompts = random.sample(all_prompts, min(args.batch_size, len(all_prompts)))
        families_in_batch = set(p["_family"] for p in batch_prompts)

        logger.info("Step %d [mixed %s]: rolling out %d prompts x %d groups x %d turns...",
                     step, ",".join(sorted(families_in_batch)),
                     len(batch_prompts), args.group_size, args.max_turns)

        all_trajectories = []
        total_turns = 0
        total_compiled = 0
        total_correct = 0
        total_speedup_count = 0
        best_speedup = 0.0
        total_prof_reward = 0.0

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_one_rollout(prompt, gi, gpu_id):
            turns = rollout_multiturn(
                client, args.model_name, prompt, backend, gpu_id,
                max_turns=args.max_turns, temperature=args.temperature,
                use_profiling=args.profiling_reward,
            )
            dr = discounted_rewards(turns, gamma=args.gamma)
            return prompt, turns, dr

        n_parallel = len(eval_gpu_ids)
        futures = []
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            for pi, prompt in enumerate(batch_prompts):
                for gi in range(args.group_size):
                    gpu_id = eval_gpu_ids[(pi * args.group_size + gi) % len(eval_gpu_ids)]
                    futures.append(pool.submit(_run_one_rollout, prompt, gi, gpu_id))

            for i, future in enumerate(as_completed(futures)):
                try:
                    prompt, turns, dr = future.result()
                    all_trajectories.append((prompt, turns, dr))
                    replay_buf.add(prompt, turns, dr)
                    for t in turns:
                        total_turns += 1
                        if t["reward"] > -1.0: total_compiled += 1
                        if t["reward"] > -0.5: total_correct += 1
                        if t.get("speedup", 0) > 0: total_speedup_count += 1
                        best_speedup = max(best_speedup, t.get("speedup", 0))
                        total_prof_reward += t.get("profiling_reward", 0.0)
                except Exception as e:
                    logger.error("Rollout failed: %s", e)

                if (i + 1) % n_parallel == 0:
                    logger.info("  Completed %d/%d rollouts, compiled=%d correct=%d best=%.3fx",
                                i+1, len(futures), total_compiled, total_correct, best_speedup)

        # --- Replay buffer mixing ---
        n_replay = int(len(all_trajectories) * args.replay_frac)
        replay_trajs = replay_buf.sample(n_replay)
        if replay_trajs:
            all_trajectories.extend(replay_trajs)
            logger.info("  Mixed in %d replay trajectories (buf=%d)", len(replay_trajs), len(replay_buf))

        n_turns = max(1, total_turns)
        compile_rate = total_compiled / n_turns
        correct_rate = total_correct / n_turns
        speedup_rate = total_speedup_count / n_turns
        mean_reward = sum(dr for _, _, drs in all_trajectories for dr in drs) / max(1, sum(len(drs) for _, _, drs in all_trajectories))

        logger.info(
            "Step %d: turns=%d compile=%.0f%% correct=%.0f%% speedup=%.0f%% best=%.3fx reward=%.3f",
            step, total_turns, compile_rate*100, correct_rate*100, speedup_rate*100,
            best_speedup, mean_reward,
        )

        # --- Compute loss with KL penalty ---
        logger.info("Step %d: computing loss (kl=%.3f)...", step, args.kl_coeff)
        model.train()
        optimizer.zero_grad()
        loss, kl_div = compute_loss_with_kl(
            model, ref_model, tokenizer, all_trajectories, device,
            group_size=args.group_size, kl_coeff=args.kl_coeff,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_time = time.time() - step_start
        loss_val = loss.item()
        logger.info("Step %d: loss=%.4f kl=%.4f time=%.0fs", step, loss_val, kl_div, step_time)

        step_log = {
            "step": step, "loss": loss_val, "kl_divergence": kl_div,
            "mean_reward": mean_reward,
            "compile_rate": compile_rate, "correct_rate": correct_rate,
            "speedup_rate": speedup_rate, "best_speedup": best_speedup,
            "total_turns": total_turns, "step_time_s": step_time,
            "replay_size": len(replay_buf), "replay_mixed": len(replay_trajs) if replay_trajs else 0,
            "profiling_reward_mean": total_prof_reward / max(1, total_compiled),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(step_log) + "\n")

        avg_turns = total_turns / max(1, len(all_trajectories) - len(replay_trajs) if replay_trajs else len(all_trajectories))

        log_wandb(wb_run, step, {
            "train/reward": mean_reward,
            "train/compile_rate": compile_rate,
            "train/correct_rate": correct_rate,
            "train/speedup_rate": speedup_rate,
            "train/best_speedup": best_speedup,
            "train/loss": loss_val,
            "train/kl_divergence": kl_div,
            "train/total_turns": total_turns,
            "train/avg_turns": avg_turns,
            "train/num_compiled": total_compiled,
            "train/num_correct": total_correct,
            "train/replay_size": len(replay_buf),
            "train/profiling_reward": total_prof_reward / max(1, total_compiled),
            "train/step_time_s": step_time,
        }, flush=True)

        # On-policy sync
        model = sync_policy_to_vllm(model, tokenizer, args.model_name, step, lora_rank=args.lora_rank)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.1)

        if (step + 1) % 5 == 0:
            ckpt_dir = output_dir / f"step-{step+1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Checkpoint: %s", ckpt_dir)

            all_ckpts = sorted(
                [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
                key=lambda d: int(d.name.split("-")[1]))
            while len(all_ckpts) > 2:
                shutil.rmtree(all_ckpts.pop(0))

            run_online_eval(model, tokenizer, device, output_dir, step+1, wb_run)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Done. Final: %s", final_dir)
    if wb_run: wb_run.finish()


if __name__ == "__main__":
    main()
