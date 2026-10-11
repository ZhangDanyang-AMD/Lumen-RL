"""Standalone single-turn GRPO training for kernel optimization.

Improvements over v1:
- Curriculum learning: easy ops first, then medium, then hard
- Larger batch (16 prompts x 4 gen = 64 samples/step)
- Higher temperature (1.3) for more exploration
- W&B real-time logging
- Checkpoint every 10 steps with LoRA resume
- Periodic held-out eval every 20 steps

Usage:
    python scripts/grpo_train.py --model-path <path> --tag sft2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grpo_train")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

EASY_OPS = {"rms_norm", "softmax", "silu_and_mul", "fused_silu_mul", "fused_mul_add",
            "fused_add_rms_norm", "dynamic_per_token_quant", "dynamic_per_tensor_quant",
            "static_per_tensor_quant"}
MEDIUM_OPS = {"gemm", "scaled_quant_gemm", "gemm_activation", "batched_gemm",
              "gated_gemm", "knn", "sampling", "blockscale_gemm"}
HARD_OPS = {"mha", "mla", "paged_attention", "fused_moe", "rope_kv_cache", "all_reduce"}


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            gt = json.loads(row["reward_model"]["ground_truth"])
            row["_family"] = gt.get("family", "")
            prompts.append(row)
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


def curriculum_filter(prompts: list[dict], step: int,
                      stage1_end: int = 30, stage2_end: int = 60) -> list[dict]:
    if step < stage1_end:
        pool = [p for p in prompts if p["_family"] in EASY_OPS]
        stage = "easy"
    elif step < stage2_end:
        pool = [p for p in prompts if p["_family"] in EASY_OPS | MEDIUM_OPS]
        stage = "easy+medium"
    else:
        pool = prompts
        stage = "all"
    if not pool:
        pool = prompts
    logger.info("Curriculum stage=%s pool=%d/%d", stage, len(pool), len(prompts))
    return pool


def generate_responses(
    prompts: list[dict],
    api_base: str,
    model_name: str,
    num_generations: int = 4,
    temperature: float = 1.3,
    max_tokens: int = 8192,
) -> list[list[str]]:
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key="dummy", timeout=300.0)
    all_responses = []

    for i, prompt in enumerate(prompts):
        messages = prompt["prompt"]
        responses = []
        for _ in range(num_generations):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                responses.append(resp.choices[0].message.content or "")
            except Exception as e:
                logger.warning("Gen failed prompt %d: %s", i, str(e)[:100])
                responses.append("")
        all_responses.append(responses)
        if (i + 1) % 4 == 0:
            logger.info("Generated %d/%d prompts", i + 1, len(prompts))

    return all_responses


def compute_rewards(prompts: list[dict], responses: list[list[str]]) -> list[list[float]]:
    try:
        from sandbox.reward import sandbox_reward_batch
    except ImportError:
        from rewards.kernel_reward import kernel_reward_batch as sandbox_reward_batch

    all_rewards = []
    for i, (prompt, resps) in enumerate(zip(prompts, responses)):
        gt = prompt["reward_model"]["ground_truth"]
        gts = [gt] * len(resps)
        reward_tensor = sandbox_reward_batch(resps, gts, backend_name="geak")
        all_rewards.append(reward_tensor.tolist())
    return all_rewards


def grpo_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = max(math.sqrt(var), 1e-8)
    return [(r - mean) / std for r in rewards]


def compute_grpo_loss(
    model, tokenizer, prompts, responses, rewards,
    device="cuda:0", max_response_tokens=2048,
) -> torch.Tensor:
    model.gradient_checkpointing_enable()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_samples = 0

    for prompt, resps, rews in zip(prompts, responses, rewards):
        advantages = grpo_advantages(rews)
        for resp, adv in zip(resps, advantages):
            if abs(adv) < 1e-8 or not resp.strip():
                continue

            prompt_text = "\n".join(m.get("content", "") for m in prompt["prompt"])
            full_text = prompt_text + "\n" + resp
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=4096 + max_response_tokens,
            ).to(device)
            input_ids = enc["input_ids"]

            prompt_enc = tokenizer(prompt_text, truncation=True, max_length=4096)
            prompt_len = len(prompt_enc["input_ids"])
            if input_ids.shape[1] <= prompt_len:
                continue

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, prompt_len - 1:-1, :]
            labels = input_ids[:, prompt_len:]
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            pg_loss = -(adv * token_log_probs.mean())
            total_loss = total_loss + pg_loss
            n_samples += 1

            del outputs, logits, log_probs, token_log_probs
            torch.cuda.empty_cache()

    model.gradient_checkpointing_disable()
    if n_samples > 0:
        return total_loss / n_samples
    return total_loss


EVAL_TASKS_PATH = Path("/home/danyzhan/held-out-benchmark/receipts/baseline_results.json")
EVAL_ARTIFACTS = Path("/home/danyzhan/held-out-benchmark/artifacts/kernel")
EVAL_SAMPLE_SIZE = 20


def run_online_eval(model, tokenizer, device, output_dir, step, wb_run):
    """Quick held-out eval using the training model directly."""
    logger.info("Step %d: running online eval on %d held-out tasks...", step, EVAL_SAMPLE_SIZE)
    eval_start = time.time()

    if not EVAL_TASKS_PATH.exists():
        logger.warning("No baseline_results.json, skipping eval")
        return

    with open(EVAL_TASKS_PATH) as f:
        baseline_results = json.load(f)

    passing = [r for r in baseline_results if r["passed"]]
    import re as _re
    eval_tasks = random.sample(passing, min(EVAL_SAMPLE_SIZE, len(passing)))

    try:
        from sandbox.reward import eval_single, sandbox_reward_batch
        from sandbox import get_backend
        backend = get_backend("geak")
    except ImportError:
        logger.warning("Sandbox not available, skipping eval")
        return

    eval_gpu_ids = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "2,3,4,5,6,7").split(",")]
    compiled = 0
    correct = 0
    speedups = []

    model.eval()
    for i, task_info in enumerate(eval_tasks):
        tid = task_info["task"]
        task_dir = EVAL_ARTIFACTS / tid / "initial"
        if not (task_dir / "kernel.py").exists():
            continue

        kernel_src = (task_dir / "kernel.py").read_text()
        perf_stdout = task_info.get("performance", {}).get("stdout", "")
        match = _re.search(r"Perf:\s+([\d.]+)\s+ms", perf_stdout)
        baseline_ms = float(match.group(1)) if match else 1.0

        prompt_text = (
            "You are an expert GPU kernel engineer. "
            "Optimize the given kernel for AMD MI300X (gfx942). "
            "Return only a unified diff patch.\n"
            f"Kernel source:\n```python\n{kernel_src[:3000]}\n```"
        )

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.0, do_sample=False,
            )
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        gpu_id = eval_gpu_ids[i % len(eval_gpu_ids)]
        family = tid.split("-")[2] if len(tid.split("-")) > 2 else ""
        result = eval_single(backend, response, tid, task_dir, baseline_ms, family, gpu_id)

        if result["reward"] > -1.0:
            compiled += 1
        if result["reward"] > -0.5:
            correct += 1
        if result.get("speedup", 0) > 0:
            speedups.append(result["speedup"])

        del output_ids
        torch.cuda.empty_cache()

    model.train()
    n = max(1, len(eval_tasks))
    eval_time = time.time() - eval_start

    eval_metrics = {
        "eval/compile_rate": compiled / n,
        "eval/correct_rate": correct / n,
        "eval/speedup_mean": sum(speedups) / max(1, len(speedups)) if speedups else 0.0,
        "eval/n_tasks": n,
        "eval/time_s": eval_time,
    }
    logger.info(
        "Step %d eval: compile=%d/%d (%.0f%%) correct=%d/%d (%.0f%%) speedup_mean=%.3f time=%.0fs",
        step, compiled, n, 100 * compiled / n,
        correct, n, 100 * correct / n,
        eval_metrics["eval/speedup_mean"], eval_time,
    )

    eval_log = {"step": step, **eval_metrics}
    eval_path = output_dir / "eval.jsonl"
    with open(eval_path, "a") as f:
        f.write(json.dumps(eval_log) + "\n")

    log_wandb(wb_run, step, eval_metrics)


def init_wandb(args, output_dir):
    try:
        import wandb
        key = Path("/home/danyzhan/wandb.key").read_text().strip()
        os.environ["WANDB_API_KEY"] = key
        os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"
        os.environ["WANDB_INIT_TIMEOUT"] = "120"
        run = wandb.init(
            project="coder-model-rl", entity="danyzhan-amd",
            name=f"grpo-kernel-{args.tag}",
            config={
                "model": args.model_path, "algo": "GRPO",
                "batch_size": args.batch_size, "num_generations": args.num_generations,
                "lr": args.lr, "lora_rank": args.lora_rank,
                "temperature": args.temperature,
                "curriculum": "easy(0-30) -> medium(30-60) -> all(60+)",
            },
            dir=str(output_dir),
            settings=wandb.Settings(init_timeout=120),
        )
        logger.info("W&B initialized: %s", run.url)
        return run
    except Exception as e:
        logger.warning("W&B init failed (continuing without): %s", e)
        return None


def log_wandb(run, step, metrics):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        logger.warning("W&B log failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="GRPO kernel training v2")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", default="sft2")
    parser.add_argument("--dataset", default="/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/rl_prompts.jsonl")
    parser.add_argument("--output-dir", default="/home/danyzhan/Lumen-RL/experiments/multi-tune-agent/outputs")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"rl-grpo-{args.tag}"
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
                candidate = output_dir / f"step-{s + 1}"
                if candidate.exists() and (
                    (candidate / "adapter_model.safetensors").exists() or
                    (candidate / "adapter_model.bin").exists()
                ):
                    resume_ckpt = candidate
                    break
            logger.info("Resuming from step %d, ckpt=%s", start_step, resume_ckpt)

    os.environ.setdefault("EVAL_GPU_IDS", "2,3,4,5,6,7")
    device = f"cuda:{args.gpu}"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if resume_ckpt:
        logger.info("Loading base + LoRA from %s", resume_ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(resume_ckpt), is_trainable=True)
    else:
        logger.info("Loading model from %s onto %s", args.model_path, device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total_p, 100 * trainable / total_p)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.1,
    )

    all_prompts = load_prompts(args.dataset)
    wb_run = None if args.no_wandb else init_wandb(args, output_dir)

    logger.info(
        "GRPO v2: steps=%d batch=%d gen=%d temp=%.1f curriculum=easy->medium->all",
        args.num_steps, args.batch_size, args.num_generations, args.temperature,
    )

    for step in range(start_step, args.num_steps):
        step_start = time.time()

        pool = curriculum_filter(all_prompts, step)
        batch_prompts = random.sample(pool, min(args.batch_size, len(pool)))

        logger.info("Step %d: generating %d x %d responses...",
                     step, len(batch_prompts), args.num_generations)
        responses = generate_responses(
            batch_prompts, args.api_base, args.model_name,
            num_generations=args.num_generations,
            temperature=args.temperature,
        )

        logger.info("Step %d: computing rewards...", step)
        rewards = compute_rewards(batch_prompts, responses)

        flat_rewards = [r for rews in rewards for r in rews]
        n_total = max(1, len(flat_rewards))
        mean_reward = sum(flat_rewards) / n_total
        compile_rate = sum(1 for r in flat_rewards if r > -1.0) / n_total
        correct_rate = sum(1 for r in flat_rewards if r > -0.5) / n_total
        speedup_rate = sum(1 for r in flat_rewards if r > 0.5) / n_total

        logger.info(
            "Step %d: reward=%.3f compile=%.0f%% correct=%.0f%% speedup=%.0f%%",
            step, mean_reward, compile_rate * 100, correct_rate * 100, speedup_rate * 100,
        )

        logger.info("Step %d: computing GRPO loss...", step)
        model.train()
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            model, tokenizer, batch_prompts, responses, rewards, device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_time = time.time() - step_start
        loss_val = loss.item()
        logger.info("Step %d: loss=%.4f time=%.0fs", step, loss_val, step_time)

        step_log = {
            "step": step, "loss": loss_val,
            "mean_reward": mean_reward, "compile_rate": compile_rate,
            "correct_rate": correct_rate, "speedup_rate": speedup_rate,
            "step_time_s": step_time,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(step_log) + "\n")

        log_wandb(wb_run, step, {
            "train/reward": mean_reward, "train/compile_rate": compile_rate,
            "train/correct_rate": correct_rate, "train/speedup_rate": speedup_rate,
            "train/loss": loss_val, "train/step_time_s": step_time,
        })

        if (step + 1) % 10 == 0:
            ckpt_dir = output_dir / f"step-{step + 1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Saved checkpoint to %s", ckpt_dir)

            all_ckpts = sorted(
                [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            while len(all_ckpts) > 2:
                old = all_ckpts.pop(0)
                import shutil
                shutil.rmtree(old)
                logger.info("Deleted old checkpoint %s", old.name)

            run_online_eval(
                model, tokenizer, device, output_dir, step + 1, wb_run,
            )

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete. Final model saved to %s", final_dir)

    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    main()
