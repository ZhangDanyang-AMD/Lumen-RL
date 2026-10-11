"""Continuously sync training.jsonl to W&B without restarting training."""

import json
import os
import sys
import time
from pathlib import Path

TRAINING_JSONL = Path("/home/danyzhan/Lumen-RL/experiments/multi-tune-agent/outputs/rl-mt-sft2/training.jsonl")
EVAL_JSONL = Path("/home/danyzhan/Lumen-RL/experiments/multi-tune-agent/outputs/rl-mt-sft2/eval.jsonl")
SYNC_INTERVAL = 30


def main():
    import wandb
    key = Path("/home/danyzhan/wandb.key").read_text().strip()
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_BASE_URL"] = "https://forge.coreweave.com/api/wandb"

    run = wandb.init(
        project="coder-model-rl", entity="danyzhan-amd",
        name="grpo-mt-epoch-sft2",
        id="grpo-mt-epoch-sft2",
        resume="allow",
        config={
            "model": "Qwen3-Coder-30B-A3B-SFT-TH-2epoch",
            "algo": "GRPO-MultiTurn-OnPolicy",
            "advantage": "TRLOO (leave-one-out)",
            "credit_assignment": "Kevin discounted gamma=0.4",
            "batch_size": 4, "group_size": 8, "effective_rollouts": 32,
            "max_turns": 8, "gamma": 0.4,
            "lr": 5e-7, "lora_rank": 32, "lora_alpha": 64,
            "temperature": 1.0,
            "curriculum": "easy(0-20) -> medium(20-40) -> all(40+)",
            "on_policy_sync": "merge_lora + restart_vllm every step",
            "reward_hacking_detection": True,
            "parallel_rollouts": 6,
            "hardware": "8x MI308X (gfx942)",
            "vllm_context": 32768,
        },
        settings=wandb.Settings(init_timeout=120),
    )
    print(f"W&B run: {run.url}")

    # Count already-synced steps to avoid re-logging
    train_synced = 0
    eval_synced = 0
    if TRAINING_JSONL.exists():
        with open(TRAINING_JSONL) as f:
            train_synced = len(f.readlines())
        print(f"Skipping {train_synced} already-synced training steps")
    if EVAL_JSONL.exists():
        with open(EVAL_JSONL) as f:
            eval_synced = len(f.readlines())
        print(f"Skipping {eval_synced} already-synced eval steps")

    while True:
        # Sync training metrics
        if TRAINING_JSONL.exists():
            with open(TRAINING_JSONL) as f:
                lines = f.readlines()
            for line in lines[train_synced:]:
                d = json.loads(line)
                total_turns = d.get("total_turns", 0)
                num_rollouts = 32
                run.log({
                    "train/reward": d["mean_reward"],
                    "train/compile_rate": d["compile_rate"],
                    "train/correct_rate": d["correct_rate"],
                    "train/speedup_rate": d.get("speedup_rate", 0),
                    "train/best_speedup": d.get("best_speedup", 0),
                    "train/loss": d["loss"],
                    "train/total_turns": total_turns,
                    "train/avg_turns": total_turns / max(1, num_rollouts),
                    "train/step_time_s": d["step_time_s"],
                }, step=d["step"])
                train_synced += 1
                print(f"Train step {d['step']}: reward={d['mean_reward']:.3f} compile={d['compile_rate']:.0%} correct={d['correct_rate']:.0%} best={d.get('best_speedup',0):.2f}x")

        # Sync eval metrics
        if EVAL_JSONL.exists():
            with open(EVAL_JSONL) as f:
                elines = f.readlines()
            for line in elines[eval_synced:]:
                d = json.loads(line)
                run.log({
                    "eval/compile": d.get("eval/compile", 0),
                    "eval/correct": d.get("eval/correct", 0),
                    "eval/speedup": d.get("eval/speedup", 0),
                }, step=d["step"])
                eval_synced += 1
                print(f"Eval step {d['step']}: compile={d.get('eval/compile',0):.0%} correct={d.get('eval/correct',0):.0%}")

        if train_synced >= 50:
            break

        time.sleep(SYNC_INTERVAL)

    run.finish()
    print("Sync complete")


if __name__ == "__main__":
    main()
