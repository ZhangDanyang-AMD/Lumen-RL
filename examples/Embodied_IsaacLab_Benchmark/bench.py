"""Isaac Lab 2.3 performance-benchmark measurements for hipPhysics tasks trained by Lumen-RL's VecEnvPPOTrainer.

    python bench.py --config configs/cartpole.yaml [--steps 300] [vec_env.num_envs=2048 ...]

The published table (docs.nvidia.com/isaaclab/2.3/overview/performance-benchmarks) reports three effective FPS
numbers per task (num_envs x steps / wall time):

  environment step                         random actions in [-1, 1]
  environment step and inference           trainer.sample_actions (policy forward + sample) + step
  environment step, inference, and train   trainer.train_step: ``vec_env.horizon`` steps of collection plus one
                                           PPO update, amortized

The trainer is built from the YAML exactly as ``python -m lumenrl.trainer.main`` builds it; this script only times
it. ``--learn-iters N`` runs ``trainer.train()`` for N steps and prints progress instead. Extra ``key=value``
arguments are OmegaConf overrides of the config.

Import torch before hipphysics (LLVM symbol clash); the run_bench_*.sh scripts set up the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from isaaclab_plugin import TASKS  # noqa: E402
from lumenrl.core.config import LumenRLConfig  # noqa: E402
from lumenrl.trainer.vec_env_ppo_trainer import VecEnvPPOTrainer  # noqa: E402

AGENT = ("Lumen-RL VecEnvPPOTrainer (lumenrl.trainer.vec_env_ppo_trainer): PPOAlgorithm.compute_loss from "
         "ALGORITHM_REGISTRY 'ppo', advantages from ADV_ESTIMATOR_REGISTRY 'gae_vec_env'")


def sync() -> None:
    torch.cuda.synchronize()


def rss_gb() -> float:
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def vram_used_gb(index: int) -> float:
    """Device-wide VRAM in use on torch's device ``index`` (hipPhysics and HIPRT allocate outside torch's caching
    allocator). rocm-smi -d numbers devices in its own order, so it is not used here."""
    free, total = torch.cuda.mem_get_info(index)
    return (total - free) / (1024 ** 3)


def step_fps(num_envs: int, steps: int, action_fn: Callable[[], None]) -> float:
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        action_fn()
    sync()
    return num_envs * steps / (time.perf_counter() - start)


class ProgressPrinter:
    """Trainer callback for --learn-iters: reward per env step, steps per episode, KL, LR and FPS."""

    def __init__(self, every: int):
        self.every = every

    def on_train_begin(self, trainer) -> None:
        print(f"{'iter':>5}  {'reward/step':>12}  {'episode len':>12}  {'kl':>9}  {'lr':>9}  {'FPS':>10}", flush=True)

    def on_step_begin(self, trainer, step: int) -> None:
        pass

    def on_step_end(self, trainer, step: int, m: dict) -> None:
        if (step + 1) % self.every == 0:
            print(f"{step + 1:>5}  {m['rollout/reward_per_env_step']:>12.4f}  {m['rollout/episode_length']:>12.1f}  "
                  f"{m['train/kl']:>9.5f}  {m['train/lr']:>9.2e}  {m['throughput/env_steps_per_s']:>10,.0f}",
                  flush=True)

    def on_train_end(self, trainer) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<task>.yaml")
    ap.add_argument("--steps", type=int, default=300, help="timed control steps for the first two columns")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--train-iters", type=int, default=0,
                    help="timed trainer steps after one untimed (default 3 for cartpole_rgb, else 6)")
    ap.add_argument("--learn-iters", type=int, default=0, help="learning check: train this many steps and print")
    ap.add_argument("--learn-every", type=int, default=10)
    ap.add_argument("--save-frame", default="", help="cartpole_rgb: write a grid of the first 16 frames to this PNG")
    ap.add_argument("--out", default="")
    args, overrides = ap.parse_known_args()

    config = LumenRLConfig.from_yaml(args.config, overrides=overrides)
    v = config.vec_env
    meta = TASKS[v.task]
    renderer = v.env_kwargs.get("renderer") if v.task == "cartpole_rgb" else None
    name = meta["result_name"] + (f"_{renderer}" if renderer else "")
    out_path = args.out or f"/work/runs/isaaclab_bench/{name}_{v.num_envs}.json"
    train_iters = args.train_iters or (3 if v.task == "cartpole_rgb" else 6)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(v.device or "cuda:0")
    torch.cuda.set_device(device)
    gpu_index = device.index or 0
    vram_base = vram_used_gb(gpu_index)

    trainer = VecEnvPPOTrainer(config)
    trainer.setup()
    env, n = trainer.env, v.num_envs

    if args.learn_iters:
        config.num_training_steps = args.learn_iters
        trainer.callbacks = [ProgressPrinter(args.learn_every)]
        trainer.train()
        trainer.cleanup()
        return

    obs = trainer.obs

    def random_step():
        nonlocal obs
        obs = env.step(torch.empty(n, env.action_dim, device=device).uniform_(-1.0, 1.0))[0]

    def inference_step():
        nonlocal obs
        obs = env.step(trainer.sample_actions(obs)[0])[0]

    for _ in range(args.warmup):
        random_step()
    step = step_fps(n, args.steps, random_step)
    rss_after_step = rss_gb()
    vram_after_step = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    render_fps = None
    if callable(getattr(env, "render", None)):
        for _ in range(10):
            env.render()
        render_fps = step_fps(n, 100, env.render)

    for _ in range(args.warmup):
        inference_step()
    infer = step_fps(n, args.steps, inference_step)
    if args.save_frame and v.task == "cartpole_rgb":
        from PIL import Image

        h, w = obs.shape[1:3]
        tiles = obs[:16].cpu().numpy()
        Image.fromarray(tiles.reshape(4, 4, h, w, 3).transpose(0, 2, 1, 3, 4).reshape(4 * h, 4 * w, 3)).save(
            args.save_frame)

    trainer.obs = obs
    trainer.train_step(-1)
    runs = [trainer.train_step(i) for i in range(train_iters)]
    train = sum(m["throughput/env_steps_per_s"] for m in runs) / len(runs)
    last = runs[-1]
    stats = {
        "reward_per_env_step": sum(m["rollout/reward_per_env_step"] for m in runs) / len(runs),
        "episode_length_steps": (n * v.horizon * len(runs)) / max(sum(m["rollout/episodes_done"] for m in runs), 1.0),
        "kl": last["train/kl"],
        "lr": last["train/lr"],
        "loss_pg": last["train/loss_pg"],
        "loss_vf": last["train/loss_vf"],
        "rollout_s": last["timing/rollout_s"],
        "update_s": last["timing/update_s"],
    }

    vram_used = vram_used_gb(gpu_index)
    result = {
        "task": meta["task"],
        "source_task": meta["source_task"],
        "agent": f"{AGENT}; {meta['hyperparameters']}",
        "config": os.path.relpath(os.path.abspath(args.config), HERE),
        "overrides": overrides,
        "num_envs": n,
        "gpu": torch.cuda.get_device_name(device),
        "seed": config.seed,
        "timed_steps": args.steps,
        "train_iterations": train_iters,
        "scene_build_s": round(trainer.env_build_s, 3),
        "training_sanity": stats,
        "fps": {
            "environment_step": round(step),
            "environment_step_and_inference": round(infer),
            "environment_step_inference_and_train": round(train),
        },
        "memory_gb": {
            "rss_after_step": round(rss_after_step, 2),
            "torch_reserved_after_step": round(vram_after_step, 2),
            "rss": round(rss_gb(), 2),
            "torch_reserved": round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 2),
            "device_vram_used": round(vram_used, 2),
            "device_vram_delta": round(vram_used - vram_base, 2),
        },
        "reference_rtx4090": meta["reference_rtx4090"],
    }
    if renderer:
        from cartpole_rgb_env import RENDERERS

        result["renderer"] = RENDERERS[renderer]
        result["render_only_frames_per_s"] = round(render_fps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    trainer.cleanup()


if __name__ == "__main__":
    main()
