"""Evaluate embodied policies and render comparison videos.

Works with both environment backends (``embodied.env.backend``): the learned
world model and the LIBERO simulator. Every policy runs on the same init states
(and, for the world model, the same seed), so the videos put e.g. the SFT
policy and a GRPO checkpoint side by side::

    python -m lumenrl.trainer.embodied_eval --config path/to/config.yaml \\
        --policy base --policy results/.../step_100/policy.safetensors \\
        --label "SFT" --label "GRPO step 100" \\
        --num-episodes 64 --video demo.mp4 [overrides...]

``base`` means the weights in ``embodied.policy.model_path``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from lumenrl.core.config import load_config
from lumenrl.core.embodied_plugin import load_embodied_plugin
from lumenrl.engine.inference.embodied_hf_engine import EmbodiedHFRollout
from lumenrl.utils.lumen_patch import apply_lumen

logger = logging.getLogger(__name__)


def _pick_inits(dataset, n: int) -> list:
    """Deterministic round-robin over tasks so every policy sees the same init states."""
    by_task: dict[str, list] = {}
    for s in dataset.states:
        by_task.setdefault(s.instruction, []).append(s)
    tasks = sorted(by_task)
    picks, k = [], 0
    while len(picks) < n:
        pool = by_task[tasks[k % len(tasks)]]
        picks.append(pool[(k // len(tasks)) % len(pool)])
        k += 1
    return picks


def _label(frame: np.ndarray, lines: list[str], ok: bool) -> np.ndarray:
    import cv2

    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 14 * len(lines) + 6), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (4, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    color = (40, 200, 40) if ok else (220, 40, 40)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, 3)
    return out


def write_comparison_video(
    path: str, runs: list[dict], labels: list[str], episodes: list[int], fps: int, scale: int = 1
) -> None:
    """Grid video: one row per episode, one column per policy; border green on success."""
    import cv2
    import imageio.v2 as imageio

    length = max(len(r["history"][e]) for r in runs for e in episodes)
    rows = []
    for e in episodes:
        cols = []
        for r, name in zip(runs, labels):
            frames = r["history"][e]
            if scale > 1:
                h, w = frames.shape[1:3]
                frames = np.stack([cv2.resize(f, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC) for f in frames])
            ok = bool(r["success"][e])
            tag = f"{name}: {'success' if ok else 'fail'} @ {len(frames) - 1} steps"
            video = [_label(frames[min(t, len(frames) - 1)], [tag, r["instructions"][e][:44]], ok) for t in range(length)]
            cols.append(np.stack(video))
        rows.append(np.concatenate(cols, axis=2))
    grid = np.concatenate(rows, axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, list(grid), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    logger.info("Wrote %s (%d frames, %dx%d)", path, grid.shape[0], grid.shape[2], grid.shape[1])


def _build_env(e, plugin, device, args):
    env_cfg = e.env
    if env_cfg.backend == "libero":
        from lumenrl.envs.libero_env import LiberoEnv, LiberoInitDataset

        dataset = LiberoInitDataset(env_cfg.task_suite)
        env = LiberoEnv(
            task_suite=env_cfg.task_suite,
            max_episode_steps=env_cfg.max_episode_steps,
            camera_height=env_cfg.camera_height,
            camera_width=env_cfg.camera_width,
            num_steps_wait=env_cfg.num_steps_wait,
            seed=args.seed,
            num_workers=args.batch,
            record_frames=bool(args.video),
        )
        return env, dataset

    from lumenrl.envs.world_model_env import WorldModelEnv, WorldModelInitDataset

    wm = plugin.build_world_model(env_cfg.model_dir, device, **dict(env_cfg.world_model_kwargs or {}))
    wm = apply_lumen(wm, (e.lumen or {}).get("world_model"), "world_model")
    rm = plugin.build_reward_model(os.path.join(env_cfg.model_dir, env_cfg.reward_model_file), device)
    env = WorldModelEnv(
        wm, rm,
        max_episode_steps=env_cfg.max_episode_steps,
        context_frames=env_cfg.context_frames,
        success_logit_threshold=env_cfg.success_logit_threshold,
        wm_batch_size=env_cfg.wm_batch_size,
        record_frames=bool(args.video),
    )
    dataset = WorldModelInitDataset(
        env_cfg.init_data_dir or os.path.join(env_cfg.model_dir, "dataset"),
        env_cfg.context_frames,
        context_frame_step=env.ctx_step,
    )
    return env, dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", action="append", default=[], help="'base' or a policy.safetensors path")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--num-episodes", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64, help="episodes per env batch (LIBERO: one process each)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=None, help="default: embodied.rollout.temperature")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--video", default="")
    ap.add_argument("--video-episodes", type=int, default=4)
    ap.add_argument("--video-scale", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="", help="write per-policy results as JSON")
    args, overrides = ap.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, overrides)
    e = cfg.embodied
    plugin = load_embodied_plugin(e.plugin)
    device = torch.device("cuda", torch.cuda.current_device())
    policies = args.policy or ["base"]
    labels = args.label or [os.path.basename(os.path.dirname(p)) if p != "base" else "base" for p in policies]
    assert len(labels) == len(policies), "one --label per --policy"

    env, dataset = _build_env(e, plugin, device, args)
    inits = _pick_inits(dataset, args.num_episodes)
    processor = plugin.build_processor(e.policy.model_path, e.policy.unnorm_key)
    policy = plugin.build_policy(e.policy.model_path, dtype=torch.bfloat16, device=device)
    policy = apply_lumen(policy, (e.lumen or {}).get("rollout"), "rollout")
    base_state = {n: p.detach().cpu().clone() for n, p in policy.named_parameters()}
    temperature = e.rollout.temperature if args.temperature is None else args.temperature

    runs = []
    for spec, name in zip(policies, labels):
        if spec == "base":
            state = base_state
        else:
            from safetensors.torch import load_file

            state = load_file(spec)
        rollout = EmbodiedHFRollout(policy, processor, temperature=temperature, batch_size=e.rollout.batch_size,
                                    seed=args.seed)
        rollout.update_weights(state.items())
        if hasattr(env, "generator"):
            env.generator.manual_seed(args.seed)
        success, chunks, history, instructions = [], [], [], []
        for s in range(0, len(inits), args.batch):
            batch = rollout.collect(env, inits[s:s + args.batch], greedy=args.greedy)
            success += batch.success.tolist()
            chunks += batch.episode_chunks.tolist()
            instructions += env.instructions
            if args.video:
                history += [np.concatenate([f.numpy() for f in h]) for h in env.history]
        rate = float(np.mean(success))
        per_task: dict[str, list] = {}
        for ok, ins in zip(success, instructions):
            per_task.setdefault(ins, []).append(bool(ok))
        logger.info("%s: success %.3f over %d episodes, mean chunks %.1f", name, rate, len(success), np.mean(chunks))
        runs.append({"label": name, "policy": spec, "success_rate": rate, "success": success,
                     "episode_chunks": chunks, "instructions": instructions, "history": history,
                     "per_task": {k: float(np.mean(v)) for k, v in sorted(per_task.items())}})
    if hasattr(env, "close"):
        env.close()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "history"} for r in runs], f, indent=1)
    if args.video:
        # Episodes the last policy fixes come first, then regressions, then the rest.
        first, last = runs[0]["success"], runs[-1]["success"]
        episodes = sorted(range(len(inits)), key=lambda i: (not (last[i] and not first[i]), first[i] == last[i]))
        write_comparison_video(args.video, runs, labels, episodes[: args.video_episodes], args.fps, args.video_scale)
    for r in runs:
        print(f"{r['label']}: success_rate={r['success_rate']:.3f}")


if __name__ == "__main__":
    main()
