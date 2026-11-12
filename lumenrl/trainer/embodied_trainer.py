"""Embodied RL trainer: GRPO for a VLA policy acting in a learned world model.

SPMD under ``torchrun``; every rank holds

* a frozen world-model environment (video world model + success classifier),
* a bf16 HF rollout replica of the policy, and
* its FSDP2 shard of the trainable policy (FP32 master weights).

One training step:

1. sync actor weights into the rollout replica (from the second step on),
2. each rank resets ``groups_per_rank`` init states x ``num_generations``
   copies and rolls the policy out until success or the step budget,
3. the episode return is group-normalized into a GRPO advantage that
   every action token of the episode shares. With ``use_rel_reward`` the
   return is the sum of per-frame ``reward_coef * r_t - r_{t-1}``;
   otherwise it is 0/1 success,
4. PPO-style clipped updates over all executed action chunks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist

from lumenrl.algorithms.advantage_estimators import compute_grpo_advantage
from lumenrl.algorithms.grpo import GRPOAlgorithm
from lumenrl.core.config import FSDPEngineConfig, LumenRLConfig, OptimizerConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.embodied_plugin import load_embodied_plugin
from lumenrl.engine.inference.embodied_hf_engine import EmbodiedHFRollout, EmbodiedTrajectoryBatch
from lumenrl.engine.training.embodied_fsdp_engine import EmbodiedFSDP2Engine, load_policy_weights
from lumenrl.envs.world_model_env import WorldModelEnv, WorldModelInitDataset
from lumenrl.utils.lumen_patch import apply_lumen

logger = logging.getLogger(__name__)


class EmbodiedTrainer:
    def __init__(self, config: LumenRLConfig) -> None:
        self.plugin = load_embodied_plugin(config.embodied.plugin)
        self.config = config
        self.ecfg = config.embodied
        self.callbacks: list = []
        self._is_distributed = dist.is_initialized()
        self._rank = dist.get_rank() if self._is_distributed else 0
        self._world = dist.get_world_size() if self._is_distributed else 1
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._actor_model = None
        self._optimizer = None
        self._metrics_file = None

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        cfg, e = self.config, self.ecfg
        torch.cuda.set_device(self._local_rank)
        device = torch.device(f"cuda:{self._local_rank}")
        torch.manual_seed(cfg.seed + self._rank)
        lumen = dict(e.lumen or {})

        env_cfg = e.env
        if env_cfg.backend == "libero":
            from lumenrl.envs.libero_env import LiberoEnv, LiberoInitDataset

            self.init_states = LiberoInitDataset(env_cfg.task_suite)
            self.env = LiberoEnv(
                task_suite=env_cfg.task_suite,
                max_episode_steps=env_cfg.max_episode_steps,
                camera_height=env_cfg.camera_height,
                camera_width=env_cfg.camera_width,
                num_steps_wait=env_cfg.num_steps_wait,
                seed=cfg.seed * 1000 + self._rank,
                num_workers=env_cfg.num_workers,
            )
        elif env_cfg.backend == "world_model":
            if self.plugin.build_world_model is None or self.plugin.build_reward_model is None:
                raise ValueError("world_model backend requires plugin build_world_model and build_reward_model")
            wm = self.plugin.build_world_model(env_cfg.model_dir, device, **dict(env_cfg.world_model_kwargs or {}))
            wm = apply_lumen(wm, lumen.get("world_model"), "world_model")
            rm = self.plugin.build_reward_model(os.path.join(env_cfg.model_dir, env_cfg.reward_model_file), device)
            self.env = WorldModelEnv(
                wm, rm,
                max_episode_steps=env_cfg.max_episode_steps,
                context_frames=env_cfg.context_frames,
                success_logit_threshold=env_cfg.success_logit_threshold,
                wm_batch_size=env_cfg.wm_batch_size,
                seed=cfg.seed * 1000 + self._rank,
                use_rel_reward=env_cfg.use_rel_reward,
                reward_coef=env_cfg.reward_coef,
            )
            self.init_states = WorldModelInitDataset(
                env_cfg.init_data_dir or os.path.join(env_cfg.model_dir, "dataset"),
                env_cfg.context_frames,
                context_frame_step=self.env.ctx_step,
            )
        else:
            raise ValueError(f"unknown embodied.env.backend {env_cfg.backend!r}")

        self.processor = self.plugin.build_processor(e.policy.model_path, e.policy.unnorm_key)
        p = cfg.policy
        resume = e.resume_from or ""
        if resume and e.resume_step <= 0:
            raise ValueError("embodied.resume_from needs embodied.resume_step, the last finished step")
        self.actor = EmbodiedFSDP2Engine(
            model_path=e.policy.model_path,
            engine_config=FSDPEngineConfig(
                model_dtype="bf16",
                seed=cfg.seed,
                reshard_after_forward=e.policy.reshard_after_forward,
            ),
            optimizer_config=OptimizerConfig(
                lr=p.learning_rate,
                weight_decay=p.weight_decay,
                clip_grad=p.max_grad_norm,
                lr_scheduler_type="constant" if p.lr_decay_style == "constant" else "cosine",
                lr_warmup_steps=p.lr_warmup_steps,
                total_training_steps=cfg.num_training_steps,
                min_lr_ratio=(p.min_lr / p.learning_rate) if p.learning_rate > 0 else 0.0,
                adam_beta1=p.adam_beta1,
                adam_beta2=p.adam_beta2,
                adam_eps=p.adam_eps,
            ),
            gradient_checkpointing=e.policy.gradient_checkpointing,
            freeze_vision_backbone=e.policy.freeze_vision_backbone,
            build_policy=self.plugin.build_policy,
            processor=self.processor,
            lumen_kwargs=lumen.get("actor"),
            resume_from=resume,
        )
        self.actor.initialize()
        self._actor_model = self.actor.module
        self._optimizer = self.actor.optimizer

        rollout_policy = self.plugin.build_policy(e.policy.model_path, dtype=torch.bfloat16, device=device)
        if resume:
            load_policy_weights(rollout_policy, resume)
        rollout_policy = apply_lumen(rollout_policy, lumen.get("rollout"), "rollout")
        self.rollout = EmbodiedHFRollout(
            rollout_policy, self.processor,
            temperature=e.rollout.temperature,
            batch_size=e.rollout.batch_size,
            seed=cfg.seed * 1000 + self._rank,
        )
        self.algorithm = GRPOAlgorithm(cfg)
        self._weights_in_sync = True

        if self._rank == 0 and e.log_dir:
            Path(e.log_dir).mkdir(parents=True, exist_ok=True)
            self._metrics_file = open(os.path.join(e.log_dir, "embodied_metrics.jsonl"), "a")
        logger.info(
            "EmbodiedTrainer ready: world=%d, backend=%s, envs/rank=%d (groups=%d x G=%d), init states=%d, resume_step=%d",
            self._world, e.env.backend, e.rollout.groups_per_rank * cfg.algorithm.grpo.num_generations,
            e.rollout.groups_per_rank, cfg.algorithm.grpo.num_generations, len(self.init_states),
            e.resume_step,
        )

    # ------------------------------------------------------------------ loop

    def train(self) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(self)
        start = int(self.ecfg.resume_step)
        if start:
            logger.info(
                "Continuing after step %d (%s); next logged step is %d. Optimizer moments start at zero.",
                start, self.ecfg.resume_from, start + 1,
            )
        for step in range(start, self.config.num_training_steps):
            for cb in self.callbacks:
                cb.on_step_begin(self, step)
            metrics = self.train_step(step)
            if self._metrics_file is not None:
                self._metrics_file.write(json.dumps({"step": step + 1, **metrics}) + "\n")
                self._metrics_file.flush()
            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)
            interval = self.ecfg.save_interval
            last = step + 1 == self.config.num_training_steps
            if interval > 0 and ((step + 1) % interval == 0 or last):
                self.save_policy(step + 1)
        if hasattr(self.env, "close"):
            self.env.close()
        for cb in self.callbacks:
            cb.on_train_end(self)

    def save_policy(self, step: int) -> Optional[str]:
        """Write the current policy as bf16 safetensors (rank 0); collective when weights are stale."""
        from safetensors.torch import save_file

        if not self._weights_in_sync:
            self.rollout.update_weights(self.actor.iter_full_params(torch.bfloat16))
            self._weights_in_sync = True
        path = None
        if self._rank == 0:
            root = self.ecfg.save_dir or os.path.join(self.ecfg.log_dir or "results", "checkpoints")
            path = os.path.join(root, f"step_{step}", "policy.safetensors")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            state = {n: p.detach().contiguous().cpu() for n, p in self.rollout.policy.named_parameters()}
            save_file(state, path)
            logger.info("Saved policy (%d tensors) to %s", len(state), path)
        if self._is_distributed:
            dist.barrier()
        return path

    def train_step(self, step: int) -> dict[str, float]:
        t_start = time.time()
        metrics: dict[str, float] = {}

        t0 = time.time()
        if not self._weights_in_sync:
            self.rollout.update_weights(self.actor.iter_full_params(torch.bfloat16))
            torch.cuda.synchronize()
            self._weights_in_sync = True
        metrics["timing/weight_sync_s"] = time.time() - t0

        t0 = time.time()
        # One wave uses ``batch_size`` sims. A step's groups are collected over
        # several waves so 4096 episodes do not mean 4096 live simulators.
        inits = self._sample_inits(step)
        width = self.ecfg.rollout.batch_size
        parts = [
            self.rollout.collect(self.env, inits[s:s + width])
            for s in range(0, len(inits), width)
        ]
        batch = EmbodiedTrajectoryBatch.cat(parts)
        metrics["timing/rollout_s"] = time.time() - t0
        metrics.update(batch.metrics)

        data, adv_metrics = self._build_training_data(batch)
        metrics.update(adv_metrics)

        t0 = time.time()
        if self.ecfg.rollout.recompute_old_log_probs:
            with self.actor.eval_mode():
                out = self.actor.forward_backward_batch(
                    {**data, "meta": {}}, loss_function=None, forward_only=True
                )
            old = out["model_output"]["log_probs"][: data["input_ids"].shape[0]].float().cpu()
            mask = data["response_mask"].bool()
            if mask.any():
                metrics["train/rollout_vs_actor_logp_abs_diff"] = (
                    (old - data["rollout_log_probs"]).abs()[mask].mean().item()
                )
            data["old_log_probs"] = old
        else:
            data["old_log_probs"] = data["rollout_log_probs"]
        metrics["timing/old_logp_s"] = time.time() - t0

        t0 = time.time()
        metrics.update(self._update_policy(data, step))
        metrics["timing/update_s"] = time.time() - t0
        self._weights_in_sync = False

        metrics["timing/step_s"] = time.time() - t_start
        return self._reduce_metrics(metrics)

    # ------------------------------------------------------------------ pieces

    def _sample_inits(self, step: int):
        g = self.config.algorithm.grpo.num_generations
        n_groups = self.ecfg.rollout.groups_per_rank
        rng = np.random.default_rng((self.config.seed, step, self._rank))
        picks = rng.choice(len(self.init_states), n_groups, replace=n_groups > len(self.init_states))
        return [self.init_states[int(i)] for i in picks for _ in range(g)]

    def _build_training_data(self, batch: EmbodiedTrajectoryBatch) -> tuple[dict[str, Any], dict[str, float]]:
        g = self.config.algorithm.grpo.num_generations
        # Relative reward: one return per episode, the sum of per-frame
        # reward_coef * r_t - r_{t-1}. Otherwise the episode is 0/1 success.
        rewards = batch.returns if batch.returns is not None else batch.success.float()
        adv_env = compute_grpo_advantage(DataProto(tensors={"rewards": rewards}), self.config).tensors["advantages"]
        grouped = rewards.view(-1, g)
        algo = self.config.algorithm
        if batch.returns is not None and algo.filter_rewards:
            mean = grouped.mean(dim=1)
            group_keep = (mean >= algo.rewards_lower_bound) & (mean <= algo.rewards_upper_bound)
        else:
            group_keep = grouped.std(dim=1, unbiased=False) > 0
        metrics = {"train/groups_with_signal_frac": group_keep.float().mean().item()}
        if batch.returns is not None:
            metrics["train/episode_return"] = rewards.mean().item()

        adv = adv_env[batch.env_ids]
        keep = torch.ones(batch.num_samples, dtype=torch.bool)
        if (batch.returns is not None and algo.filter_rewards) or self.ecfg.rollout.filter_zero_advantage_groups:
            keep = group_keep.repeat_interleave(g)[batch.env_ids]
        metrics["train/samples_kept"] = float(keep.sum())

        idx = keep.nonzero().squeeze(-1)
        n_act = self.processor.num_action_tokens
        response_mask = torch.ones((idx.numel(), n_act))
        if idx.numel() == 0:
            # Every rank must run the same collectives: keep one masked-out row.
            idx = torch.zeros(1, dtype=torch.long)
            response_mask = torch.zeros((1, n_act))

        ids, mask = self.processor.build_text_batch(batch.instructions)
        inst = batch.instruction_ids[idx]
        data = {
            "input_ids": ids[inst],
            "attention_mask": mask[inst],
            "images": batch.images[idx],
            "action_tokens": batch.action_tokens[idx],
            "rollout_log_probs": batch.rollout_log_probs[idx],
            "advantages": adv[idx].float(),
            "response_mask": response_mask,
            "temperature": self.ecfg.rollout.temperature,
            "micro_batch_size": self.ecfg.train_micro_batch_size,
        }
        return data, metrics

    def _update_policy(self, data: dict[str, Any], step: int) -> dict[str, float]:
        grpo = self.config.algorithm.grpo
        n = int(data["input_ids"].shape[0])
        rng = np.random.default_rng((self.config.seed, step, self._rank, 7))
        dev = torch.device(f"cuda:{self._local_rank}")
        sums: dict[str, list[float]] = {}

        with self.actor.train_mode():
            for _epoch in range(grpo.num_ppo_epochs):
                order = torch.from_numpy(rng.permutation(n))
                for chunk in torch.tensor_split(order, grpo.num_mini_batches):
                    # Ranks must take the same number of optimizer steps; an
                    # empty mini-batch becomes one masked-out row.
                    empty = chunk.numel() == 0
                    if empty:
                        chunk = order[:1]
                    mb = {k: (v[chunk] if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                    if empty:
                        mb["response_mask"] = torch.zeros_like(mb["response_mask"])
                    tokens = mb["response_mask"].sum().to(dev)
                    if self._is_distributed:
                        dist.all_reduce(tokens)
                    mb["meta"] = {"calculate_entropy": True, "batch_num_tokens": max(int(tokens.item()), 1)}
                    self.actor.optimizer_zero_grad()
                    out = self.actor.forward_backward_batch(mb, loss_function=self._loss_fn)
                    grad_norm = self.actor.optimizer_step()
                    sums.setdefault("train/grad_norm", []).append(grad_norm)
                    for k, vals in out["metrics"].items():
                        sums.setdefault(k, []).extend(float(v) for v in vals)
            lr = self.actor.lr_scheduler_step()
        result = {k: float(np.mean(v)) for k, v in sums.items() if v}
        result["train/lr"] = lr
        return result

    def _loss_fn(self, model_output: dict[str, torch.Tensor], data: dict[str, Any]):
        dev = model_output["log_probs"].device
        mask = data["response_mask"].to(dev)
        logp = model_output["log_probs"]
        old = data["old_log_probs"].to(dev)
        dp = DataProto(
            tensors={
                "log_probs": logp,
                "old_log_probs": old,
                "advantages": data["advantages"].to(dev),
                "response_mask": mask,
            },
            meta={"dp_size": self._world, "batch_num_tokens": data["meta"]["batch_num_tokens"]},
        )
        loss, metrics = self.algorithm.compute_loss(dp)
        entropy = model_output.get("entropy")
        denom = mask.sum().clamp_min(1)
        with torch.no_grad():
            ratio = torch.exp(logp.detach() - old)
            clip_low = self.config.algorithm.grpo.clip_ratio
            clip_high = self.config.algorithm.clip_ratio_high or 0.28
            clipped = ((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).float()
            metrics["train/clip_frac"] = float((clipped * mask).sum() / denom)
            metrics["train/approx_kl"] = float((((ratio - 1) - (logp.detach() - old)) * mask).sum() / denom)
            if entropy is not None:
                metrics["train/entropy"] = float((entropy.detach() * mask).sum() / denom)
        if entropy is not None and self.ecfg.entropy_coeff > 0:
            ent = (entropy * mask).sum() / data["meta"]["batch_num_tokens"] * self._world
            loss = loss - self.ecfg.entropy_coeff * ent
        return loss, metrics

    def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Mean over the ranks that report each key (key sets can differ per rank)."""
        if not self._is_distributed:
            return metrics
        gathered: list = [None] * self._world
        dist.all_gather_object(gathered, sorted(metrics))
        keys = sorted(set().union(*gathered))
        dev = f"cuda:{self._local_rank}"
        vals = torch.tensor([metrics.get(k, 0.0) for k in keys], dtype=torch.float64, device=dev)
        have = torch.tensor([float(k in metrics) for k in keys], dtype=torch.float64, device=dev)
        dist.all_reduce(vals)
        dist.all_reduce(have)
        return dict(zip(keys, (vals / have.clamp_min(1)).tolist()))

    def cleanup(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
