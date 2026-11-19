"""PPO trainer for massively parallel GPU environments (Isaac Lab-style simulators).

Enabled with ``vec_env.enabled: true``; ``lumenrl.trainer.main`` dispatches here. The environment and the
actor-critic come from the ``vec_env.plugin`` module (see ``lumenrl.core.vec_env_plugin``); everything else
lives here. Each training step:

1. rollout: ``vec_env.horizon`` steps of all ``vec_env.num_envs`` environments, sampling from the diagonal
   Gaussian ``policy(obs) -> (mu, std, value)`` into a time-major buffer that stays on the GPU;
2. advantages from the ``gae_vec_env`` estimator of ``ADV_ESTIMATOR_REGISTRY`` (time-out bootstrap optional),
   optional value normalization, per-rollout advantage normalization;
3. ``algorithm.ppo.num_ppo_epochs`` epochs over shuffled mini-batches with ``PPOAlgorithm.compute_loss``
   (clipped surrogate + 0.5 x clipped value loss) plus the entropy bonus and the action-bounds penalty,
   gradient clipping and the KL-adaptive learning rate.

Multi-GPU: launch with ``torchrun --nproc_per_node=N`` (one process per GPU, LOCAL_RANK picks the device). Every
rank steps its own ``num_envs`` environments. Gradients, advantage / value / observation statistics and the KL used
by the learning-rate schedule are reduced across ranks, so all ranks hold the same policy. Single GPU runs without
any collective.

Metrics stay on the GPU during the step and are read back once at its end.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from lumenrl.algorithms.advantage_estimators import ADV_ESTIMATOR_REGISTRY
from lumenrl.algorithms.ppo import PPOAlgorithm  # noqa: F401  registers "ppo"
from lumenrl.algorithms.vec_env_ppo import (
    Obs,
    RunningMeanStd,
    gaussian_entropy,
    gaussian_kl,
    gaussian_log_prob,
    tree_map,
)
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.core.vec_env_plugin import load_vec_env_plugin


class VecEnvPPOTrainer:
    def __init__(self, config: LumenRLConfig) -> None:
        if config.algorithm.name.lower() != "ppo":
            raise ValueError(f"VecEnvPPOTrainer runs algorithm 'ppo', config has {config.algorithm.name!r}")
        if config.algorithm.adv_estimator not in ("", "gae_vec_env"):
            raise ValueError(
                f"VecEnvPPOTrainer needs adv_estimator 'gae_vec_env', config has {config.algorithm.adv_estimator!r}"
            )
        self.config = config
        self.vcfg = config.vec_env
        self.callbacks: list[Any] = []
        self._rank = 0
        self.env: Any = None
        self.policy: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.obs: Obs | None = None
        self.env_build_s = 0.0
        self.global_step = 0
        self._buf: dict[str, Any] | None = None
        self._metrics_file = None
        self._rank = 0
        self.world_size = 1

    # ---- lifecycle --------------------------------------------------------------------------------
    def _init_distributed(self) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size <= 1:
            return
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            from datetime import timedelta

            timeout = timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", 7200)))
            dist.init_process_group(backend="cpu:gloo,cuda:nccl", timeout=timeout)
        self._rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def setup(self) -> None:
        v, cfg = self.vcfg, self.config
        self._init_distributed()
        self.device = torch.device(v.device or f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.seed)

        plugin = load_vec_env_plugin(v.plugin)
        t0 = time.perf_counter()
        self.env = plugin.build_env(v.task, num_envs=v.num_envs, device=str(self.device),
                                    seed=cfg.seed + self._rank, **dict(v.env_kwargs))
        self.obs = self.env.reset()
        self._sync()
        self.env_build_s = time.perf_counter() - t0

        self.policy = plugin.build_policy(v.task, self.env, self.obs, **dict(v.policy_kwargs)).to(self.device)
        self._broadcast_policy()
        if v.normalize_input and getattr(self.policy, "obs_norm", None) is None:
            raise ValueError("vec_env.normalize_input needs policy.obs_norm (a RunningObsNorm)")
        self.algorithm = ALGORITHM_REGISTRY.get("ppo")(cfg)
        self.estimator = ADV_ESTIMATOR_REGISTRY["gae_vec_env"]
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=v.learning_rate)
        self.value_rms = RunningMeanStd(self.device) if v.normalize_value else None

        count = v.num_envs * v.horizon
        if v.minibatch_size > 0:
            if count % v.minibatch_size:
                raise ValueError(f"num_envs * horizon = {count} is not a multiple of minibatch_size")
            self.num_mini_batches = count // v.minibatch_size
        else:
            self.num_mini_batches = int(cfg.algorithm.ppo.num_mini_batches)

        if v.resume_from:
            self.load_state(v.resume_from)
            self.global_step = v.resume_step
        if v.log_dir and self._rank == 0:
            Path(v.log_dir).mkdir(parents=True, exist_ok=True)
            self._metrics_file = open(Path(v.log_dir) / "vec_env_metrics.jsonl", "a")

    def train(self) -> None:
        v, start = self.vcfg, self.global_step
        for cb in self.callbacks:
            cb.on_train_begin(self)
        last = start
        for step in range(start, self.config.num_training_steps):
            for cb in self.callbacks:
                cb.on_step_begin(self, step)
            metrics = self.train_step(step)
            if self._metrics_file is not None:
                self._metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
                self._metrics_file.flush()
            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)
            last = step + 1
            if v.save_interval > 0 and last % v.save_interval == 0:
                self.save_state(last)
        if v.save_interval > 0 and last > start and last % v.save_interval:
            self.save_state(last)
        for cb in self.callbacks:
            cb.on_train_end(self)

    def train_step(self, step: int) -> dict[str, Any]:
        """Collect one rollout from every environment, then run one PPO update."""
        v = self.vcfg
        self._sync()
        t0 = time.perf_counter()
        reward_sum = torch.zeros((), device=self.device)
        done_sum = torch.zeros((), device=self.device)
        obs = self.obs
        for t in range(v.horizon):
            actions = self.act(obs, t)
            obs, reward, terminated, truncated = self.env.step(actions)
            done = terminated | truncated
            self._record(t, reward * v.reward_scale, done, truncated)
            reward_sum += reward.sum()
            done_sum += done.sum()
        self.obs = obs
        self._sync()
        t1 = time.perf_counter()
        device_metrics = self.update(obs)
        self._sync()
        t2 = time.perf_counter()

        names = list(device_metrics)
        packed = torch.stack([device_metrics[k].detach().float().reshape(()) for k in names]
                             + [reward_sum, done_sum])
        if self.world_size > 1:
            dist.all_reduce(packed)
        *values, reward_total, done_total = packed.cpu().tolist()
        values = [x / self.world_size for x in values]  # losses and KL: mean over ranks
        env_steps = v.num_envs * v.horizon * self.world_size  # all ranks, for the per-step rates
        self.global_step = step + 1
        metrics: dict[str, Any] = {f"train/{k}": x for k, x in zip(names, values)}
        metrics.update({
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "rollout/reward_per_env_step": reward_total / env_steps,
            "rollout/episode_length": env_steps / max(done_total, 1.0),
            "rollout/episodes_done": done_total,
            "timing/rollout_s": t1 - t0,
            "timing/update_s": t2 - t1,
            "timing/step_s": t2 - t0,
            "throughput/env_steps_per_s": (env_steps / self.world_size) / (t2 - t0),  # this rank
        })
        return metrics

    def cleanup(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    # ---- rollout ----------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, obs: Obs) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample from the current policy: ``(actions, log_probs, values, mu, std)``. Updates the input
        normalizer when ``vec_env.normalize_input`` is set."""
        if self.vcfg.normalize_input:
            self._update_obs_norm(obs)
        mu, std, value = self.policy(obs)
        actions = mu + std * torch.randn_like(mu)
        return actions, gaussian_log_prob(mu, std, actions), self._denorm_value(value), mu, std

    @torch.no_grad()
    def act(self, obs: Obs, t: int) -> Tensor:
        """Sample actions and record the first half of transition ``t``."""
        actions, logp, value, mu, std = self.sample_actions(obs)
        if self._buf is None:
            self._allocate(obs, mu)
        buf = self._buf
        if isinstance(obs, dict):
            for k, x in obs.items():
                buf["obs"][k][t].copy_(x)
        else:
            buf["obs"][t].copy_(obs)
        buf["actions"][t] = actions
        buf["log_probs"][t] = logp
        buf["values"][t] = value
        buf["mu"][t] = mu
        buf["std"][t] = std
        return actions

    @torch.no_grad()
    def _record(self, t: int, rewards: Tensor, dones: Tensor, time_outs: Tensor) -> None:
        buf = self._buf
        buf["rewards"][t] = rewards.to(torch.float32)
        buf["dones"][t] = dones.to(torch.float32)
        buf["time_outs"][t] = time_outs.to(torch.float32)

    def _allocate(self, obs: Obs, mu: Tensor) -> None:
        T, n, a = self.vcfg.horizon, mu.shape[0], mu.shape[-1]
        dev, f = self.device, torch.float32
        self._buf = {
            "obs": tree_map(lambda x: torch.empty((T, *x.shape), dtype=x.dtype, device=dev), obs),
            "actions": torch.zeros(T, n, a, device=dev, dtype=f),
            "mu": torch.zeros(T, n, a, device=dev, dtype=f),
            "std": torch.zeros(T, n, a, device=dev, dtype=f),
            "log_probs": torch.zeros(T, n, device=dev, dtype=f),
            "values": torch.zeros(T, n, device=dev, dtype=f),
            "rewards": torch.zeros(T, n, device=dev, dtype=f),
            "dones": torch.zeros(T, n, device=dev, dtype=f),
            "time_outs": torch.zeros(T, n, device=dev, dtype=f),
        }

    @torch.no_grad()
    def _pooled_moments(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-column mean, variance and count of ``x`` [B, D] over the batches of every rank."""
        if self.world_size == 1:
            return x.mean(0), x.var(0, unbiased=False), torch.tensor(float(x.shape[0]), device=x.device)
        d = x.shape[1]
        packed = torch.cat([x.double().sum(0), x.double().square().sum(0),
                            torch.tensor([float(x.shape[0])], device=x.device, dtype=torch.float64)])
        dist.all_reduce(packed)
        total = packed[-1]
        mean = packed[:d] / total
        var = (packed[d:2 * d] / total - mean.square()).clamp_min(0.0)
        return mean.float(), var.float(), total.float()

    @torch.no_grad()
    def _update_obs_norm(self, obs: Tensor) -> None:
        mean, var, count = self._pooled_moments(obs.reshape(-1, obs.shape[-1]).float())
        self.policy.obs_norm.merge(mean, var, count)

    @torch.no_grad()
    def _mean_across_ranks(self, t: Tensor) -> Tensor:
        if self.world_size == 1:
            return t
        t = t.clone()
        dist.all_reduce(t)
        return t / self.world_size

    @torch.no_grad()
    def _broadcast_policy(self) -> None:
        """Rank 0's initial weights and normalizer statistics go to every rank."""
        if self.world_size == 1:
            return
        for t in list(self.policy.parameters()) + list(self.policy.buffers()):
            dist.broadcast(t.data, src=0)

    def _all_reduce_grads(self) -> None:
        """Average the gradients over ranks (one flat collective per mini-batch)."""
        if self.world_size == 1:
            return
        params = [p for p in self.policy.parameters() if p.grad is not None]
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(flat)
        flat.div_(self.world_size)
        offset = 0
        for p in params:
            n = p.grad.numel()
            p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n

    def _denorm_value(self, raw: Tensor) -> Tensor:
        if self.value_rms is None:
            return raw
        return raw * self.value_rms.std() + self.value_rms.mean

    # ---- update -----------------------------------------------------------------------------------
    def _set_lr(self, kl: Tensor) -> None:
        v = self.vcfg
        lr = self.optimizer.param_groups[0]["lr"]
        kl_value = float(self._mean_across_ranks(kl))  # every rank must pick the same learning rate
        if kl_value > 2.0 * v.desired_kl:
            lr = max(lr / 1.5, v.lr_min)
        elif 0.0 < kl_value < v.desired_kl / 2.0:
            lr = min(lr * 1.5, v.lr_max)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, last_obs: Obs) -> dict[str, Tensor]:
        """One PPO update over the rollout in the buffer; ``last_obs`` follows its last step. Returns the
        mean of each loss metric over the last mini-batch plus the mean KL, as device tensors."""
        v, buf = self.vcfg, self._buf
        ppo_cfg = self.config.algorithm.ppo
        with torch.no_grad():
            _, _, last_raw = self.policy(last_obs)
            tensors = {
                "rewards": buf["rewards"],
                "values": buf["values"],
                "dones": buf["dones"],
                "last_values": self._denorm_value(last_raw),
            }
            if v.time_out_bootstrap:
                tensors["time_outs"] = buf["time_outs"]
            gae = self.estimator(DataProto(tensors=tensors), self.config)
            adv, returns = gae.tensors["advantages"], gae.tensors["returns"]
            old_values = buf["values"]
            if self.value_rms is not None:
                mean_r, var_r, count_r = self._pooled_moments(returns.reshape(-1, 1))
                self.value_rms.merge(mean_r[0], var_r[0], count_r)
                mean, std = self.value_rms.mean, self.value_rms.std()
                returns = (returns - mean) / std
                old_values = (old_values - mean) / std
            mean_a, var_a, _ = self._pooled_moments(adv.reshape(-1, 1))
            adv = (adv - mean_a[0]) / var_a[0].sqrt().clamp(min=1e-8)

        T, n = buf["rewards"].shape
        count = T * n
        flat = lambda x: x.reshape(count, *x.shape[2:])  # noqa: E731
        f_obs = tree_map(flat, buf["obs"])
        f_act, f_mu, f_std = flat(buf["actions"]), flat(buf["mu"]), flat(buf["std"])
        f_logp, f_adv, f_ret, f_val = flat(buf["log_probs"]), flat(adv), flat(returns), flat(old_values)
        num_mb = self.num_mini_batches
        mb = count // num_mb
        adaptive = v.desired_kl > 0
        kls = []
        metrics: dict[str, Tensor] = {}
        for _ in range(int(ppo_cfg.num_ppo_epochs)):
            order = torch.randperm(count, device=self.device)
            epoch_kl = []
            for k in range(num_mb):
                b = order[k * mb:(k + 1) * mb]
                mu, std, value = self.policy(tree_map(lambda x: x[b], f_obs))
                logp = gaussian_log_prob(mu, std, f_act[b])
                batch = DataProto(tensors={
                    "log_probs": logp,
                    "old_log_probs": f_logp[b],
                    "advantages": f_adv[b],
                    "values": value,
                    "old_values": f_val[b],
                    "returns": f_ret[b],
                })
                loss, metrics = self.algorithm.compute_loss(batch, metrics_on_device=True)
                if v.entropy_coef:
                    loss = loss - v.entropy_coef * gaussian_entropy(std).mean()
                if v.bounds_coef:
                    outside = mu.clamp(-v.action_bound, v.action_bound) - mu
                    loss = loss + v.bounds_coef * outside.square().sum(-1).mean()
                with torch.no_grad():
                    kl = gaussian_kl(f_mu[b], f_std[b], mu, std).mean()
                epoch_kl.append(kl)
                if adaptive and v.kl_schedule == "minibatch":
                    self._set_lr(kl)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._all_reduce_grads()
                nn.utils.clip_grad_norm_(self.policy.parameters(), v.max_grad_norm)
                self.optimizer.step()
            kl_epoch = torch.stack(epoch_kl).mean()
            kls.append(kl_epoch)
            if adaptive and v.kl_schedule == "epoch":
                self._set_lr(kl_epoch)
        metrics = {k: x for k, x in metrics.items() if isinstance(x, Tensor)}
        metrics["kl"] = torch.stack(kls).mean()
        return metrics

    # ---- checkpoints ------------------------------------------------------------------------------
    def save_state(self, step: int) -> Path:
        v = self.vcfg
        root = Path(v.save_dir or (Path(v.log_dir or ".") / "checkpoints"))
        path = root / f"step_{step}" / "trainer_state.pt"
        if self._rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "value_rms": None if self.value_rms is None else self.value_rms.state_dict(),
            }, path)
        if self.world_size > 1:
            dist.barrier()
        return path

    def load_state(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.value_rms is not None and state.get("value_rms") is not None:
            self.value_rms.load_state_dict(state["value_rms"])

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
