"""Pieces of PPO for vectorized continuous-control environments: diagonal-Gaussian policy math and running
normalizers. The training loop that uses them is ``lumenrl.trainer.vec_env_ppo_trainer.VecEnvPPOTrainer``.

Policies built for that trainer return ``(mu, std, value)`` from ``forward(obs)``. A policy that normalizes its
inputs owns a :class:`RunningObsNorm` as ``policy.obs_norm`` and applies it inside ``forward``; the trainer updates
its statistics during collection when ``vec_env.normalize_input`` is set.
"""

from __future__ import annotations

import math
from typing import Callable, Union

import torch
import torch.nn as nn
from torch import Tensor

Obs = Union[Tensor, dict[str, Tensor]]


def gaussian_log_prob(mu: Tensor, std: Tensor, action: Tensor) -> Tensor:
    var = std.square()
    return (-0.5 * ((action - mu).square() / var + torch.log(var) + math.log(2 * math.pi))).sum(-1)


def gaussian_entropy(std: Tensor) -> Tensor:
    return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum(-1)


def gaussian_kl(mu: Tensor, std: Tensor, new_mu: Tensor, new_std: Tensor) -> Tensor:
    """KL(old || new) of two diagonal Gaussians, per sample."""
    return (
        torch.log(new_std) - torch.log(std) + (std.square() + (mu - new_mu).square()) / (2 * new_std.square()) - 0.5
    ).sum(-1)


def tree_map(fn: Callable[[Tensor], Tensor], obs: Obs) -> Obs:
    if isinstance(obs, dict):
        return {k: fn(v) for k, v in obs.items()}
    return fn(obs)


class RunningMeanStd:
    """Running scalar mean / variance (parallel-variance update). Used for value normalization."""

    def __init__(self, device: torch.device):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.tensor(1e-4, device=device)

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        x = x.reshape(-1).float()
        count = x.numel()
        mean, var = x.mean(), x.var(unbiased=False)
        total = self.count + count
        delta = mean - self.mean
        self.mean = self.mean + delta * count / total
        self.var = (self.var * self.count + var * count + delta.square() * self.count * count / total) / total
        self.count = total

    def std(self) -> Tensor:
        return torch.sqrt(self.var + 1e-8)

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.mean, self.var, self.count = (state[k].to(self.mean.device) for k in ("mean", "var", "count"))


class RunningObsNorm(nn.Module):
    """Input normalization with clipping (RL-Games normalize_input, clip_observations). Collection calls update();
    forward() only applies the statistics, so the PPO update re-normalizes stored raw observations."""

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        batch = x.reshape(-1, x.shape[-1])
        count = batch.shape[0]
        mean = batch.mean(0)
        var = batch.var(0, unbiased=False)
        total = self.count + count
        delta = mean - self.mean
        self.mean.copy_(self.mean + delta * count / total)
        self.var.copy_((self.var * self.count + var * count + delta.square() * self.count * count / total) / total)
        self.count.copy_(total)

    def forward(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)
