"""Group Relative Policy Optimization (GRPO)."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.loss_functions import kl_penalty, policy_gradient_loss
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.core.types import AlgorithmName

logger = logging.getLogger(__name__)


def _response_mask(batch: DataProto) -> Tensor | None:
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


def _expand_adv_to_tokens(advantages: Tensor, mask: Tensor | None) -> Tensor:
    """Broadcast [B] or [B, 1] advantages to [B, T] using ``mask``."""
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    if mask is None:
        return advantages
    return advantages.expand_as(mask.to(dtype=advantages.dtype)) * mask.to(
        dtype=advantages.dtype
    )


class GRPOAlgorithm(BaseAlgorithm):
    """GRPO: group-normalized scalar rewards as advantages (no critic).

    Trajectories must be laid out so that each group of ``num_generations``
    consecutive rows corresponds to the same prompt. Configure
    ``config.algorithm.grpo.num_generations``.
    """

    def compute_advantages(self, batch: DataProto) -> DataProto:
        if "rewards" not in batch.tensors:
            raise KeyError("GRPO requires tensor key 'rewards' on the batch.")
        rewards = batch.tensors["rewards"]
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        cfg = self._config.algorithm.grpo
        g = cfg.num_generations
        if rewards.shape[0] % g != 0:
            raise ValueError(
                f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
            )
        grouped = rewards.view(-1, g)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        adv = (grouped - mean) / std
        adv_flat = adv.reshape(-1)
        batch.tensors["advantages"] = adv_flat
        logger.debug("GRPO advantages: mean=%.6f std=%.6f", adv_flat.mean().item(), adv_flat.std().item())
        return batch

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        if "log_probs" not in batch.tensors or "old_log_probs" not in batch.tensors:
            raise KeyError("GRPO loss requires 'log_probs' and 'old_log_probs'.")
        if "advantages" not in batch.tensors:
            raise KeyError("GRPO loss requires precomputed 'advantages'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        mask = _response_mask(batch)

        adv_tok = _expand_adv_to_tokens(adv, mask)
        clip = self._config.algorithm.grpo.clip_ratio
        pg = policy_gradient_loss(logp, old_logp, adv_tok, clip, mask=mask)

        loss = pg
        metrics: dict[str, Any] = {"loss_pg": float(pg.detach().cpu())}

        kl_c = self._config.algorithm.grpo.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask)
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.GRPO.value, GRPOAlgorithm)
