"""Decoupled Advantage Policy Optimization (DAPO-style) algorithm."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.loss_functions import asymmetric_clip_loss, kl_penalty
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


def _apply_overlong_shaping(
    rewards: Tensor,
    batch: DataProto,
    max_len: int,
    penalty: float,
) -> Tensor:
    """Subtract a linear penalty when sequence length exceeds ``max_len``."""
    if penalty <= 0.0:
        return rewards
    lengths = batch.meta.get("response_lengths")
    if lengths is None:
        return rewards
    lens_t = torch.as_tensor(lengths, device=rewards.device, dtype=torch.float32)
    if lens_t.shape[0] != rewards.shape[0]:
        return rewards
    over = torch.clamp(lens_t - float(max_len), min=0.0)
    shaped = rewards.to(dtype=torch.float32) - penalty * over
    return shaped.to(dtype=rewards.dtype)


class DAPOAlgorithm(BaseAlgorithm):
    """DAPO-style training: asymmetric clipping, dynamic sampling, token-level PG.

    Advantages are group-relative (like GRPO) with optional variance-based
    dynamic sampling. Loss uses asymmetric PPO-style clipping bounds.
    """

    def compute_advantages(self, batch: DataProto) -> DataProto:
        if "rewards" not in batch.tensors:
            raise KeyError("DAPO requires tensor key 'rewards' on the batch.")
        rewards = batch.tensors["rewards"]
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)

        cfg = self._config.algorithm.dapo
        g = cfg.num_generations
        if rewards.shape[0] % g != 0:
            raise ValueError(
                f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
            )

        if cfg.overlong_reward_shaping:
            rewards = _apply_overlong_shaping(
                rewards,
                batch,
                max_len=int(self._config.policy.max_total_sequence_length),
                penalty=float(batch.meta.get("overlong_penalty", 1e-4)),
            )

        grouped = rewards.view(-1, g)
        std = grouped.std(dim=1, unbiased=False)

        if cfg.dynamic_sampling:
            keep = std > 1e-6
            if not torch.any(keep):
                logger.warning("DAPO dynamic sampling removed all groups; keeping all.")
                keep = torch.ones_like(keep, dtype=torch.bool)
            row_mask = keep.unsqueeze(-1).expand(-1, g).reshape(-1)
        else:
            row_mask = torch.ones(rewards.shape[0], dtype=torch.bool, device=rewards.device)

        mean = grouped.mean(dim=1, keepdim=True)
        std_safe = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        adv = (grouped - mean) / std_safe
        adv_flat = adv.reshape(-1)

        batch.tensors["advantages"] = adv_flat
        batch.tensors["dapo_sample_mask"] = row_mask.to(dtype=torch.float32)
        logger.info(
            "NaN-DEBUG DAPO advantages: active_frac=%.4f, adv nan=%d inf=%d "
            "min=%.4f max=%.4f mean=%.4f, rewards min=%.4f max=%.4f mean=%.4f, "
            "std min=%.6f max=%.6f",
            float(row_mask.float().mean().cpu()),
            adv_flat.isnan().sum().item(), adv_flat.isinf().sum().item(),
            adv_flat.min().item(), adv_flat.max().item(), adv_flat.mean().item(),
            rewards.min().item(), rewards.max().item(), rewards.mean().item(),
            std.min().item(), std.max().item(),
        )
        return batch

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        if "log_probs" not in batch.tensors or "old_log_probs" not in batch.tensors:
            raise KeyError("DAPO loss requires 'log_probs' and 'old_log_probs'.")
        if "advantages" not in batch.tensors:
            raise KeyError("DAPO loss requires 'advantages'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        mask = _response_mask(batch)
        sample_mask = batch.tensors.get("dapo_sample_mask")

        cfg = self._config.algorithm.dapo
        if adv.dim() == 1:
            adv = adv.unsqueeze(-1)
        if mask is not None:
            adv = adv.expand_as(mask.to(dtype=adv.dtype)) * mask.to(dtype=adv.dtype)
        else:
            adv = adv.expand_as(logp)

        if sample_mask is not None:
            sm = sample_mask
            if sm.dim() == 1:
                sm = sm.unsqueeze(-1)
            sm = sm.expand_as(logp).to(dtype=logp.dtype)
            if mask is not None:
                sm = sm * mask.to(dtype=logp.dtype)
        else:
            sm = mask.to(dtype=logp.dtype) if mask is not None else torch.ones_like(logp)

        low = float(cfg.clip_ratio_low)
        high = float(cfg.clip_ratio_high)
        if cfg.token_level_pg:
            pg = asymmetric_clip_loss(logp, old_logp, adv, low, high, mask=sm)
        else:
            # Sequence-level: mean logp per row, scalar adv per row
            denom = torch.clamp(sm.sum(dim=-1, keepdim=True), min=1.0)
            seq_logp = (logp * sm).sum(dim=-1, keepdim=True) / denom
            seq_old = (old_logp * sm).sum(dim=-1, keepdim=True) / denom
            seq_adv = adv.sum(dim=-1, keepdim=True) / torch.clamp(
                sm.sum(dim=-1, keepdim=True), min=1.0
            )
            pg = asymmetric_clip_loss(seq_logp, seq_old, seq_adv, low, high, mask=None)

        loss = pg
        metrics: dict[str, Any] = {"loss_pg": float(pg.detach().cpu())}

        kl_c = cfg.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask)
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.DAPO.value, DAPOAlgorithm)
