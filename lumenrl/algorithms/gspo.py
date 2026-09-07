"""Group Sequence Policy Optimization (GSPO).

Sequence-level importance ratio with symmetric PPO-style clipping.
Designed for MoE models (e.g., Qwen3-30B-A3B) where token-level ratio
variance destabilises training.  Reference: Qwen3 technical report.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.loss_functions import gspo_loss, kl_penalty
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.core.types import AlgorithmName


def _response_mask(batch: DataProto) -> Tensor | None:
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


class GSPOAlgorithm(BaseAlgorithm):
    """GSPO: sequence-level ratio clipping, group-normalized advantages.

    Trajectories must be laid out so that each group of ``num_generations``
    consecutive rows corresponds to the same prompt.  Configure
    ``config.algorithm.gspo.num_generations``.
    """

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        if "log_probs" not in batch.tensors or "old_log_probs" not in batch.tensors:
            raise KeyError("GSPO loss requires 'log_probs' and 'old_log_probs'.")
        if "advantages" not in batch.tensors:
            raise KeyError("GSPO loss requires precomputed 'advantages'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        mask = _response_mask(batch)

        clip = float(self._config.algorithm.gspo.clip_ratio)

        pg = gspo_loss(logp, old_logp, adv, clip, mask=mask)

        loss = pg
        metrics: dict[str, Any] = {"loss_pg": float(pg.detach().cpu())}

        kl_c = self._config.algorithm.gspo.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask)
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.GSPO.value, GSPOAlgorithm)
