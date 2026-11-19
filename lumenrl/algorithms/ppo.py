"""Proximal Policy Optimization (PPO) with GAE and value loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lumenrl.algorithms.base_algorithm import BaseAlgorithm
from lumenrl.algorithms.loss_functions import kl_penalty, policy_gradient_loss, value_loss
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.core.types import AlgorithmName

class PPOAlgorithm(BaseAlgorithm):
    """Standard PPO with clipped surrogate, optional KL to reference, and value loss."""

    def compute_loss(
        self, batch: DataProto, *, metrics_on_device: bool = False
    ) -> tuple[Tensor, dict[str, Any]]:
        """Clipped surrogate plus ``0.5 *`` clipped value loss.

        ``metrics_on_device=False`` (the default) returns metrics as Python floats, which synchronizes
        the device once per entry. Vectorized-env updates call this once per mini-batch, so
        ``VecEnvPPO`` passes ``metrics_on_device=True`` and reads the metrics once per update.
        """
        required = ("log_probs", "old_log_probs", "advantages", "values", "returns")
        for k in required:
            if k not in batch.tensors:
                raise KeyError(f"PPO loss missing tensor '{k}'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        values = batch.tensors["values"]
        rets = batch.tensors["returns"]
        mask = batch.tensors.get("attention_mask")
        # No mask means every element counts. The loss functions reduce with mean() in that case,
        # so there is no ones-tensor to build and no mask reduction to synchronize on.
        if mask is None:
            mask_t = None if metrics_on_device else torch.ones_like(adv)
        else:
            mask_t = mask.to(dtype=adv.dtype)

        clip = self._config.algorithm.ppo.clip_ratio
        pg = policy_gradient_loss(logp, old_logp, adv, clip, mask=mask_t)

        old_values = batch.tensors.get("old_values", values.detach())
        vf = value_loss(values, old_values, rets, clip, mask=mask_t)

        loss = pg + 0.5 * vf

        def publish(value: Tensor | float) -> Tensor | float:
            if metrics_on_device:
                if torch.is_tensor(value):
                    return value.detach()
                return torch.tensor(float(value), device=loss.device, dtype=loss.dtype)
            if torch.is_tensor(value):
                return float(value.detach().cpu())
            return float(value)

        metrics: dict[str, Any] = {"loss_pg": publish(pg), "loss_vf": publish(vf)}

        kl_c = self._config.algorithm.ppo.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl_mask = None if mask_t is None else mask_t.bool()
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=kl_mask)
            loss = loss + kl_c * kl
            metrics["kl"] = publish(kl)

        with torch.no_grad():
            if mask_t is None:
                explained = 1.0 - F.mse_loss(values, rets) / torch.var(rets.float()).clamp_min(1e-8)
            else:
                active = mask_t.bool()
                # bool() on a device scalar synchronizes. Keep it on the historical (host metrics) path only.
                has_tokens = True if metrics_on_device else bool(mask_t.sum() > 0)
                explained = (
                    1.0
                    - F.mse_loss(values[active], rets[active])
                    / torch.var(rets[active].float()).clamp_min(1e-8)
                    if has_tokens
                    else 0.0
                )
            metrics["explained_variance"] = publish(explained)

        metrics["loss_total"] = publish(loss)
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.PPO.value, PPOAlgorithm)
