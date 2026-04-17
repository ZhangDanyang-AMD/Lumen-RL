"""Policy (actor) worker: training backends and log-prob computation."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.nn.functional as F

from lumenrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    kl_penalty,
    policy_gradient_loss,
)
from lumenrl.core.protocol import DataProto
from lumenrl.core.types import AlgorithmName, TrainingBackend
from lumenrl.engine.training.fsdp_backend import FSDP2Backend
from lumenrl.engine.training.megatron_backend import MegatronBackend
from lumenrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class LumenActorWorker(BaseWorker):
    """Trainable policy worker using FSDP2 or Megatron backends."""

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(rank, world_size, config)
        self._model: torch.nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Build policy network from ``config.policy.training_backend``."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        backend_raw = str(
            policy.get("training_backend", TrainingBackend.FSDP2.value)
        ).lower()
        training_cfg = policy.get("training", {}) or {}
        quant = get_nested_config(self.config, "quantization", "training", default={}) or {}

        if backend_raw in (TrainingBackend.FSDP2.value, "fsdp", "fsdp2"):
            model = FSDP2Backend.build_model(str(policy.get("model_name", "")), training_cfg)
            model = FSDP2Backend.apply_lumen_optimizations(model, quant)
            fsdp_cfg = training_cfg.get("fsdp_cfg") or {}
            model = FSDP2Backend.apply_fsdp2(model, fsdp_cfg)
            self._log.info("LumenActorWorker: initialized FSDP2 backend.")
        elif backend_raw in (TrainingBackend.MEGATRON.value, "megatron"):
            model = MegatronBackend.build_model(str(policy.get("model_name", "")), training_cfg)
            meg_cfg = training_cfg.get("megatron_cfg") or policy.get("megatron_cfg") or {}
            if is_dataclass(meg_cfg):
                meg_cfg = asdict(meg_cfg)
            elif not isinstance(meg_cfg, dict):
                meg_cfg = dict(vars(meg_cfg))
            model = MegatronBackend.apply_lumen_spec(model, dict(meg_cfg))
            self._log.info("LumenActorWorker: initialized Megatron backend.")
        else:
            raise ValueError(f"Unknown policy.training_backend: {backend_raw}")

        self._model = model.to(self._device)
        lr = float(policy.get("lr", 1e-4))
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

    def compute_log_probs(self, batch: DataProto) -> DataProto:
        """Return per-token log probabilities for the policy."""
        if self._model is None:
            raise RuntimeError("init_model() must be called before compute_log_probs().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        self._model.eval()
        input_ids = batch["input_ids"].to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids)
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        out = DataProto(
            tensors={"log_probs": token_log_probs.cpu(), "input_ids": batch["input_ids"]},
            meta=dict(batch.meta),
        )
        return out

    def train_step(self, batch: DataProto) -> dict[str, float]:
        """Compute the RL surrogate loss, backward, and step the optimizer.

        The worker recomputes the forward pass locally so gradients flow
        through the model parameters.  The algorithm name and hyperparameters
        are passed via ``batch.meta``.
        """
        if self._model is None or self._optimizer is None:
            raise RuntimeError("init_model() must be called before train_step().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        self._model.train()
        input_ids = batch["input_ids"].to(self._device)
        logits = self._model(input_ids)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        old_logp = batch.tensors.get("old_log_probs")
        adv = batch.tensors.get("advantages")
        algo_name = str(batch.meta.get("algorithm", "grpo")).lower()

        if old_logp is not None and adv is not None:
            old_logp = old_logp.to(self._device)
            adv = adv.to(self._device)

            if adv.dim() == 1:
                adv = adv.unsqueeze(-1).expand_as(token_log_probs)
            elif adv.shape[-1] != token_log_probs.shape[-1]:
                adv = adv[..., : token_log_probs.shape[-1]]

            if old_logp.shape[-1] != token_log_probs.shape[-1]:
                old_logp = old_logp[..., : token_log_probs.shape[-1]]

            mask = batch.tensors.get("attention_mask")
            if mask is not None:
                mask = mask.to(self._device)[..., : token_log_probs.shape[-1]].float()

            algo_cfg = batch.meta.get("algo_config", {})
            if algo_name == AlgorithmName.DAPO.value:
                clip_low = float(algo_cfg.get("clip_ratio_low", 0.8))
                clip_high = float(algo_cfg.get("clip_ratio_high", 1.28))
                loss = asymmetric_clip_loss(
                    token_log_probs, old_logp, adv, clip_low, clip_high, mask=mask,
                )
            else:
                clip = float(algo_cfg.get("clip_ratio", 0.2))
                loss = policy_gradient_loss(
                    token_log_probs, old_logp, adv, clip, mask=mask,
                )

            kl_c = float(algo_cfg.get("kl_coeff", 0.0))
            ref_logp = batch.tensors.get("ref_log_probs")
            if kl_c > 0.0 and ref_logp is not None:
                ref_logp = ref_logp.to(self._device)
                if ref_logp.shape[-1] != token_log_probs.shape[-1]:
                    ref_logp = ref_logp[..., : token_log_probs.shape[-1]]
                kl = kl_penalty(token_log_probs, ref_logp, mask=mask)
                loss = loss + kl_c * kl
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()

        metrics: dict[str, float] = {"loss": float(loss.detach().cpu())}
        self._log.debug("train_step loss=%f (algo=%s)", metrics["loss"], algo_name)
        return metrics

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU state dict for weight sync to rollout engines."""
        if self._model is None:
            raise RuntimeError("init_model() must be called before get_state_dict().")
        return {k: v.detach().cpu() for k, v in self._model.state_dict().items()}

    def cleanup(self) -> None:
        self._model = None
        self._optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
