"""Policy (actor) worker: training backends and log-prob computation."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.nn.functional as F

from lumenrl.core.protocol import DataProto
from lumenrl.core.types import TrainingBackend
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
        """Single PPO / GRPO style gradient step; returns scalar diagnostics."""
        if self._model is None or self._optimizer is None:
            raise RuntimeError("init_model() must be called before train_step().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        self._model.train()
        input_ids = batch["input_ids"].to(self._device)
        logits = self._model(input_ids)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        adv = batch.tensors.get("advantages")
        if adv is not None:
            adv = adv.to(self._device)
            if adv.shape == shift_labels.shape:
                per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view_as(shift_labels)
                loss = (per_token * adv[:, 1:]).mean()

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()

        metrics: dict[str, float] = {"loss": float(loss.detach().cpu())}
        self._log.debug("train_step loss=%f", metrics["loss"])
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
