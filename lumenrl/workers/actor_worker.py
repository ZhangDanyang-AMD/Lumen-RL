"""Policy (actor) worker: training backends and log-prob computation."""

from __future__ import annotations

import logging
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
from lumenrl.engine.training.base_engine import BaseEngine, EngineRegistry
from lumenrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class LumenActorWorker(BaseWorker):
    """Trainable policy worker using the Engine abstraction layer.

    Delegates model construction, optimizer, LR scheduling, and offload
    management to the Engine layer (FSDP2 or Megatron).
    """

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(rank, world_size, config)
        self._engine: BaseEngine | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Build policy network via EngineRegistry."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        backend_raw = str(
            policy.get("training_backend", TrainingBackend.FSDP2.value)
        ).lower()

        if backend_raw in ("fsdp", "fsdp2"):
            backend_key = "fsdp2"
        elif backend_raw == "megatron":
            backend_key = "megatron"
        else:
            raise ValueError(f"Unknown policy.training_backend: {backend_raw}")

        model_name = str(policy.get("model_name", ""))
        training_cfg = policy.get("training", {}) or {}
        quant = get_nested_config(self.config, "quantization", "training", default={}) or {}

        engine_config = self._build_engine_config(backend_key, training_cfg, policy)
        optimizer_config = self._build_optimizer_config(policy)
        model_config = self._build_model_config(policy)

        engine_cls = EngineRegistry.get_engine_cls(
            model_type="language_model",
            backend=backend_key,
        )

        if backend_key in ("fsdp", "fsdp2"):
            self._engine = engine_cls(
                model_config=model_config,
                engine_config=engine_config,
                optimizer_config=optimizer_config,
                model_name=model_name,
                quant_config=quant,
            )
        else:
            self._engine = engine_cls(
                model_config=model_config,
                engine_config=engine_config,
                optimizer_config=optimizer_config,
                model_name=model_name,
            )

        self._engine.initialize()
        self._log.info("LumenActorWorker: initialized %s engine.", backend_key)

    def _build_engine_config(
        self, backend: str, training_cfg: dict, policy: dict,
    ) -> dict[str, Any]:
        if backend in ("fsdp", "fsdp2"):
            fsdp_cfg = training_cfg.get("fsdp_cfg") or {}
            if not isinstance(fsdp_cfg, dict):
                from dataclasses import asdict, is_dataclass
                fsdp_cfg = asdict(fsdp_cfg) if is_dataclass(fsdp_cfg) else dict(vars(fsdp_cfg))
            return {
                "param_offload": fsdp_cfg.get("param_offload", False),
                "optimizer_offload": fsdp_cfg.get("optimizer_offload", False),
                "grad_offload": fsdp_cfg.get("grad_offload", False),
                "reshard_after_forward": fsdp_cfg.get("reshard_after_forward", True),
                "model_dtype": training_cfg.get("optimizer_dtype", "bf16"),
                "seed": int(policy.get("seed", 42)),
            }
        elif backend == "megatron":
            meg_cfg = training_cfg.get("megatron_cfg") or policy.get("megatron_cfg") or {}
            if not isinstance(meg_cfg, dict):
                from dataclasses import asdict, is_dataclass
                meg_cfg = asdict(meg_cfg) if is_dataclass(meg_cfg) else dict(vars(meg_cfg))
            return {
                "tensor_model_parallel_size": meg_cfg.get("tensor_model_parallel_size", 1),
                "pipeline_model_parallel_size": meg_cfg.get("pipeline_model_parallel_size", 1),
                "context_parallel_size": meg_cfg.get("context_parallel_size", 1),
                "expert_model_parallel_size": meg_cfg.get("expert_model_parallel_size", 1),
                "sequence_parallel": meg_cfg.get("sequence_parallel", False),
                "param_offload": meg_cfg.get("param_offload", False),
                "optimizer_offload": meg_cfg.get("optimizer_offload", False),
                "grad_offload": meg_cfg.get("grad_offload", False),
                "seed": int(policy.get("seed", 42)),
                "dtype": meg_cfg.get("dtype", "bf16"),
                "use_distributed_optimizer": meg_cfg.get("use_distributed_optimizer", False),
            }
        return {}

    def _build_optimizer_config(self, policy: dict) -> dict[str, Any]:
        lr = float(policy.get("learning_rate", policy.get("lr", 1e-6)))
        return {
            "lr": lr,
            "weight_decay": float(policy.get("weight_decay", 0.01)),
            "clip_grad": float(policy.get("max_grad_norm", 1.0)),
            "lr_warmup_steps": int(policy.get("lr_warmup_steps", 10)),
            "lr_warmup_steps_ratio": float(policy.get("warmup_ratio", 0.0)),
        }

    def _build_model_config(self, policy: dict) -> dict[str, Any]:
        return {
            "local_path": str(policy.get("model_name", "")),
            "trust_remote_code": True,
        }

    def compute_log_probs(self, batch: DataProto) -> DataProto:
        """Return per-token log probabilities for the policy."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before compute_log_probs().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        data = {"input_ids": batch["input_ids"].to(self._device)}
        if "attention_mask" in batch:
            data["attention_mask"] = batch["attention_mask"].to(self._device)

        with self._engine.eval_mode():
            output = self._engine.infer_batch(data)

        log_probs = output["model_output"]["log_probs"]
        out = DataProto(
            tensors={"log_probs": log_probs.cpu(), "input_ids": batch["input_ids"]},
            meta=dict(batch.meta),
        )
        return out

    def train_step(self, batch: DataProto) -> dict[str, float]:
        """Compute the RL surrogate loss, backward, and step the optimizer.

        The worker recomputes the forward pass locally so gradients flow
        through the model parameters.  The algorithm name and hyperparameters
        are passed via ``batch.meta``.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before train_step().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        def loss_fn(model_output, data):
            token_log_probs = model_output["log_probs"]
            input_ids = data["input_ids"]

            old_logp = batch.tensors.get("old_log_probs")
            adv = batch.tensors.get("advantages")
            algo_name = str(batch.meta.get("algorithm", "grpo")).lower()

            if old_logp is not None and adv is not None:
                old_logp = old_logp.to(token_log_probs.device)
                adv = adv.to(token_log_probs.device)

                if adv.dim() == 1:
                    adv = adv.unsqueeze(-1).expand_as(token_log_probs)
                elif adv.shape[-1] != token_log_probs.shape[-1]:
                    adv = adv[..., :token_log_probs.shape[-1]]

                if old_logp.shape[-1] != token_log_probs.shape[-1]:
                    old_logp = old_logp[..., :token_log_probs.shape[-1]]

                mask = batch.tensors.get("attention_mask")
                if mask is not None:
                    mask = mask.to(token_log_probs.device)[..., :token_log_probs.shape[-1]].float()

                algo_cfg = batch.meta.get("algo_config", {})
                if algo_name == AlgorithmName.DAPO.value:
                    clip_low = float(algo_cfg.get("clip_ratio_low", 0.2))
                    clip_high = float(algo_cfg.get("clip_ratio_high", 0.28))
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
                    ref_logp = ref_logp.to(token_log_probs.device)
                    if ref_logp.shape[-1] != token_log_probs.shape[-1]:
                        ref_logp = ref_logp[..., :token_log_probs.shape[-1]]
                    kl = kl_penalty(token_log_probs, ref_logp, mask=mask)
                    loss = loss + kl_c * kl
            else:
                shift_labels = input_ids[:, 1:].contiguous().to(token_log_probs.device)
                loss = F.cross_entropy(
                    token_log_probs.view(-1),
                    shift_labels.view(-1),
                )

            return loss, {"loss": float(loss.detach())}

        data = {"input_ids": batch["input_ids"].to(self._device)}
        if "attention_mask" in batch:
            data["attention_mask"] = batch["attention_mask"].to(self._device)

        with self._engine.train_mode():
            output = self._engine.train_batch(data, loss_fn)

        metrics: dict[str, float] = {}
        if "metrics" in output:
            for k, v in output["metrics"].items():
                if isinstance(v, list):
                    metrics[k] = sum(v) / max(len(v), 1)
                else:
                    metrics[k] = float(v)

        if "loss" in output:
            if isinstance(output["loss"], list):
                metrics["loss"] = sum(output["loss"]) / max(len(output["loss"]), 1)
            else:
                metrics["loss"] = float(output["loss"])

        lr = self._engine.lr_scheduler_step()
        metrics["lr"] = lr

        algo_name = str(batch.meta.get("algorithm", "grpo")).lower()
        self._log.debug("train_step loss=%f (algo=%s)", metrics.get("loss", 0.0), algo_name)
        return metrics

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU state dict for weight sync to rollout engines."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before get_state_dict().")
        params, _ = self._engine.get_per_tensor_param()
        return {k: v.detach().cpu() for k, v in params}

    def cleanup(self) -> None:
        self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
