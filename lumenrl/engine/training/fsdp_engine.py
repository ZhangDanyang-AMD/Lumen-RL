# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/workers/engine/fsdp/transformer_impl.py

"""FSDP2 training engine implementation."""

from __future__ import annotations

import gc
import logging
import os
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from lumenrl.core.config import (
    FSDPEngineConfig,
    HFModelConfig,
    OptimizerConfig,
)
from lumenrl.engine.training.base_engine import BaseEngine, BaseEngineCtx, EngineRegistry
from lumenrl.engine.training.fsdp_backend import (
    FSDP2Backend,
    set_requires_gradient_sync,
)
from lumenrl.utils.fsdp_utils import (
    fsdp2_clip_grad_norm_,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from lumenrl.utils.lr_scheduler import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from lumenrl.utils.torch_functional import (
    calculate_sum_pi_squared_from_logits,
    entropy_from_logits,
    logprobs_from_logits,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "WARN"))


class FSDP2Engine(BaseEngine):
    """Concrete engine implementation backed by PyTorch FSDP2 (``fully_shard``).

    Integrates the existing ``FSDP2Backend`` for model construction and sharding,
    and adds optimizer, LR scheduler, micro-batching, gradient accumulation,
    offload, and checkpoint management on top.
    """

    def __init__(
        self,
        model_config: HFModelConfig | dict[str, Any],
        engine_config: FSDPEngineConfig | dict[str, Any],
        optimizer_config: OptimizerConfig | dict[str, Any],
        model_name: str = "",
        quant_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.model_config = model_config if isinstance(model_config, HFModelConfig) else HFModelConfig(**model_config)
        self.engine_config = engine_config if isinstance(engine_config, FSDPEngineConfig) else FSDPEngineConfig(**engine_config)
        self.optimizer_config = optimizer_config if isinstance(optimizer_config, OptimizerConfig) else OptimizerConfig(**optimizer_config)
        self.model_name = model_name or self.model_config.local_path
        self.quant_config = quant_config or {}

        self.mode: str | None = None
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora.enabled and self.model_config.lora.rank > 0

        self.module: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler = None

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def initialize(self) -> None:
        """Build model, apply FSDP2 sharding, create optimizer and LR scheduler."""
        self._build_model_optimizer()
        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

    def _build_model_optimizer(self) -> None:
        training_cfg = self._make_backend_config()

        module = FSDP2Backend.build_model(self.model_name, training_cfg)
        module = FSDP2Backend.apply_lumen_optimizations(module, self.quant_config)

        if self._is_lora:
            module = self._build_lora_module(module)

        fsdp_cfg = self._make_fsdp_config()
        module = FSDP2Backend.apply_fsdp2(module, fsdp_cfg)

        self.module = module

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(module)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    def _make_backend_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.model_config.local_path:
            cfg["model_dtype"] = self.engine_config.model_dtype
        else:
            cfg["use_tiny_lm"] = True
        return cfg

    def _make_fsdp_config(self) -> dict[str, Any]:
        mp = self.engine_config.mixed_precision or {}
        return {
            "enabled": True,
            "param_dtype": mp.get("param_dtype", self.engine_config.model_dtype),
            "reduce_dtype": mp.get("reduce_dtype", "fp32"),
            "reshard_after_forward": self.engine_config.reshard_after_forward,
            "param_offload": self.engine_config.param_offload,
        }

    def _build_lora_module(self, module: nn.Module) -> nn.Module:
        from peft import LoraConfig as PeftLoraConfig
        from peft import PeftModel, TaskType, get_peft_model

        module.enable_input_require_grads()
        lora = self.model_config.lora

        if lora.adapter_path:
            module = PeftModel.from_pretrained(module, lora.adapter_path, is_trainable=True)
        else:
            lora_cfg = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora.rank,
                lora_alpha=lora.alpha,
                target_modules=lora.target_modules,
                exclude_modules=lora.exclude_modules,
                bias="none",
            )
            module = get_peft_model(module, lora_cfg)
        return module

    def _build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        cfg = self.optimizer_config
        params = [p for p in module.parameters() if p.requires_grad]
        if cfg.optimizer_type == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        raise NotImplementedError(f"Unsupported optimizer: {cfg.optimizer_type}")

    def _build_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg = self.optimizer_config
        num_warmup = cfg.lr_warmup_steps
        if num_warmup <= 0:
            num_warmup = int(cfg.lr_warmup_steps_ratio * cfg.total_training_steps)

        if cfg.lr_scheduler_type == "constant":
            return get_constant_schedule_with_warmup(optimizer, num_warmup)
        elif cfg.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup,
                cfg.total_training_steps,
                min_lr_ratio=cfg.min_lr_ratio,
                num_cycles=cfg.num_cycles,
            )
        raise NotImplementedError(f"LR scheduler type {cfg.lr_scheduler_type} not supported")

    # ------------------------------------------------------------------
    # Mode context managers
    # ------------------------------------------------------------------

    def train_mode(self, **kwargs) -> ContextManager:
        return _EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs) -> ContextManager:
        return _EngineEvalModeCtx(self, **kwargs)

    # ------------------------------------------------------------------
    # Data-parallel helpers
    # ------------------------------------------------------------------

    def get_data_parallel_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def get_data_parallel_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def get_data_parallel_group(self):
        return dist.group.WORLD if dist.is_initialized() else None

    def is_mp_src_rank_with_outputs(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward_backward_batch(
        self,
        data: dict[str, Any],
        loss_function: Callable,
        forward_only: bool = False,
    ) -> dict[str, Any]:
        micro_batches = self._prepare_micro_batches(data)

        output_lst: list[dict] = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for i, mb in enumerate(micro_batches):
            if not forward_only and len(micro_batches) > 1 and self.module is not None:
                is_last = i == len(micro_batches) - 1
                set_requires_gradient_sync(self.module, is_last)

            with ctx:
                loss, meta = self.forward_step(mb, loss_function, forward_only)
                if not forward_only:
                    loss.backward()

            output_lst.append(meta)

        return self._postprocess_batch(output_lst)

    def forward_step(
        self,
        micro_batch: dict[str, Any],
        loss_function: Callable,
        forward_only: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Single micro-batch forward (and optional backward).

        Returns ``(loss_tensor, meta_dict)`` where *meta_dict* contains
        ``model_output``, scalar ``loss``, and ``metrics``.
        """
        assert self.module is not None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = micro_batch["input_ids"].to(device)
        attention_mask = micro_batch.get("attention_mask")
        position_ids = micro_batch.get("position_ids")

        model_kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.to(device)
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids.to(device)

        raw_output = self.module(**model_kwargs)
        logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output

        model_output = self._prepare_model_outputs(logits, input_ids, micro_batch)

        if loss_function is not None:
            loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        else:
            assert forward_only
            loss = torch.tensor(0.0, device=device)
            metrics = {}

        meta = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }
        return loss, meta

    def _prepare_model_outputs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        micro_batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute log_probs (and optionally entropy / sum-pi-squared) from logits."""
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = logprobs_from_logits(shift_logits, shift_labels)
        result: dict[str, torch.Tensor] = {"log_probs": log_probs}

        meta = micro_batch.get("meta", {}) if isinstance(micro_batch, dict) else {}
        if meta.get("calculate_entropy", False):
            result["entropy"] = entropy_from_logits(shift_logits)
        if meta.get("calculate_sum_pi_squared", False):
            result["sum_pi_squared"] = calculate_sum_pi_squared_from_logits(shift_logits)

        return result

    def _prepare_micro_batches(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Split *data* into micro-batches according to ``micro_batch_size`` or ``max_token_len_per_gpu``."""
        micro_batch_size = data.get("micro_batch_size")
        if micro_batch_size is not None and micro_batch_size > 0:
            input_ids = data["input_ids"]
            total = input_ids.shape[0]
            batches = []
            for start in range(0, total, micro_batch_size):
                end = min(start + micro_batch_size, total)
                mb = {k: v[start:end] if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                batches.append(mb)
            return batches
        return [data]

    def _postprocess_batch(self, output_lst: list[dict]) -> dict[str, Any]:
        model_output: dict[str, list] = {}
        losses: list[float] = []
        aggregated_metrics: dict[str, list] = {}

        for o in output_lst:
            if "model_output" in o:
                for key, val in o["model_output"].items():
                    model_output.setdefault(key, []).append(val)
            if "loss" in o:
                losses.append(o["loss"])
            if "metrics" in o:
                for key, val in o["metrics"].items():
                    aggregated_metrics.setdefault(key, []).append(val)

        concat_output: dict[str, torch.Tensor] = {}
        for key, vals in model_output.items():
            if all(isinstance(v, torch.Tensor) for v in vals):
                concat_output[key] = torch.cat(vals, dim=0)
            else:
                concat_output[key] = vals

        return {
            "model_output": concat_output,
            "loss": losses,
            "metrics": aggregated_metrics,
        }

    # ------------------------------------------------------------------
    # Optimizer / LR
    # ------------------------------------------------------------------

    def optimizer_zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self) -> float:
        assert self.optimizer is not None
        assert self.module is not None
        clip = self.optimizer_config.clip_grad

        try:
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=clip)
        except Exception:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite: %s — skipping update", grad_norm)
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()

        return grad_norm.item()

    def lr_scheduler_step(self) -> float:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_last_lr()[0]
        return 0.0

    # ------------------------------------------------------------------
    # Offload / device movement
    # ------------------------------------------------------------------

    def to(
        self,
        device: str,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        if self.module is None:
            return
        if device in ("cuda", "gpu"):
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        assert self.module is not None
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        state = {
            "model": self.module.state_dict(),
            "global_step": global_step,
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        save_path = os.path.join(local_path, f"step_{global_step}")
        os.makedirs(save_path, exist_ok=True)
        rank = self.rank
        torch.save(state, os.path.join(save_path, f"rank_{rank}.pt"))

        if dist.is_initialized():
            dist.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(self, local_path: str, **kwargs: Any) -> None:
        assert self.module is not None
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        rank = self.rank
        ckpt_path = os.path.join(local_path, f"rank_{rank}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found: %s", ckpt_path)
            return

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.module.load_state_dict(state["model"])
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if dist.is_initialized():
            dist.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_fsdp_optimizer(self.optimizer)

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_per_tensor_param(self, **kwargs):
        assert self.module is not None
        load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        return params.items(), None

    def disable_adapter(self) -> ContextManager:
        if hasattr(self.module, "disable_adapter"):
            return self.module.disable_adapter()
        return nullcontext()


# ------------------------------------------------------------------
# Registered engine variants
# ------------------------------------------------------------------


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"])
class FSDP2EngineWithLMHead(FSDP2Engine):
    """Language-model variant — the default for policy/actor training."""
    pass


# ------------------------------------------------------------------
# Mode context managers
# ------------------------------------------------------------------


class _EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDP2Engine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        super().__enter__()
        if self.engine.module is not None:
            self.engine.module.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.engine, FSDP2Engine):
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_val, exc_tb)


class _EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDP2Engine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        if self.engine.module is not None:
            self.engine.module.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
