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
# Original: verl/workers/engine/megatron/transformer_impl.py

"""Megatron-Core training engine implementation.

Provides full Megatron-Core integration with tensor parallelism (TP),
pipeline parallelism (PP/VPP), context parallelism (CP), expert
parallelism (EP), and distributed optimizer support.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from lumenrl.core.config import McoreEngineConfig, OptimizerConfig
from lumenrl.engine.training.base_engine import BaseEngine, BaseEngineCtx, EngineRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "WARN"))


def _set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MegatronEngine(BaseEngine):
    """Training engine backed by Megatron-Core.

    Supports TP, PP (1F1B and interleaved VPP), CP, EP, sequence
    parallelism, and distributed optimizer via Megatron's native APIs.

    Requires ``megatron-core`` and ``megatron-bridge`` packages.
    """

    def __init__(
        self,
        model_config: Any,
        engine_config: McoreEngineConfig | dict[str, Any],
        optimizer_config: OptimizerConfig | dict[str, Any],
        model_name: str = "",
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.engine_config = (
            engine_config
            if isinstance(engine_config, McoreEngineConfig)
            else McoreEngineConfig(**engine_config)
        )
        self.optimizer_config = (
            optimizer_config
            if isinstance(optimizer_config, OptimizerConfig)
            else OptimizerConfig(**optimizer_config)
        )
        self.model_name = model_name

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_offload_grad = self.engine_config.grad_offload

        self.mode: str | None = None
        self.module: list[nn.Module] | None = None
        self.optimizer: Any = None
        self.lr_scheduler: Any = None

        self.bridge: Any = None
        self.tf_config: Any = None
        self.is_value_model: bool = False

        _set_random_seed(self.engine_config.seed)

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build Megatron model, set up parallelism, create optimizer."""
        self._init_device_mesh()
        self._build_tf_config()
        self._build_megatron_module()

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_lr_scheduler()

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

    # Adapted from verl (https://github.com/verl-project/verl)
    # Original: verl/workers/engine/megatron/transformer_impl.py::_init_device_mesh
    def _init_device_mesh(self) -> None:
        """Initialize Megatron-Core parallel state with TP/PP/CP/EP."""
        try:
            from megatron.core import parallel_state as mpu
        except ImportError as e:
            raise ImportError(
                "megatron-core is required for MegatronEngine. "
                "Install it with: pip install megatron-core"
            ) from e

        if mpu.is_initialized():
            return

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=(
                self.engine_config.virtual_pipeline_model_parallel_size
            ),
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            nccl_communicator_config_path=None,
        )

    # Adapted from verl (https://github.com/verl-project/verl)
    # Original: verl/workers/engine/megatron/transformer_impl.py::_build_tf_config
    def _build_tf_config(self) -> None:
        """Build Megatron TransformerConfig from HF config via megatron-bridge."""
        try:
            from megatron.bridge.models import AutoBridge
        except ImportError:
            try:
                from megatron.core.models.bridge import AutoBridge
            except ImportError:
                logger.warning(
                    "megatron-bridge not available; using stub transformer config."
                )
                self.tf_config = None
                self.bridge = None
                return

        hf_model_path = self.model_name or getattr(self.model_config, "local_path", "")
        trust_remote = getattr(self.model_config, "trust_remote_code", True)

        dtype_str = self.engine_config.dtype
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        param_dtype = dtype_map.get(dtype_str, torch.bfloat16)

        bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=trust_remote)
        provider = bridge.to_megatron_provider(load_weights=False)

        from megatron.core.transformer.enums import AttnBackend

        provider_overrides = {
            "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
            "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
            "virtual_pipeline_model_parallel_size": (
                self.engine_config.virtual_pipeline_model_parallel_size
            ),
            "context_parallel_size": self.engine_config.context_parallel_size,
            "sequence_parallel": self.engine_config.sequence_parallel,
            "variable_seq_lengths": True,
            "attention_backend": AttnBackend.flash,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_load_balancing_type": "none",
        }

        provider.apply_overrides_and_finalize(
            dtype=param_dtype,
            overrides=provider_overrides,
        )
        self.bridge = bridge
        self.provider = provider
        self.tf_config = None
        self.param_dtype = param_dtype

    # Adapted from verl (https://github.com/verl-project/verl)
    # Original: verl/workers/engine/megatron/transformer_impl.py::_build_megatron_module
    def _build_megatron_module(self) -> None:
        """Build and load the Megatron GPTModel, then wrap with DDP."""
        from megatron.core import parallel_state as mpu

        if self.bridge is None:
            from lumenrl.engine.training.megatron_backend import _MegatronStubLM
            self.module = [_MegatronStubLM()]
            logger.warning("MegatronEngine: using stub model (megatron-bridge not available)")
            return

        hf_model_path = self.model_name or getattr(self.model_config, "local_path", "")
        model_type = getattr(self.model_config, "model_type", "language_model")
        self.is_value_model = model_type == "value_model"

        wrap_with_ddp = not self.engine_config.forward_only

        try:
            from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        except ImportError:
            raise ImportError(
                "verl.utils.megatron_utils is required for Megatron model construction. "
                "Ensure verl is installed or megatron_utils is available."
            )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=self.is_value_model,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
        )

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=getattr(self.model_config, "hf_config", None),
            bridge=self.bridge,
            provider=self.provider,
        )
        self.tf_config = updated_tf_config

        allowed_mismatched = ["output_layer.weight"] if self.is_value_model else []
        self.bridge.load_hf_weights(
            module,
            hf_model_path,
            allowed_mismatched_params=allowed_mismatched,
        )

        self.module = module
        if dist.get_rank() == 0:
            total_params = sum(
                p.numel() for m in (module if isinstance(module, list) else [module]) for p in m.parameters()
            )
            logger.info(
                "MegatronEngine: built model with %d params (TP=%d PP=%d CP=%d EP=%d)",
                total_params,
                mpu.get_tensor_model_parallel_world_size(),
                mpu.get_pipeline_model_parallel_world_size(),
                getattr(mpu, "get_context_parallel_world_size", lambda: 1)(),
                getattr(mpu, "get_expert_model_parallel_world_size", lambda: 1)(),
            )

    def _build_optimizer(self) -> Any:
        """Create optimizer for Megatron module."""
        cfg = self.optimizer_config
        if self.module is None:
            return None

        params = []
        modules = self.module if isinstance(self.module, list) else [self.module]
        for m in modules:
            params.extend(p for p in m.parameters() if p.requires_grad)

        if cfg.optimizer_type == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        raise NotImplementedError(f"Unsupported optimizer: {cfg.optimizer_type}")

    def _build_lr_scheduler(self) -> Any:
        if self.optimizer is None:
            return None
        from lumenrl.utils.lr_scheduler import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )
        cfg = self.optimizer_config
        num_warmup = cfg.lr_warmup_steps
        if num_warmup <= 0:
            num_warmup = int(cfg.lr_warmup_steps_ratio * cfg.total_training_steps)
        if cfg.lr_scheduler_type == "constant":
            return get_constant_schedule_with_warmup(self.optimizer, num_warmup)
        elif cfg.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
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
        return _MegatronTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs) -> ContextManager:
        return _MegatronEvalModeCtx(self, **kwargs)

    # ------------------------------------------------------------------
    # Data-parallel helpers
    # ------------------------------------------------------------------

    def get_data_parallel_rank(self) -> int:
        try:
            from megatron.core import parallel_state as mpu
            return mpu.get_data_parallel_rank()
        except (ImportError, AssertionError):
            return dist.get_rank() if dist.is_initialized() else 0

    def get_data_parallel_size(self) -> int:
        try:
            from megatron.core import parallel_state as mpu
            return mpu.get_data_parallel_world_size()
        except (ImportError, AssertionError):
            return dist.get_world_size() if dist.is_initialized() else 1

    def get_data_parallel_group(self):
        try:
            from megatron.core import parallel_state as mpu
            return mpu.get_data_parallel_group()
        except (ImportError, AssertionError):
            return dist.group.WORLD if dist.is_initialized() else None

    def is_mp_src_rank_with_outputs(self) -> bool:
        """True only on the last PP stage's TP rank 0."""
        try:
            from megatron.core import parallel_state as mpu
            is_last_pp = mpu.is_pipeline_last_stage()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            return is_last_pp and tp_rank == 0
        except (ImportError, AssertionError):
            return True

    # ------------------------------------------------------------------
    # Forward / backward (with Pipeline Parallelism)
    # ------------------------------------------------------------------

    # Adapted from verl (https://github.com/verl-project/verl)
    # Original: verl/workers/engine/megatron/transformer_impl.py::forward_backward_batch
    def forward_backward_batch(
        self,
        data: dict[str, Any],
        loss_function: Callable,
        forward_only: bool = False,
    ) -> dict[str, Any]:
        """Run forward/backward across micro-batches using Megatron's PP scheduler.

        Uses ``get_forward_backward_func()`` which handles 1F1B scheduling
        for PP>1 and interleaved scheduling for VPP.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func

        micro_batches = self._prepare_micro_batches(data)
        num_microbatches = len(micro_batches)

        pp_size = mpu.get_pipeline_model_parallel_world_size()

        if pp_size <= 1:
            return self._forward_backward_no_pp(
                micro_batches, loss_function, forward_only,
            )

        vpp_size = self.engine_config.virtual_pipeline_model_parallel_size or 1
        batch_gen = _make_batch_generator(micro_batches, vpp_size)

        forward_backward_func = get_forward_backward_func()

        def forward_step_func(data_iterator, model):
            mb = next(data_iterator)
            return self.forward_step(mb, loss_function, forward_only)

        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=batch_gen,
            model=self.module,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=data.get("seq_length", 1),
            micro_batch_size=micro_batches[0]["input_ids"].shape[0] if micro_batches else 1,
        )

        return self._postprocess_pp_outputs(losses)

    def _forward_backward_no_pp(
        self,
        micro_batches: list[dict[str, Any]],
        loss_function: Callable,
        forward_only: bool,
    ) -> dict[str, Any]:
        """Simple forward/backward without pipeline parallelism."""
        output_lst: list[dict] = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for mb in micro_batches:
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
        """Single micro-batch forward pass."""
        modules = self.module if isinstance(self.module, list) else [self.module]
        model = modules[0] if len(modules) == 1 else modules[-1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = micro_batch["input_ids"].to(device)
        attention_mask = micro_batch.get("attention_mask")
        position_ids = micro_batch.get("position_ids")

        model_kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.to(device)
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids.to(device)

        raw_output = model(**model_kwargs)
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
        from lumenrl.utils.torch_functional import (
            calculate_sum_pi_squared_from_logits,
            entropy_from_logits,
            logprobs_from_logits,
        )

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
        micro_batch_size = data.get("micro_batch_size")
        if micro_batch_size is not None and micro_batch_size > 0:
            input_ids = data["input_ids"]
            total = input_ids.shape[0]
            batches = []
            for start in range(0, total, micro_batch_size):
                end = min(start + micro_batch_size, total)
                mb = {
                    k: v[start:end] if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                }
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

    def _postprocess_pp_outputs(self, losses: Any) -> dict[str, Any]:
        """Postprocess outputs from Megatron's PP forward_backward_func."""
        if self.is_mp_src_rank_with_outputs() and losses:
            avg_loss = sum(float(l) for l in losses) / len(losses) if losses else 0.0
        else:
            avg_loss = 0.0
        return {
            "model_output": {},
            "loss": [avg_loss],
            "metrics": {},
        }

    # ------------------------------------------------------------------
    # Optimizer / LR
    # ------------------------------------------------------------------

    def optimizer_zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self) -> float:
        assert self.optimizer is not None
        clip = self.optimizer_config.clip_grad

        modules = self.module if isinstance(self.module, list) else [self.module]
        params = [p for m in modules for p in m.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=clip)

        if not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite: %s — skipping update", grad_norm)
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()

        return float(grad_norm)

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

        modules = self.module if isinstance(self.module, list) else [self.module]

        if device in ("cuda", "gpu"):
            if model:
                for m in modules:
                    m.cuda()
            if optimizer and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        elif device == "cpu":
            if model:
                for m in modules:
                    m.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if optimizer and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to("cpu", non_blocking=True)

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
            self.to("cuda", model=True, optimizer=False, grad=False)

        modules = self.module if isinstance(self.module, list) else [self.module]
        state: dict[str, Any] = {
            "global_step": global_step,
        }
        for i, m in enumerate(modules):
            state[f"model_{i}"] = m.state_dict()
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        save_path = os.path.join(local_path, f"step_{global_step}")
        os.makedirs(save_path, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(state, os.path.join(save_path, f"rank_{rank}.pt"))

        if dist.is_initialized():
            dist.barrier()

        if self._is_offload_param:
            self.to("cpu", model=True, optimizer=False, grad=False)

    def load_checkpoint(self, local_path: str, **kwargs: Any) -> None:
        assert self.module is not None
        if self._is_offload_param:
            self.to("cuda", model=True, optimizer=False, grad=False)

        rank = dist.get_rank() if dist.is_initialized() else 0
        ckpt_path = os.path.join(local_path, f"rank_{rank}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found: %s", ckpt_path)
            return

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        modules = self.module if isinstance(self.module, list) else [self.module]
        for i, m in enumerate(modules):
            key = f"model_{i}"
            if key in state:
                m.load_state_dict(state[key])
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if dist.is_initialized():
            dist.barrier()

        if self._is_offload_param:
            self.to("cpu", model=True, optimizer=False, grad=False)

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_per_tensor_param(self, **kwargs):
        """Export model weights, optionally converting to HF format via bridge."""
        assert self.module is not None

        if self._is_offload_param:
            self.to("cuda", model=True, optimizer=False, grad=False)

        modules = self.module if isinstance(self.module, list) else [self.module]

        if self.bridge is not None and hasattr(self.bridge, "export_hf_weights"):
            hf_state = self.bridge.export_hf_weights(modules)
            if self._is_offload_param:
                self.to("cpu", model=True, optimizer=False, grad=False)
            return hf_state.items(), None

        combined = {}
        for i, m in enumerate(modules):
            for name, param in m.state_dict().items():
                combined[f"module_{i}.{name}"] = param
        if self._is_offload_param:
            self.to("cpu", model=True, optimizer=False, grad=False)
        return combined.items(), None

    def disable_adapter(self) -> ContextManager:
        modules = self.module if isinstance(self.module, list) else [self.module]
        if hasattr(modules[0], "disable_adapter"):
            return modules[0].disable_adapter()
        return nullcontext()


# ------------------------------------------------------------------
# Registered engine variants
# ------------------------------------------------------------------


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    """Language-model variant for Megatron backend."""
    pass


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngine):
    """Value-model variant for Megatron backend."""
    pass


# ------------------------------------------------------------------
# Mode context managers
# ------------------------------------------------------------------


class _MegatronTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        super().__enter__()
        modules = self.engine.module
        if modules is not None:
            for m in (modules if isinstance(modules, list) else [modules]):
                m.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_val, exc_tb)


class _MegatronEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        modules = self.engine.module
        if modules is not None:
            for m in (modules if isinstance(modules, list) else [modules]):
                m.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)


# ------------------------------------------------------------------
# PP utilities
# ------------------------------------------------------------------


def _make_batch_generator(
    micro_batches: list[dict[str, Any]],
    vpp_size: int = 1,
):
    """Create a data iterator compatible with Megatron's PP scheduler."""
    idx = 0

    def gen():
        nonlocal idx
        while idx < len(micro_batches):
            yield micro_batches[idx]
            idx += 1

    return iter(gen())
