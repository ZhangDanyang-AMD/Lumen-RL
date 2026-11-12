"""FSDP2 engine for VLA policies (OpenVLA-OFT).

Reuses ``FSDP2Engine`` for optimizer / LR schedule / micro-batching / offload
/ checkpointing and replaces the two model-specific pieces: building the
policy (FP32 master weights, transformer blocks sharded individually) and the
per-micro-batch forward, which returns per-action-token log-probs instead of
next-token LM log-probs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from lumenrl.core.config import FSDPEngineConfig, HFModelConfig, OptimizerConfig
from lumenrl.utils.lumen_patch import apply_lumen
from lumenrl.engine.training.fsdp_backend import _resolve_dtype
from lumenrl.engine.training.fsdp_engine import FSDP2Engine

logger = logging.getLogger(__name__)


def load_policy_weights(module: nn.Module, path: str) -> None:
    """Load a ``policy.safetensors`` of named parameters onto ``module``.

    Buffers (for example the rotary ``inv_freq``) stay as the base checkpoint
    left them. The file is the bf16 rollout replica, so an fp32 training
    module receives a cast copy.
    """
    from safetensors.torch import load_file

    saved = load_file(path, device="cpu")
    params = dict(module.named_parameters())
    missing = [name for name in params if name not in saved]
    unexpected = [name for name in saved if name not in params]
    if missing or unexpected:
        raise RuntimeError(
            f"{path} does not match this policy: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )
    module.load_state_dict(saved, strict=False)
    logger.info("Loaded %d policy tensors from %s", len(saved), path)


class EmbodiedFSDP2Engine(FSDP2Engine):
    def __init__(
        self,
        model_path: str,
        engine_config: FSDPEngineConfig | dict[str, Any],
        optimizer_config: OptimizerConfig | dict[str, Any],
        build_policy: Callable[..., nn.Module],
        processor: Any,
        gradient_checkpointing: bool = True,
        freeze_vision_backbone: bool = False,
        lumen_kwargs: Optional[dict[str, Any]] = None,
        resume_from: str = "",
    ) -> None:
        super().__init__(
            model_config=HFModelConfig(local_path=model_path, enable_gradient_checkpointing=gradient_checkpointing),
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            model_name=model_path,
        )
        self.empty_cache_per_micro_batch = False  # fixed-shape batches; see FSDP2Engine
        self.freeze_vision_backbone = freeze_vision_backbone
        self.lumen_kwargs = lumen_kwargs or {}
        self.resume_from = resume_from
        self.build_policy = build_policy
        self.processor = processor

    def _build_model_optimizer(self) -> None:
        module = self.build_policy(self.model_name, dtype=torch.float32, device="cpu")
        if self.resume_from:
            load_policy_weights(module, self.resume_from)
        if self.freeze_vision_backbone:
            module.vision_backbone.requires_grad_(False)
        if self.model_config.enable_gradient_checkpointing:
            module.gradient_checkpointing_enable()
        module = apply_lumen(module, self.lumen_kwargs, "actor")
        self.module = self._shard(module)
        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(self.module)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    def _shard(self, module: nn.Module) -> nn.Module:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        module.to(device)
        if not dist.is_initialized():
            return module
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        fsdp = self._make_fsdp_config()
        mp = MixedPrecisionPolicy(
            param_dtype=_resolve_dtype(fsdp["param_dtype"], torch.bfloat16),
            reduce_dtype=_resolve_dtype(fsdp["reduce_dtype"], torch.float32),
        )
        offload = None
        if fsdp.get("param_offload"):
            from torch.distributed.fsdp import CPUOffloadPolicy

            offload = CPUOffloadPolicy(pin_memory=True)
        kw = dict(mp_policy=mp, reshard_after_forward=fsdp["reshard_after_forward"], offload_policy=offload)
        for block in module.fsdp_wrap_modules():
            fully_shard(block, **kw)
        fully_shard(module, **kw)
        logger.info("[rank %d] VLA policy sharded with FSDP2 (%d blocks)", self.rank, len(module.fsdp_wrap_modules()))
        return module

    def forward_step(
        self,
        micro_batch: dict[str, Any],
        loss_function: Optional[Callable],
        forward_only: bool,
    ) -> tuple[torch.Tensor, dict]:
        assert self.module is not None
        device = self._device_or_cuda()
        meta = micro_batch.get("meta", {}) or {}
        out = self.module(
            input_ids=micro_batch["input_ids"].to(device),
            attention_mask=micro_batch["attention_mask"].to(device),
            pixel_values=self.processor.preprocess_images(micro_batch["images"].to(device)),
            action_tokens=micro_batch["action_tokens"].to(device),
            temperature=float(micro_batch.get("temperature", 1.0)),
            calculate_entropy=bool(meta.get("calculate_entropy", False)),
        )
        model_output = {k: v for k, v in out.items() if k != "logits"}
        if loss_function is not None:
            loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        else:
            assert forward_only
            loss, metrics = torch.zeros((), device=device), {}
        return loss, {"model_output": model_output, "loss": loss.detach().item(), "metrics": metrics}

    @torch.no_grad()
    def iter_full_params(self, dtype: torch.dtype = torch.bfloat16):
        """Yield ``(name, full tensor)`` for every parameter; collective on all ranks."""
        from torch.distributed.tensor import DTensor

        assert self.module is not None
        for name, p in self.module.named_parameters():
            t = p.detach().to(dtype)
            if isinstance(t, DTensor):
                t = t.full_tensor()
            yield name, t
