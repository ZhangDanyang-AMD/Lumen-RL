# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Megatron-Core training engine for LumenRL (BF16, TP/PP/EP/DP).

Builds a Megatron-Core ``GPTModel`` (Qwen3 dense / Qwen3 MoE), loads HF
weights, and runs the DAPO/GRPO RL step through Megatron modules while
plugging into LumenRL's Ray controller via the ``BaseEngine`` interface.

Supports:
- Tensor Parallel (TP), Expert Parallel (EP), Expert Tensor Parallel (ETP)
- Pipeline Parallel (PP) with Megatron-Core's 1F1B schedule
- Distributed optimizer, sequence parallel, activation recompute
- Weight sync to ATOM with TP/EP/PP all-gather
- Shared experts (Qwen3.5-MoE)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lumenrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    kl_penalty,
    policy_gradient_loss,
)
from lumenrl.core.protocol import DataProto
from lumenrl.core.types import AlgorithmName
from lumenrl.engine.training.base_engine import EngineRegistry
from lumenrl.engine.training.megatron_base_engine import MegatronBaseEngine
from lumenrl.engine.training.qwen3_megatron_bridge import (
    Qwen3Dims,
    hf_to_megatron,
    load_hf_safetensors,
    megatron_to_hf,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "INFO"))

LUMENRL_DEBUG = os.environ.get("LUMENRL_DEBUG", "0") in ("1", "true", "True")

import math  # noqa: E402

try:
    from flash_attn import (
        flash_attn_func as _flash_attn_func,
        flash_attn_varlen_func as _flash_attn_varlen_func,
    )
except Exception:  # pragma: no cover - flash_attn optional
    _flash_attn_func = None
    try:
        from aiter import flash_attn_varlen_func as _flash_attn_varlen_func
    except Exception:
        _flash_attn_varlen_func = None


def _gather_with_stride(
    parts: list[torch.Tensor], partition_dim: int, partition_stride: int,
) -> torch.Tensor:
    """Reassemble TP-sharded partitions respecting ``partition_stride``.

    ``partition_stride=1`` is a plain concatenation.  ``partition_stride=2``
    handles SwiGLU interleaving (gate and up chunks alternate across TP ranks).
    """
    if partition_stride <= 1:
        return torch.cat(parts, dim=partition_dim)
    chunks = [p.chunk(partition_stride, dim=partition_dim) for p in parts]
    interleaved = [
        chunks[r][s]
        for s in range(partition_stride)
        for r in range(len(parts))
    ]
    return torch.cat(interleaved, dim=partition_dim)


def _shard_with_stride(
    full: torch.Tensor, partition_dim: int, partition_stride: int,
    tp_rank: int, tp_size: int,
) -> torch.Tensor:
    """Extract the TP shard for *tp_rank* from a full tensor.

    Inverse of ``_gather_with_stride``.  Mirrors Megatron's
    ``_initialize_affine_weight_cpu`` logic: split into
    ``stride * tp_size`` chunks, take every ``tp_size``-th starting
    from ``tp_rank``, then concatenate along ``partition_dim``.
    """
    if tp_size <= 1:
        return full
    per_chunk = full.shape[partition_dim] // (partition_stride * tp_size)
    chunks = full.split(per_chunk, dim=partition_dim)
    my_chunks = chunks[tp_rank::tp_size]
    return torch.cat(my_chunks, dim=partition_dim).contiguous()


class FlashSelfAttentionCore(torch.nn.Module):
    """Flash-attention drop-in for Megatron's local-spec ``DotProductAttention``.

    The local (non-TE) core attention materializes the full ``[b, np, sq, sk]``
    score matrix -> **O(L^2)** memory, which OOMs at long RL response lengths
    (resp=20480). This replacement calls ``flash_attn_func`` (O(L) memory) and is
    swapped into the GPT layer spec's ``self_attention.submodules.core_attention``
    when ``megatron_cfg.attention_backend == "flash"``.

    Assumes causal self-attention on a single unpadded sequence (LumenRL's
    per-sequence forward). GQA (num_query_groups < num_heads) is handled natively
    by flash-attn, so we skip the KV ``repeat_interleave`` the local path does.
    """

    def __init__(self, config, layer_number: int = 1, attn_mask_type=None,
                 attention_type=None, cp_comm_type=None, softmax_scale=None, **kwargs):
        super().__init__()
        if _flash_attn_func is None:
            raise ImportError(
                "megatron_cfg.attention_backend='flash' requires the flash_attn "
                "package (import failed). Install flash-attn or set attention_backend='unfused'."
            )
        self.config = config
        self.layer_number = max(1, layer_number or 1)
        head_dim = getattr(config, "kv_channels", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        if packed_seq_params is not None and _flash_attn_varlen_func is not None:
            # Megatron already squeezed [s,b=1,h,d] -> [t,h,d] for thd format.
            # Input is 3D; flash_attn_varlen_func expects [total, h, d].
            q, k, v = query, key, value
            if LUMENRL_DEBUG and self.layer_number == 1:
                logger.info(
                    "[DBG] FlashSelfAttentionCore(L%d): packed path q=%s k=%s v=%s "
                    "cu_q=%s max_q=%d",
                    self.layer_number, list(q.shape), list(k.shape), list(v.shape),
                    packed_seq_params.cu_seqlens_q.tolist()[:6],
                    packed_seq_params.max_seqlen_q,
                )
            result = _flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=packed_seq_params.cu_seqlens_q,
                cu_seqlens_k=packed_seq_params.cu_seqlens_kv,
                max_seqlen_q=packed_seq_params.max_seqlen_q,
                max_seqlen_k=packed_seq_params.max_seqlen_kv,
                causal=True,
                softmax_scale=self.softmax_scale,
            )
            out = result[0] if isinstance(result, tuple) else result
            if LUMENRL_DEBUG and self.layer_number == 1:
                logger.info("[DBG] FlashSelfAttentionCore(L1): varlen done out=%s", list(out.shape))
            # Return [t, h, d]; Megatron reshapes to [t, 1, h*d] after us.
            return out

        # Non-packed: Megatron layout [s, b, h, d] -> flash layout [b, s, h, d]
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        out = _flash_attn_func(q, k, v, causal=True, softmax_scale=self.softmax_scale)
        # [b, sq, np, hn] -> [sq, b, np*hn]
        out = out.transpose(0, 1).contiguous()
        return out.reshape(out.shape[0], out.shape[1], -1)


class MegatronEngine(MegatronBaseEngine):
    """Megatron-Core GPTModel engine (Qwen3 dense/MoE, BF16, TP/PP/EP/DP).

    The flash-attn attention core and the R3 routing-replay plumbing are what
    distinguish this from :class:`MegatronNativeEngine`, which uses the
    TransformerEngine layer spec. Everything the two share lives in
    :class:`MegatronBaseEngine`.
    """

    def __init__(self, model_config, engine_config, optimizer_config, model_name: str = ""):
        super().__init__(model_config, engine_config, optimizer_config, model_name)
        self._step = 0

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return bool(self.engine_config.get("optimizer_cpu_offload", False))

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        import megatron.core.transformer.transformer_block as tb
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.torch_norm import WrappedTorchNorm as WTN
        from megatron.core.transformer.transformer_config import TransformerConfig

        ec = self.engine_config
        tp = int(ec.get("tensor_model_parallel_size", 1))
        pp = int(ec.get("pipeline_model_parallel_size", 1))
        cp = int(ec.get("context_parallel_size", 1))
        ep = int(ec.get("expert_model_parallel_size", 1))
        seed = int(ec.get("seed", 42))
        # Long-sequence memory knobs (default off -> unchanged smoke behavior):
        #   attention_backend="flash"  -> O(L) flash attn instead of O(L^2) local core
        #   log_probs_chunk_size>0     -> memory-efficient fused/chunked token log-prob
        self._attention_backend = str(ec.get("attention_backend") or "unfused").lower()
        self._use_packed_sequences = bool(ec.get("use_packed_sequences", True))
        self._logprob_chunk_size = int(ec.get("log_probs_chunk_size") or 0)
        self._r3_enabled = bool(ec.get("moe_enable_routing_replay", False))

        etp = ec.get("expert_tensor_parallel_size", None)
        if etp is not None:
            etp = int(etp)

        import sys as _sys
        def _early_diag(tag):
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            print(
                f"[MEM_DIAG rank={self._rank()}] {tag}: "
                f"total={total/2**30:.2f} GiB, free={free/2**30:.2f} GiB, "
                f"alloc={alloc/2**30:.2f} GiB, "
                f"non_torch={(total - free - alloc)/2**30:.2f} GiB",
                file=_sys.stderr, flush=True,
            )

        _early_diag("BEFORE mpu.initialize_model_parallel")
        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=cp,
                expert_model_parallel_size=ep,
                expert_tensor_parallel_size=etp,
            )
        _early_diag("AFTER mpu.initialize_model_parallel")
        model_parallel_cuda_manual_seed(seed)

        # ---- HF config -> Qwen3 dims / TransformerConfig ----
        cfg_path = os.path.join(self.model_name, "config.json")
        with open(cfg_path) as fh:
            hf = json.load(fh)
        head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
        num_experts = int(hf.get("num_experts", 0) or ec.get("num_experts", 0))
        moe_ffn = int(hf.get("moe_intermediate_size", 0) or 0)
        shared_ffn = int(hf.get("shared_expert_intermediate_size", 0) or 0)
        shared_gate = bool(hf.get("shared_expert_gate", False))
        self._dims = Qwen3Dims(
            num_layers=hf["num_hidden_layers"], hidden=hf["hidden_size"],
            num_heads=hf["num_attention_heads"], num_kv_groups=hf["num_key_value_heads"],
            head_dim=head_dim, ffn=hf["intermediate_size"], vocab=hf["vocab_size"],
            num_experts=num_experts, moe_ffn=moe_ffn,
            shared_expert_ffn=shared_ffn, shared_expert_gate=shared_gate,
        )
        # WrappedTorchNorm asserts `not config.sequence_parallel`, but the
        # assert is overly conservative: SP communication (all-gather /
        # reduce-scatter) happens in ColumnParallelLinear / RowParallelLinear,
        # not in the norm. FusedLayerNorm only tags weight.sequence_parallel
        # for gradient sync. Monkey-patch WrappedTorchNorm to allow SP and
        # replicate that tag on the torch norm weight.
        _orig_wtn_new = WTN.__new__

        @staticmethod
        def _sp_wtn_new(cls, config, hidden_size, eps=1e-5, **kwargs):
            saved = config.sequence_parallel
            config.sequence_parallel = False
            norm = _orig_wtn_new(cls, config, hidden_size, eps, **kwargs)
            config.sequence_parallel = saved
            if saved and hasattr(norm, "weight"):
                norm.weight.sequence_parallel = True
            return norm

        WTN.__new__ = _sp_wtn_new
        tb.LayerNormImpl = WTN
        # Activation recomputation: Megatron local-spec attention (no TE flash) keeps
        # the full O(seq^2) score matrix, so long-sequence training (resp=20480) OOMs
        # without recompute. Off by default (smoke, short seq); enable via megatron_cfg.
        recompute_kwargs: dict = {}
        rc_gran = ec.get("recompute_granularity") or None
        if rc_gran:
            recompute_kwargs["recompute_granularity"] = rc_gran
            recompute_kwargs["recompute_method"] = ec.get("recompute_method") or "uniform"
            recompute_kwargs["recompute_num_layers"] = int(ec.get("recompute_num_layers") or 1)
        pp_kwargs: dict = {}
        first_pp_layers = ec.get("num_layers_in_first_pipeline_stage")
        last_pp_layers = ec.get("num_layers_in_last_pipeline_stage")
        if first_pp_layers is not None:
            pp_kwargs["num_layers_in_first_pipeline_stage"] = int(first_pp_layers)
        if last_pp_layers is not None:
            pp_kwargs["num_layers_in_last_pipeline_stage"] = int(last_pp_layers)
        if pp > 1:
            pp_kwargs["variable_seq_lengths"] = True
        moe_kwargs: dict = {}
        if num_experts > 0:
            moe_kwargs.update(
                num_moe_experts=num_experts,
                moe_ffn_hidden_size=moe_ffn if moe_ffn > 0 else hf["intermediate_size"],
                moe_router_topk=int(hf.get("num_experts_per_tok", 8)),
                moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", False)),
                # MCore defaults to allgather, which requires every dense-DP
                # rank to contribute the same token count. RL batches contain
                # variable-length responses, so EP ranks can otherwise enter
                # different collectives and deadlock. All-to-all supports the
                # variable token splits used by packed GRPO batches.
                moe_token_dispatcher_type=str(
                    ec.get("moe_token_dispatcher_type", "alltoall")
                ),
                expert_model_parallel_size=ep,
                moe_enable_routing_replay=self._r3_enabled,
            )
            if shared_ffn > 0:
                moe_kwargs["moe_shared_expert_intermediate_size"] = shared_ffn
                if shared_gate:
                    moe_kwargs["moe_shared_expert_gate"] = True
        tfcfg = TransformerConfig(
            num_layers=hf["num_hidden_layers"], hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_query_groups=hf["num_key_value_heads"], kv_channels=head_dim,
            ffn_hidden_size=hf["intermediate_size"], gated_linear_unit=True,
            activation_func=F.silu, add_bias_linear=False,
            add_qkv_bias=bool(hf.get("attention_bias", False)),
            normalization="RMSNorm", layernorm_epsilon=hf.get("rms_norm_eps", 1e-6),
            qk_layernorm=True, hidden_dropout=0.0, attention_dropout=0.0,
            attention_softmax_in_fp32=bool(
                ec.get("attention_softmax_in_fp32", False)
            ),
            bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
            sequence_parallel=bool(ec.get("sequence_parallel", False)),
            use_cpu_initialization=True,
            **recompute_kwargs,
            **moe_kwargs,
            **pp_kwargs,
        )
        spec = get_gpt_layer_local_spec(
            num_experts=num_experts if num_experts > 0 else None,
            moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", False)),
            qk_layernorm=True,
        )
        spec.submodules.input_layernorm = WTN
        spec.submodules.pre_mlp_layernorm = WTN
        spec.submodules.self_attention.submodules.q_layernorm = WTN
        spec.submodules.self_attention.submodules.k_layernorm = WTN
        if self._attention_backend == "flash":
            if _flash_attn_func is None:
                logger.warning(
                    "MegatronEngine[%d]: attention_backend='flash' requested but "
                    "flash_attn not installed — falling back to 'unfused' "
                    "(O(L^2) memory, may OOM on long sequences)",
                    self._rank(),
                )
                self._attention_backend = "unfused"
            else:
                spec.submodules.self_attention.submodules.core_attention = FlashSelfAttentionCore
                logger.info("MegatronEngine[%d]: using flash-attention core (O(L) memory)", self._rank())

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._pp_rank = pp_rank
        self._pp_size = pp_size
        self._tp_rank = mpu.get_tensor_model_parallel_rank()
        self._tp_size = tp

        # Compute per-PP-rank layer counts for the bridge.
        self._layers_per_pp_rank: list[int] | None = None
        if pp_size > 1:
            total = hf["num_hidden_layers"]
            first = int(first_pp_layers) if first_pp_layers is not None else None
            last = int(last_pp_layers) if last_pp_layers is not None else None
            if first is not None or last is not None:
                remaining = total - (first or 0) - (last or 0)
                mid_stages = pp_size - (1 if first is not None else 0) - (1 if last is not None else 0)
                per_mid = remaining // mid_stages if mid_stages > 0 else 0
                lpp: list[int] = []
                for s in range(pp_size):
                    if s == 0 and first is not None:
                        lpp.append(first)
                    elif s == pp_size - 1 and last is not None:
                        lpp.append(last)
                    else:
                        lpp.append(per_mid)
                self._layers_per_pp_rank = lpp
            else:
                per_stage = total // pp_size
                self._layers_per_pp_rank = [per_stage] * pp_size
        model = GPTModel(
            config=tfcfg, transformer_layer_spec=spec, vocab_size=hf["vocab_size"],
            max_sequence_length=hf.get("max_position_embeddings", 32768),
            pre_process=(pp_rank == 0),
            post_process=(pp_rank == pp_size - 1),
            parallel_output=False,
            position_embedding_type="rope",
            rotary_base=hf.get("rope_theta", 1000000.0),
            share_embeddings_and_output_weights=bool(hf.get("tie_word_embeddings", False)),
        )

        # ---- detect expert format ----
        self._use_grouped_mlp = any(
            ".experts.weight1" in n for n, _ in model.named_parameters()
        )

        # ---- load HF weights ----
        ep_rank = mpu.get_expert_model_parallel_rank() if num_experts > 0 else 0
        ep_size = mpu.get_expert_model_parallel_world_size() if num_experts > 0 else 1
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        logger.info(
            "MegatronEngine[%d]: loading HF weights from %s (ep_rank=%d/%d, experts=%d)",
            self._rank(), self.model_name, ep_rank, ep_size, num_experts,
        )
        hf_state = load_hf_safetensors(self.model_name)
        meg_state = hf_to_megatron(
            hf_state, self._dims,
            ep_rank=ep_rank, ep_size=ep_size,
            pp_rank=pp_rank, pp_size=pp_size,
            layers_per_pp_rank=self._layers_per_pp_rank,
            use_grouped_mlp=self._use_grouped_mlp,
        )
        del hf_state
        # TP shard: hf_to_megatron returns full tensors; the GPTModel's
        # ColumnParallelLinear / RowParallelLinear only hold 1/tp of each
        # weight.  Shard using the partition metadata set by Megatron.
        # Expert weights use ETP (expert_tensor_parallel), not TP.
        tp_rank = mpu.get_tensor_model_parallel_rank()
        etp_val = etp if etp is not None else 1
        etp_rank = (mpu.get_expert_tensor_parallel_rank()
                    if etp_val > 1 else 0)
        if tp > 1 or etp_val > 1:
            for name, param in model.named_parameters():
                if name not in meg_state:
                    continue
                if not getattr(param, "tensor_model_parallel", False):
                    continue
                is_expert = (".experts." in name
                             and ".shared_experts." not in name)
                shard_size = etp_val if is_expert else tp
                shard_rank = etp_rank if is_expert else tp_rank
                if shard_size <= 1:
                    continue
                pdim = getattr(param, "partition_dim", 0)
                pstride = getattr(param, "partition_stride", 1)
                meg_state[name] = _shard_with_stride(
                    meg_state[name], pdim, pstride, shard_rank, shard_size,
                )
        incompat = model.load_state_dict(meg_state, strict=False)
        real_missing = [k for k in incompat.missing_keys if "_extra_state" not in k]
        if real_missing:
            raise RuntimeError(f"Megatron load missing keys: {real_missing[:6]} ...")
        if incompat.unexpected_keys:
            logger.warning("Megatron load unexpected keys: %s", incompat.unexpected_keys[:6])
        del meg_state
        import gc
        import sys
        gc.collect()

        def _gpu_diag(tag: str) -> None:
            if not torch.cuda.is_available():
                return
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            resv = torch.cuda.memory_reserved()
            msg = (
                f"[GPU_DIAG rank={self._rank()}] {tag}: "
                f"total={total / 2**30:.2f} GiB, free={free / 2**30:.2f} GiB, "
                f"alloc={alloc / 2**30:.2f} GiB, reserved={resv / 2**30:.2f} GiB"
            )
            print(msg, file=sys.stderr, flush=True)
            logger.info(msg)

        _gpu_diag("BEFORE model.cuda()")
        self.module = model.cuda().bfloat16()
        self._tfcfg = tfcfg
        _gpu_diag("AFTER model.cuda()")
        n_params = sum(p.numel() for p in self.module.parameters())
        n_grad = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        print(
            f"[GPU_DIAG rank={self._rank()}] params={n_params:,} "
            f"({n_params * 2 / 2**30:.2f} GiB bf16), requires_grad={n_grad:,}",
            file=sys.stderr, flush=True,
        )

        # ---- Megatron DistributedDataParallel: shards optimizer state across DP ----
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        oc = self.optimizer_config
        self._clip = float(oc.get("clip_grad", 1.0))
        ddp_cfg = DistributedDataParallelConfig(
            grad_reduce_in_fp32=bool(ec.get("grad_reduce_in_fp32", False)),
            overlap_grad_reduce=bool(ec.get("overlap_grad_reduce", False)),
            use_distributed_optimizer=bool(ec.get("use_distributed_optimizer", True)),
            average_in_collective=True,
            bucket_size=ec.get("bucket_size", None),
        )
        _gpu_diag("BEFORE DDP init")
        self._ddp = DDP(config=tfcfg, ddp_config=ddp_cfg, module=self.module)
        _gpu_diag("AFTER DDP init")

        cpu_offload = bool(ec.get("optimizer_cpu_offload", False))
        offload_frac = float(ec.get("optimizer_offload_fraction", 1.0))
        opt_cfg = OptimizerConfig(
            optimizer="adam", lr=float(oc.get("lr", 1e-6)),
            weight_decay=float(oc.get("weight_decay", 0.1)),
            adam_beta1=float(oc.get("adam_beta1", 0.9)),
            adam_beta2=float(oc.get("adam_beta2", 0.95)),
            adam_eps=float(oc.get("adam_eps", 1e-8)),
            clip_grad=self._clip, bf16=True, fp16=False,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=bool(ec.get("use_distributed_optimizer", True)),
            optimizer_cpu_offload=cpu_offload,
            optimizer_offload_fraction=offload_frac,
        )
        _gpu_diag("BEFORE optimizer init")
        self.optimizer = get_megatron_optimizer(opt_cfg, model_chunks=[self._ddp])

        warmup = int(oc.get("lr_warmup_steps", 10))
        base_lr = float(oc.get("lr", 1e-6))
        wd = float(oc.get("weight_decay", 0.1))
        self.lr_scheduler = OptimizerParamScheduler(
            self.optimizer, init_lr=0.0, max_lr=base_lr, min_lr=base_lr,
            lr_warmup_steps=warmup, lr_decay_steps=max(warmup + 1, 1000),
            lr_decay_style="constant", start_wd=wd, end_wd=wd,
            wd_incr_steps=0, wd_incr_style="constant",
        )
        if self._rank() == 0:
            n = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
            logger.info(
                "MegatronEngine: model+distributed-optimizer ready, %d params, "
                "dp_size=%d, MILES-R3=%s",
                n, self.get_data_parallel_size(), self._r3_enabled,
            )

    # ------------------------------------------------------------------




    def is_mp_src_rank_with_outputs(self) -> bool:
        return self._tp_rank == 0 and self._pp_rank == self._pp_size - 1


    # ------------------------------------------------------------------
    def _forward_logits(self, ids: torch.Tensor, model=None) -> torch.Tensor:
        """Run the model on a single unpadded sequence -> logits [L, V] (float).

        ``model`` defaults to the unwrapped GPTModel (eval); pass ``self._ddp``
        during training so DDP grad hooks fire and grads land in the buffer.

        Sequence-parallel reduce_scatter requires the sequence length to be
        divisible by TP.  We right-pad to the next multiple and trim after.
        """
        m = model if model is not None else self.module
        L = ids.numel()
        tp = self._tp_size
        pad_len = (tp - L % tp) % tp
        if pad_len:
            ids = torch.cat([ids, ids.new_zeros(pad_len)])
        L_padded = ids.numel()
        inp = ids.view(1, L_padded)
        pos = torch.arange(L_padded, device=ids.device).view(1, L_padded)
        if LUMENRL_DEBUG:
            logger.info("[DBG] _forward_logits: L=%d pad=%d L_padded=%d tp=%d", L, pad_len, L_padded, tp)
        out = m(input_ids=inp, position_ids=pos, attention_mask=None)
        logits = out.logits if hasattr(out, "logits") else out
        if LUMENRL_DEBUG:
            logger.info("[DBG] _forward_logits: raw_out shape=%s → logits[:%d]", list(logits.shape), L)
        return logits.view(L_padded, -1)[:L].float()

    def _forward_logits_packed(self, packed, model=None) -> torch.Tensor:
        """Run a packed batch through GPTModel with varlen attention.

        Returns logits ``[total_real_tokens, V]`` (float32), trimmed to
        real tokens (excluding TP-alignment padding).

        Requires ``attention_backend='flash'`` (FlashSelfAttentionCore with
        varlen support).
        """
        from megatron.core.packed_seq_params import PackedSeqParams

        m = model if model is not None else self.module
        total_real = int(packed.cu_seqlens[-1].item())
        total_padded = packed.input_ids.shape[-1]
        pad_len = total_padded - total_real

        cu = packed.cu_seqlens
        if pad_len > 0:
            cu = torch.cat([cu, torch.tensor(
                [total_padded], dtype=torch.int32, device=cu.device,
            )])
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=packed.max_seqlen,
            max_seqlen_kv=packed.max_seqlen,
        )
        if LUMENRL_DEBUG:
            logger.info(
                "[DBG] _forward_logits_packed: total_padded=%d real=%d pad=%d seqs=%d max_sl=%d tp=%d "
                "ids_shape=%s pos_shape=%s cu=%s",
                total_padded, total_real, pad_len,
                packed.cu_seqlens.shape[0] - 1, packed.max_seqlen, self._tp_size,
                list(packed.input_ids.shape), list(packed.position_ids.shape),
                cu.tolist(),
            )
        out = m(
            input_ids=packed.input_ids,
            position_ids=packed.position_ids,
            attention_mask=None,
            packed_seq_params=psp,
        )
        if LUMENRL_DEBUG:
            logger.info("[DBG] _forward_logits_packed: forward done, out type=%s", type(out).__name__)
        logits = out.logits if hasattr(out, "logits") else out
        return logits.view(-1, logits.shape[-1])[:total_real].float()


    # ---- MILES R3: rollout top-k expert-id replay ---------------------
    def _r3_routes(self, batch: DataProto) -> torch.Tensor | list[Any] | None:
        if not self._r3_enabled:
            return None
        routes = batch.ragged.get("rollout_routing")
        if routes is not None:
            if len(routes) != batch.batch_size:
                raise ValueError(
                    "rollout_routing row count mismatch: "
                    f"got {len(routes)}, expected {batch.batch_size}"
                )
            if any(row is None for row in routes):
                raise RuntimeError(
                    "moe.r3.enabled=true but rollout_routing contains missing rows."
                )
            return routes
        routes = batch.tensors.get("rollout_routed_experts")
        if routes is None:
            raise RuntimeError(
                "moe.r3.enabled=true but neither ragged rollout_routing nor "
                "rollout_routed_experts is present. "
                "R3 must not silently fall back to training-time routing."
            )
        if routes.ndim != 4:
            raise ValueError(
                "rollout_routed_experts must have shape [batch, seq_len-1, "
                f"num_layers, top_k], got {tuple(routes.shape)}"
            )
        return routes

    def _r3_set_packed_routes(
        self,
        routes: torch.Tensor | list[Any],
        attention_mask: torch.Tensor,
        packed,
    ) -> None:
        """Load one packed chunk into Megatron's native RouterReplay buffers.

        vLLM captures routes for ``sequence_length - 1`` input positions. The
        final sequence token and TP-alignment padding do not contribute to the
        policy loss, but Megatron still routes them, so deterministic valid
        expert ids are appended for those positions.
        """
        from megatron.core.transformer.moe.router_replay import (
            RouterReplay,
            RouterReplayAction,
        )

        instances = RouterReplay.global_router_replay_instances
        if not instances:
            raise RuntimeError(
                "MILES R3 is enabled but Megatron created no RouterReplay "
                "instances; check moe_enable_routing_replay and the Megatron fork."
            )

        dense = isinstance(routes, torch.Tensor)
        if dense:
            num_layers = int(routes.shape[2])
            topk = int(routes.shape[3])
            route_dtype = routes.dtype
            route_device = routes.device
        else:
            if not routes:
                raise ValueError("R3 received an empty ragged routing chunk.")
            sample = torch.as_tensor(routes[0])
            if sample.ndim != 3:
                raise ValueError(
                    "ragged rollout_routing rows must have shape "
                    f"[seq_len-1, num_layers, top_k], got {tuple(sample.shape)}"
                )
            num_layers = int(sample.shape[1])
            topk = int(sample.shape[2])
            route_dtype = sample.dtype
            route_device = sample.device
        filler = torch.arange(topk, dtype=route_dtype, device=route_device)
        rows: list[torch.Tensor] = []
        for row in range(attention_mask.shape[0]):
            start, length = self._real_block(attention_mask[row])
            if length <= 0:
                continue
            if dense:
                real = routes[row, start:start + max(0, length - 1)]
            else:
                raw = routes[row]
                if isinstance(raw, torch.Tensor):
                    real = raw.detach().to(
                        device=route_device, dtype=route_dtype,
                    ).clone()
                else:
                    real = torch.tensor(
                        raw, device=route_device, dtype=route_dtype,
                    )
                if real.ndim != 3 or tuple(real.shape[1:]) != (num_layers, topk):
                    raise ValueError(
                        f"R3 routing for row {row} has shape {tuple(real.shape)}, "
                        f"expected [seq_len-1, {num_layers}, {topk}]"
                    )
            if real.shape[0] != max(0, length - 1):
                raise ValueError(
                    f"R3 route length mismatch for row {row}: got {real.shape[0]}, "
                    f"expected {length - 1}"
                )
            rows.append(torch.cat(
                [real, filler.view(1, 1, topk).expand(1, num_layers, topk)],
                dim=0,
            ))

        if not rows:
            raise ValueError("R3 received a packed chunk with no real tokens.")
        replay = torch.cat(rows, dim=0)
        total_padded = int(packed.input_ids.shape[-1])
        if replay.shape[0] < total_padded:
            pad = filler.view(1, 1, topk).expand(
                total_padded - replay.shape[0], num_layers, topk,
            )
            replay = torch.cat([replay, pad], dim=0)
        if replay.shape[0] != total_padded:
            raise ValueError(
                f"R3 packed token mismatch: routes={replay.shape[0]}, "
                f"packed={total_padded}"
            )

        # Sequence parallel scatters the token dimension into contiguous TP
        # shards before TopKRouter. Replay must use the identical local shard.
        if bool(getattr(self._tfcfg, "sequence_parallel", False)) and self._tp_size > 1:
            if replay.shape[0] % self._tp_size:
                raise ValueError(
                    f"R3 token count {replay.shape[0]} is not divisible by TP={self._tp_size}"
                )
            shard = replay.shape[0] // self._tp_size
            replay = replay[self._tp_rank * shard:(self._tp_rank + 1) * shard]

        if replay.shape[1] != len(instances):
            raise ValueError(
                f"R3 rollout has {replay.shape[1]} MoE layers but this Megatron "
                f"stage has {len(instances)} RouterReplay instances."
            )
        RouterReplay.clear_global_indices()
        RouterReplay.set_replay_data(
            # Keep controller/Ray transport compact (int16), but torch.gather
            # requires integer index tensors in int32/int64 on ROCm.
            [
                replay[:, layer, :].to(dtype=torch.int64).contiguous()
                for layer in range(replay.shape[1])
            ]
        )
        RouterReplay.set_global_router_replay_action(
            RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def _r3_set_backward() -> None:
        from megatron.core.transformer.moe.router_replay import (
            RouterReplay,
            RouterReplayAction,
        )
        RouterReplay.set_global_router_replay_action(
            RouterReplayAction.REPLAY_BACKWARD
        )

    @staticmethod
    def _r3_clear() -> None:
        from megatron.core.transformer.moe.router_replay import RouterReplay
        RouterReplay.clear_global_router_replay_action()
        RouterReplay.clear_global_indices()

    @staticmethod
    def _r3_metrics(
        routes: torch.Tensor | list[Any],
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        if isinstance(routes, torch.Tensor):
            num_layers = int(routes.shape[2])
            valid_count = int((routes >= 0).all(dim=-1).sum().item())
        else:
            sample = torch.as_tensor(routes[0])
            num_layers = int(sample.shape[1])
            valid_count = sum(
                int((torch.as_tensor(row) >= 0).all(dim=-1).sum().item())
                for row in routes
            )
        expected_positions = int(
            (attention_mask.sum(dim=1).sub(1).clamp_min(0).sum() * num_layers).item()
        )
        return {
            "moe/r3_enabled": 1.0,
            "moe/r3_route_coverage": valid_count / max(1, expected_positions),
            "moe/r3_route_tokens": float(valid_count // max(1, num_layers)),
        }

    # ---- memory-efficient log-prob helpers (see FlashSelfAttentionCore/#2) ----


    # ---- engine-level compute_log_probs (actor delegates here) ----
    def engine_compute_log_probs(self, batch: DataProto) -> DataProto:
        seqs = batch["input_ids"]
        B, S = seqs.shape
        am = batch.tensors.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        want_ent = bool(batch.meta.get("calculate_entropy", False))
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)
        print(f"[TRACE] engine_compute_log_probs: B={B} S={S} pp={self._pp_size} tp={self._tp_size} "
              f"attn_backend={self._attention_backend} can_varlen={_flash_attn_varlen_func is not None}", flush=True)
        if LUMENRL_DEBUG:
            logger.info("[DBG] compute_log_probs: B=%d S=%d want_ent=%s temp=%.2f pp=%d tp=%d",
                        B, S, want_ent, temperature, self._pp_size, self._tp_size)

        if self._pp_size > 1:
            return self._engine_compute_log_probs_pp(
                seqs, am, S, want_ent, temperature, batch,
            )

        can_pack = (
            self._use_packed_sequences
            and
            self._attention_backend == "flash"
            and _flash_attn_varlen_func is not None
        )

        if can_pack:
            return self._engine_compute_log_probs_packed(
                seqs, am, B, S, want_ent, temperature, batch,
            )
        if self._r3_enabled:
            raise RuntimeError(
                "MILES R3 currently requires Megatron packed-sequence training "
                "with flash attention; refusing an inconsistent row-wise forward."
            )

        # Fallback: row-by-row forward (unfused attention or no varlen)
        if LUMENRL_DEBUG:
            logger.info("[DBG] compute_log_probs: using row-by-row fallback (attention_backend=%s)",
                        self._attention_backend)

        n_iters = B
        if dist.is_initialized() and dist.get_world_size() > 1:
            cnt = torch.tensor([B], device="cuda")
            dist.all_reduce(cnt, op=dist.ReduceOp.MAX)
            n_iters = int(cnt.item())

        lp_out = torch.zeros(B, S - 1, dtype=torch.float32)
        ent_out = torch.zeros(B, S - 1, dtype=torch.float32) if want_ent else None
        self.module.eval()
        with torch.no_grad():
            for r in range(n_iters):
                if r >= B:
                    ids = seqs[B - 1][am[B - 1].bool()].to("cuda")
                    self._forward_logits(ids)
                    continue
                start, L = self._real_block(am[r])
                if L < 2:
                    continue
                ids = seqs[r, start:start + L].to("cuda")
                logits = self._forward_logits(ids) / temperature
                tok_lp, ent = self._logprob_entropy_nograd(logits[:-1], ids[1:], want_ent)
                lp_out[r, start:start + L - 1] = tok_lp.cpu()
                if want_ent and ent is not None:
                    ent_out[r, start:start + L - 1] = ent.cpu()
        tensors = {"log_probs": lp_out, "input_ids": batch["input_ids"]}
        if want_ent:
            tensors["entropy"] = ent_out
        return DataProto(tensors=tensors, meta=dict(batch.meta))

    def _equalize_chunk_count(self, chunks: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Ensure all ranks in the EP group execute the same number of forward
        passes, preventing MoE allgather deadlock.

        Ranks with fewer chunks append dummy repeats of their last chunk. The
        caller must discard results from indices beyond the original count.
        """
        if not dist.is_initialized() or dist.get_world_size() < 2:
            return chunks
        real_count = len(chunks)
        count_t = torch.tensor([real_count], device="cuda")
        print(f"[TRACE] _equalize_chunk_count: rank={dist.get_rank()} local_count={real_count} calling allreduce", flush=True)
        dist.all_reduce(count_t, op=dist.ReduceOp.MAX)
        global_max = int(count_t.item())
        print(f"[TRACE] _equalize_chunk_count: rank={dist.get_rank()} global_max={global_max}", flush=True)
        while len(chunks) < global_max:
            chunks.append(chunks[-1])
        if LUMENRL_DEBUG and global_max > real_count:
            logger.info("[DBG] _equalize_chunk_count: padded %d → %d chunks",
                        real_count, global_max)
        return chunks

    def _engine_compute_log_probs_packed(
        self, seqs, am, B, S, want_ent, temperature, batch,
    ) -> DataProto:
        """Packed (varlen) compute_log_probs — all sequences in 1-2 forwards."""
        from lumenrl.engine.training.packing import (
            pack_sequences, packed_token_log_probs, packed_token_entropy,
            unpack_log_probs,
        )

        max_tok = int(batch.meta.get("max_token_len_per_gpu", 8192) or 8192)
        seq_lens = am.sum(dim=1).long()
        r3_routes = self._r3_routes(batch)
        print(f"[TRACE] _engine_compute_log_probs_packed: B={B} S={S} max_tok={max_tok} "
              f"seq_lens={seq_lens.tolist()}", flush=True)

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < B:
            tok_count = 0
            end = start
            while end < B:
                sl = int(seq_lens[end].item())
                if tok_count + sl > max_tok and end > start:
                    break
                tok_count += sl
                end += 1
            chunks.append((start, end))
            start = end

        real_chunk_count = len(chunks)
        print(f"[TRACE] _engine_compute_log_probs_packed: {real_chunk_count} chunks, calling equalize", flush=True)
        chunks = self._equalize_chunk_count(chunks)
        print(f"[TRACE] _engine_compute_log_probs_packed: equalize done, {len(chunks)} chunks", flush=True)

        if LUMENRL_DEBUG:
            logger.info("[DBG] compute_log_probs_packed: B=%d real_chunks=%d total_chunks=%d max_tok=%d",
                        B, real_chunk_count, len(chunks), max_tok)

        lp_out = torch.zeros(B, S - 1, dtype=torch.float32)
        ent_out = torch.zeros(B, S - 1, dtype=torch.float32) if want_ent else None
        self.module.eval()

        with torch.no_grad():
            for ci, (cs, ce) in enumerate(chunks):
                print(f"[TRACE] packed chunk {ci}/{len(chunks)}: rows [{cs}:{ce}]", flush=True)
                ids_chunk = seqs[cs:ce].to("cuda")
                mask_chunk = am[cs:ce].to("cuda")
                packed = pack_sequences(ids_chunk, mask_chunk, tp_align=self._tp_size)
                print(f"[TRACE] packed chunk {ci}: packed done, total_tokens={packed.input_ids.shape[-1]} "
                      f"cu_seqlens={packed.cu_seqlens.tolist()}", flush=True)
                if r3_routes is not None:
                    self._r3_set_packed_routes(
                        r3_routes[cs:ce], am[cs:ce], packed,
                    )
                try:
                    logits = self._forward_logits_packed(packed)
                finally:
                    if r3_routes is not None:
                        self._r3_clear()
                print(f"[TRACE] packed chunk {ci}: forward done, logits={list(logits.shape)}", flush=True)

                if ci >= real_chunk_count:
                    del logits, packed, ids_chunk, mask_chunk
                    continue

                flat_lp = packed_token_log_probs(
                    logits, packed.input_ids.squeeze(0), packed.cu_seqlens,
                    temperature=temperature,
                )
                chunk_lp = unpack_log_probs(
                    flat_lp, packed.cu_seqlens, packed.seq_lens, S,
                )
                lp_out[cs:ce] = chunk_lp.cpu()

                if want_ent:
                    flat_ent = packed_token_entropy(
                        logits.detach(), packed.cu_seqlens,
                        temperature=temperature, upcast=True,
                    )
                    chunk_ent = unpack_log_probs(
                        flat_ent, packed.cu_seqlens, packed.seq_lens, S,
                    )
                    ent_out[cs:ce] = chunk_ent.cpu()

                del logits, flat_lp, packed, ids_chunk, mask_chunk

        # Ensure packed forward work is complete before policy collectives.
        # Keep allocator segments cached: empty_cache() repeatedly tears down
        # ROCr allocations and can exhaust HSA queue/event resources in long
        # runs. Expandable segments handle the varying packed shapes.
        torch.cuda.synchronize()
        tensors = {"log_probs": lp_out, "input_ids": batch["input_ids"]}
        if want_ent:
            tensors["entropy"] = ent_out
        return DataProto(tensors=tensors, meta=dict(batch.meta))

    def _engine_compute_log_probs_pp(
        self, seqs, am, S, want_ent, temperature, batch,
    ) -> DataProto:
        """PP>1 log-prob computation via Megatron's pipeline schedule."""
        from functools import partial as _partial

        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        B = seqs.shape[0]
        is_last_pp = mpu.is_pipeline_last_stage()

        micro_batches = []
        for r in range(B):
            start, L = self._real_block(am[r])
            if L < 2:
                micro_batches.append((r, start, 0, None))
                continue
            ids = seqs[r, start:start + L].to("cuda")
            micro_batches.append((r, start, L, ids))

        results = []

        def _logprob_loss_func(row_idx, start, L, ids_local, temperature_local, want_ent_local, output_tensor):
            if L < 2:
                results.append((row_idx, start, L, None, None))
                return torch.tensor(0.0, device="cuda"), {}
            logits = output_tensor.view(L, -1).float() / temperature_local
            tok_lp, ent = self._logprob_entropy_nograd(logits[:-1], ids_local[1:], want_ent_local)
            results.append((row_idx, start, L, tok_lp.cpu(), ent.cpu() if ent is not None else None))
            return torch.tensor(0.0, device="cuda"), {}

        mb_iter = iter(micro_batches)

        def _forward_step(data_iterator, model):
            r, start, L, ids = next(data_iterator)
            if L < 2:
                dummy = torch.zeros(1, 1, self._dims.hidden, device="cuda", dtype=torch.bfloat16)
                return dummy, _partial(_logprob_loss_func, r, start, L, None, temperature, want_ent)
            inp = ids.view(1, L)
            pos = torch.arange(L, device=ids.device).view(1, L)
            out = model(input_ids=inp, position_ids=pos, attention_mask=None)
            return out, _partial(_logprob_loss_func, r, start, L, ids, temperature, want_ent)

        self.module.eval()
        forward_backward_func = get_forward_backward_func()
        config = self._tfcfg
        saved_timers = config.timers
        config.timers = None
        with torch.no_grad():
            forward_backward_func(
                forward_step_func=_forward_step,
                data_iterator=mb_iter,
                model=[self.module],
                num_microbatches=B,
                seq_length=S,
                micro_batch_size=1,
                forward_only=True,
                collect_non_loss_data=False,
            )
        config.timers = saved_timers

        lp_out = torch.zeros(B, S - 1, dtype=torch.float32)
        ent_out = torch.zeros(B, S - 1, dtype=torch.float32) if want_ent else None
        if is_last_pp:
            for row_idx, start, L, tok_lp, ent in results:
                if L < 2:
                    continue
                lp_out[row_idx, start:start + L - 1] = tok_lp
                if want_ent and ent is not None:
                    ent_out[row_idx, start:start + L - 1] = ent

        tensors = {"log_probs": lp_out, "input_ids": batch["input_ids"]}
        if want_ent:
            tensors["entropy"] = ent_out
        return DataProto(tensors=tensors, meta=dict(batch.meta))

    # ---- engine-level DAPO/GRPO update (actor delegates here) ----
    def engine_update_policy(self, batch: DataProto) -> dict[str, float]:
        if batch.batch_size == 0:
            return {"loss": 0.0, "lr": self._cur_lr(), "grad_norm": 0.0}
        meta = dict(batch.meta)
        if meta.get("task_type") == "sft":
            return self._engine_update_sft(batch)
        algo_name = str(meta.get("algorithm", "dapo")).lower()
        temperature = float(meta.get("temperature", 1.0) or 1.0)
        bnt = meta.get("batch_num_tokens")
        dp = int(meta.get("dp_size", self.get_data_parallel_size()) or 1)
        algo_cfg_full = meta.get("algo_config", {}) or {}
        _sub = algo_cfg_full.get(algo_name)
        _sub = _sub if isinstance(_sub, dict) else {}

        def _cfg(key, default):
            v = _sub.get(key, algo_cfg_full.get(key, default))
            return default if v is None else v

        t = batch.tensors
        seqs = t["input_ids"]
        am = t.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        B, S = seqs.shape

        if LUMENRL_DEBUG:
            logger.info("[DBG] engine_update_policy: B=%d S=%d algo=%s temp=%.2f pp=%d tp=%d",
                        B, S, algo_name, temperature, self._pp_size, self._tp_size)

        if self._pp_size > 1:
            return self._engine_update_policy_pp(
                batch, seqs, am, B, S, algo_name, temperature,
                bnt, dp, _cfg, t,
            )

        can_pack = (
            self._use_packed_sequences
            and
            self._attention_backend == "flash"
            and _flash_attn_varlen_func is not None
        )

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        if can_pack:
            metrics = self._engine_update_policy_packed(
                batch, seqs, am, B, S, algo_name, temperature, bnt, dp, _cfg, t, meta,
            )
        else:
            if self._r3_enabled:
                raise RuntimeError(
                    "MILES R3 currently requires Megatron packed-sequence "
                    "training with flash attention."
                )
            if LUMENRL_DEBUG:
                logger.info("[DBG] engine_update_policy: using row-by-row fallback (attention_backend=%s)",
                            self._attention_backend)
            metrics = self._engine_update_policy_rowwise(
                seqs, am, B, S, algo_name, temperature, bnt, dp, _cfg, t,
            )

        grad_norm = self._optimizer_step()
        lr = self._sched_step()
        metrics["lr"] = lr
        metrics["grad_norm"] = grad_norm
        if LUMENRL_DEBUG:
            logger.info("[DBG] engine_update_policy done: loss=%.6f grad_norm=%.4f lr=%.2e",
                        metrics.get("loss", 0.0), grad_norm, lr)
        return metrics

    def _engine_update_policy_rowwise(
        self, seqs, am, B, S, algo_name, temperature, bnt, dp, _cfg, t,
    ) -> dict[str, float]:
        """Row-by-row forward+backward (fallback for unfused attention)."""
        n_iters = B
        if dist.is_initialized() and dist.get_world_size() > 1:
            cnt = torch.tensor([B], device="cuda")
            dist.all_reduce(cnt, op=dist.ReduceOp.MAX)
            n_iters = int(cnt.item())

        loss_accum = 0.0
        ppo_kl_sum = 0.0
        ppo_kl_tok = 0.0
        rc_kl_sum = 0.0
        rc_kl_tok = 0.0
        n_rows = 0

        for r in range(n_iters):
            if r >= B:
                ids = seqs[B - 1][am[B - 1].bool()].to("cuda")
                with torch.no_grad():
                    self._forward_logits(ids)
                continue
            start, L = self._real_block(am[r])
            if L < 2:
                continue
            ids = seqs[r, start:start + L].to("cuda")
            logits = self._forward_logits(ids, model=self._ddp) / temperature
            token_lp = self._token_logprob_train(logits[:-1], ids[1:]).view(1, -1)
            Lm = token_lp.shape[-1]
            dev = token_lp.device

            def _col(name, shift, _r=r, _start=start):
                x = t.get(name)
                if x is None:
                    return None
                x = x[_r].to(dev)
                s0 = _start + (1 if shift else 0)
                return x[s0:].reshape(1, -1)

            old_lp = _col("old_log_probs", shift=False)
            if old_lp is None:
                continue
            resp_mask = _col("response_mask", shift=True)
            adv_t = t.get("advantages")
            if adv_t is None:
                continue
            if adv_t.dim() == 1:
                adv = adv_t[r].to(dev).view(1, 1).expand(1, Lm).float()
            else:
                adv = adv_t[r].to(dev)[start + 1:].reshape(1, -1).float()
            ris = _col("rollout_is_weights", shift=False)
            ref_lp0 = _col("ref_log_probs", shift=False)
            rlp0 = _col("rollout_log_probs", shift=False)

            cand = [token_lp, old_lp, adv]
            for v in (resp_mask, ris, ref_lp0, rlp0):
                if v is not None:
                    cand.append(v)
            Le = min(v.shape[-1] for v in cand)
            token_lp = token_lp[..., :Le]
            old_lp = old_lp[..., :Le]
            adv = adv[..., :Le]
            mask = resp_mask[..., :Le].float() if resp_mask is not None else None
            ris = ris[..., :Le] if ris is not None else None
            ref_lp = ref_lp0[..., :Le] if ref_lp0 is not None else None
            rlp = rlp0[..., :Le] if rlp0 is not None else None

            if algo_name == AlgorithmName.DAPO.value:
                loss = asymmetric_clip_loss(
                    token_lp, old_lp, adv,
                    float(_cfg("clip_ratio_low", 0.2)), float(_cfg("clip_ratio_high", 0.28)),
                    mask=mask, clip_ratio_c=float(_cfg("clip_ratio_c", 0.0)),
                    batch_num_tokens=bnt, dp_size=dp, rollout_is_weights=ris,
                )
            else:
                loss = policy_gradient_loss(
                    token_lp, old_lp, adv, float(_cfg("clip_ratio", 0.2)), mask=mask,
                )
            kl_c = float(_cfg("kl_coeff", 0.0))
            if kl_c > 0.0 and ref_lp is not None:
                loss = loss + kl_c * kl_penalty(token_lp, ref_lp, mask=mask)

            loss.backward()
            loss_accum += float(loss.detach())
            n_rows += 1
            if mask is not None:
                with torch.no_grad():
                    tok = float(mask.sum())
                    ppo_kl_sum += float(((old_lp - token_lp) * mask).sum())
                    ppo_kl_tok += tok
                    if rlp is not None:
                        rc_kl_sum += float(((rlp - token_lp) * mask).sum())
                        rc_kl_tok += tok
            del logits, token_lp, loss

        torch.cuda.empty_cache()
        metrics: dict[str, float] = {
            "loss": loss_accum / max(1, n_rows),
        }
        if ppo_kl_tok > 0:
            metrics["ppo_kl_sum"] = ppo_kl_sum
            metrics["ppo_kl_tok"] = ppo_kl_tok
        if rc_kl_tok > 0:
            metrics["rollout_corr_kl_sum"] = rc_kl_sum
            metrics["rollout_corr_kl_tok"] = rc_kl_tok
        return metrics

    def _engine_update_policy_packed(
        self, batch, seqs, am, B, S, algo_name, temperature, bnt, dp, _cfg, t, meta,
    ) -> dict[str, float]:
        """Packed (varlen) forward + per-row loss + backward."""
        from lumenrl.engine.training.packing import (
            pack_sequences, packed_token_log_probs, unpack_log_probs,
        )

        max_tok = int(meta.get("max_token_len_per_gpu", 8192) or 8192)
        seq_lens = am.sum(dim=1).long()
        r3_routes = self._r3_routes(batch)

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < B:
            tok_count = 0
            end = start
            while end < B:
                sl = int(seq_lens[end].item())
                if tok_count + sl > max_tok and end > start:
                    break
                tok_count += sl
                end += 1
            chunks.append((start, end))
            start = end

        real_chunk_count = len(chunks)
        chunks = self._equalize_chunk_count(chunks)

        if LUMENRL_DEBUG:
            logger.info("[DBG] update_policy_packed: B=%d real_chunks=%d total_chunks=%d max_tok=%d",
                        B, real_chunk_count, len(chunks), max_tok)

        loss_accum = 0.0
        ppo_kl_sum = 0.0
        ppo_kl_tok = 0.0
        pg_clip_sum = 0.0
        pg_clip_lower_sum = 0.0
        rc_kl_sum = 0.0
        rc_kl_tok = 0.0
        n_rows = 0

        for ci, (cs, ce) in enumerate(chunks):
            is_dummy = ci >= real_chunk_count
            chunk_B = ce - cs
            ids_chunk = seqs[cs:ce].to("cuda")
            mask_chunk = am[cs:ce].to("cuda")
            packed = pack_sequences(ids_chunk, mask_chunk, tp_align=self._tp_size)
            if r3_routes is not None:
                self._r3_set_packed_routes(
                    r3_routes[cs:ce], am[cs:ce], packed,
                )

            if is_dummy:
                # A padded chunk must execute the same forward *and backward*
                # collective schedule as a real chunk. EP can span multiple
                # dense-DP groups, whose token balancing may produce different
                # real chunk counts. A no-grad forward here lets one group move
                # on while another is still in MoE backward, deadlocking RCCL.
                logits = self._forward_logits_packed(packed, model=self._ddp)
                dummy_loss = logits.sum() * 0.0
                if r3_routes is not None:
                    self._r3_set_backward()
                dummy_loss.backward()
                if r3_routes is not None:
                    self._r3_clear()
                del logits, dummy_loss, packed, ids_chunk, mask_chunk
                torch.cuda.synchronize()
                continue

            logits = self._forward_logits_packed(packed, model=self._ddp)

            flat_lp = packed_token_log_probs(
                logits, packed.input_ids.squeeze(0), packed.cu_seqlens,
                temperature=temperature,
            )
            token_log_probs = unpack_log_probs(
                flat_lp, packed.cu_seqlens, packed.seq_lens, S,
            )

            chunk_loss = torch.tensor(0.0, device="cuda")
            for r_local in range(chunk_B):
                r_global = cs + r_local
                sl = int(packed.seq_lens[r_local].item())
                if sl < 2:
                    continue

                lp_start = S - 1 - (sl - 1)
                row_lp = token_log_probs[r_local, lp_start:].unsqueeze(0)
                dev = row_lp.device

                am_row = am[r_global]
                real_start, real_len = self._real_block(am_row)

                def _col(name, shift, _rg=r_global, _rs=real_start):
                    x = t.get(name)
                    if x is None:
                        return None
                    x = x[_rg].to(dev)
                    s0 = _rs + (1 if shift else 0)
                    return x[s0:].reshape(1, -1)

                old_lp = _col("old_log_probs", shift=False)
                if old_lp is None:
                    continue
                resp_mask = _col("response_mask", shift=True)
                adv_t_full = t.get("advantages")
                if adv_t_full is None:
                    continue

                Lm = row_lp.shape[-1]
                if adv_t_full.dim() == 1:
                    adv = adv_t_full[r_global].to(dev).view(1, 1).expand(1, Lm).float()
                else:
                    adv = adv_t_full[r_global].to(dev)[real_start + 1:].reshape(1, -1).float()
                ris = _col("rollout_is_weights", shift=False)
                ref_lp0 = _col("ref_log_probs", shift=False)
                rlp0 = _col("rollout_log_probs", shift=False)

                cand = [row_lp, old_lp, adv]
                for v in (resp_mask, ris, ref_lp0, rlp0):
                    if v is not None:
                        cand.append(v)
                Le = min(v.shape[-1] for v in cand)
                row_lp_t = row_lp[..., :Le]
                old_lp = old_lp[..., :Le]
                adv = adv[..., :Le]
                mask = resp_mask[..., :Le].float() if resp_mask is not None else None
                ris = ris[..., :Le] if ris is not None else None
                ref_lp = ref_lp0[..., :Le] if ref_lp0 is not None else None
                rlp = rlp0[..., :Le] if rlp0 is not None else None

                if algo_name == AlgorithmName.DAPO.value:
                    row_loss = asymmetric_clip_loss(
                        row_lp_t, old_lp, adv,
                        float(_cfg("clip_ratio_low", 0.2)), float(_cfg("clip_ratio_high", 0.28)),
                        mask=mask, clip_ratio_c=float(_cfg("clip_ratio_c", 0.0)),
                        batch_num_tokens=bnt, dp_size=dp, rollout_is_weights=ris,
                    )
                else:
                    row_loss = policy_gradient_loss(
                        row_lp_t, old_lp, adv, float(_cfg("clip_ratio", 0.2)), mask=mask,
                    )
                kl_c = float(_cfg("kl_coeff", 0.0))
                if kl_c > 0.0 and ref_lp is not None:
                    row_loss = row_loss + kl_c * kl_penalty(row_lp_t, ref_lp, mask=mask)

                chunk_loss = chunk_loss + row_loss
                loss_accum += float(row_loss.detach())
                n_rows += 1
                if mask is not None:
                    with torch.no_grad():
                        tok = float(mask.sum())
                        ppo_kl_sum += float(((old_lp - row_lp_t) * mask).sum())
                        ppo_kl_tok += tok
                        neg_kl = torch.clamp(row_lp_t - old_lp, min=-20.0, max=20.0)
                        ratio = torch.exp(neg_kl)
                        clip_low = float(_cfg("clip_ratio_low", 0.2))
                        clip_high = float(_cfg("clip_ratio_high", 0.28))
                        pg1 = -adv * ratio
                        pg2 = -adv * torch.clamp(
                            ratio, 1.0 - clip_low, 1.0 + clip_high,
                        )
                        clip1 = torch.maximum(pg1, pg2)
                        pg_clip_sum += float(((pg2 > pg1).float() * mask).sum())
                        clip_c = float(_cfg("clip_ratio_c", 0.0))
                        if clip_c > 0.0:
                            pg3 = -adv * clip_c
                            pg_clip_lower_sum += float(
                                (
                                    (clip1 > pg3).float()
                                    * (adv < 0).float()
                                    * mask
                                ).sum()
                            )
                        if rlp is not None:
                            rc_kl_sum += float(((rlp - row_lp_t) * mask).sum())
                            rc_kl_tok += tok

            if chunk_loss.requires_grad:
                if r3_routes is not None:
                    self._r3_set_backward()
                chunk_loss.backward()
            if r3_routes is not None:
                self._r3_clear()
            del logits, flat_lp, token_log_probs, packed, chunk_loss
            # Retire each chunk's HIP work before reusing allocator blocks.
            # Do not empty the allocator cache here: a long run executes
            # thousands of chunks, and repeatedly releasing segments exhausted
            # HSA queue/event resources despite ample free VRAM. Expandable
            # segments handle the varying packed shapes; release once below.
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        metrics: dict[str, float] = {
            "loss": loss_accum / max(1, n_rows),
        }
        if r3_routes is not None:
            metrics.update(self._r3_metrics(r3_routes, am))
        if ppo_kl_tok > 0:
            metrics["ppo_kl_sum"] = ppo_kl_sum
            metrics["ppo_kl_tok"] = ppo_kl_tok
            metrics["actor/pg_clipfrac"] = pg_clip_sum / ppo_kl_tok
            metrics["actor/pg_clipfrac_lower"] = pg_clip_lower_sum / ppo_kl_tok
        if rc_kl_tok > 0:
            metrics["rollout_corr_kl_sum"] = rc_kl_sum
            metrics["rollout_corr_kl_tok"] = rc_kl_tok
        return metrics

    def _engine_update_policy_pp(
        self, batch, seqs, am, B, S, algo_name, temperature, bnt, dp, _cfg, t,
    ) -> dict[str, float]:
        """PP>1 training step via Megatron's pipeline schedule."""
        from functools import partial as _partial

        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        is_last_pp = mpu.is_pipeline_last_stage()

        micro_batches = []
        for r in range(B):
            start, L = self._real_block(am[r])
            if L < 2:
                micro_batches.append({"row": r, "start": start, "L": 0, "ids": None})
                continue
            ids = seqs[r, start:start + L].to("cuda")
            micro_batches.append({"row": r, "start": start, "L": L, "ids": ids})

        loss_accum = [0.0]
        n_rows = [0]
        ppo_kl = [0.0, 0.0]  # sum, tok
        rc_kl = [0.0, 0.0]   # sum, tok

        def _train_loss_func(mb, output_tensor):
            r = mb["row"]
            start = mb["start"]
            L = mb["L"]
            ids = mb["ids"]

            if L < 2:
                dummy = torch.tensor(0.0, device="cuda", requires_grad=True)
                return dummy, {}

            logits = output_tensor.view(L, -1).float() / temperature
            token_lp = self._token_logprob_train(logits[:-1], ids[1:]).view(1, -1)
            Lm = token_lp.shape[-1]
            dev = token_lp.device

            def _col(name, shift):
                x = t.get(name)
                if x is None:
                    return None
                x = x[r].to(dev)
                s0 = start + (1 if shift else 0)
                return x[s0:].reshape(1, -1)

            old_lp = _col("old_log_probs", shift=False)
            if old_lp is None:
                dummy = torch.tensor(0.0, device="cuda", requires_grad=True)
                return dummy, {}

            resp_mask = _col("response_mask", shift=True)
            adv_t_full = t.get("advantages")
            if adv_t_full is None:
                dummy = torch.tensor(0.0, device="cuda", requires_grad=True)
                return dummy, {}
            if adv_t_full.dim() == 1:
                adv = adv_t_full[r].to(dev).view(1, 1).expand(1, Lm).float()
            else:
                adv = adv_t_full[r].to(dev)[start + 1:].reshape(1, -1).float()
            ris = _col("rollout_is_weights", shift=False)
            ref_lp0 = _col("ref_log_probs", shift=False)
            rlp0 = _col("rollout_log_probs", shift=False)

            cand = [token_lp, old_lp, adv]
            for v in (resp_mask, ris, ref_lp0, rlp0):
                if v is not None:
                    cand.append(v)
            Le = min(v.shape[-1] for v in cand)
            token_lp = token_lp[..., :Le]
            old_lp = old_lp[..., :Le]
            adv = adv[..., :Le]
            mask = resp_mask[..., :Le].float() if resp_mask is not None else None
            ris = ris[..., :Le] if ris is not None else None
            ref_lp = ref_lp0[..., :Le] if ref_lp0 is not None else None
            rlp = rlp0[..., :Le] if rlp0 is not None else None

            if algo_name == AlgorithmName.DAPO.value:
                loss = asymmetric_clip_loss(
                    token_lp, old_lp, adv,
                    float(_cfg("clip_ratio_low", 0.2)), float(_cfg("clip_ratio_high", 0.28)),
                    mask=mask, clip_ratio_c=float(_cfg("clip_ratio_c", 0.0)),
                    batch_num_tokens=bnt, dp_size=dp, rollout_is_weights=ris,
                )
            else:
                loss = policy_gradient_loss(
                    token_lp, old_lp, adv, float(_cfg("clip_ratio", 0.2)), mask=mask,
                )
            kl_c = float(_cfg("kl_coeff", 0.0))
            if kl_c > 0.0 and ref_lp is not None:
                loss = loss + kl_c * kl_penalty(token_lp, ref_lp, mask=mask)

            loss_accum[0] += float(loss.detach())
            n_rows[0] += 1
            if mask is not None:
                with torch.no_grad():
                    tok = float(mask.sum())
                    ppo_kl[0] += float(((old_lp - token_lp) * mask).sum())
                    ppo_kl[1] += tok
                    if rlp is not None:
                        rc_kl[0] += float(((rlp - token_lp) * mask).sum())
                        rc_kl[1] += tok

            # Megatron schedule divides loss by num_microbatches; pre-multiply
            # to keep gradient magnitude consistent with PP=1.
            return loss * B, {"loss": float(loss.detach())}

        mb_iter = iter(micro_batches)

        def _forward_step(data_iterator, model):
            mb = next(data_iterator)
            if mb["L"] < 2:
                dummy = torch.zeros(
                    1, 1, self._dims.hidden, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True,
                )
                return dummy, _partial(_train_loss_func, mb)
            ids = mb["ids"]
            L = mb["L"]
            inp = ids.view(1, L)
            pos = torch.arange(L, device=ids.device).view(1, L)
            out = model(input_ids=inp, position_ids=pos, attention_mask=None)
            return out, _partial(_train_loss_func, mb)

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        forward_backward_func = get_forward_backward_func()
        config = self._tfcfg
        saved_timers = config.timers
        config.timers = None
        forward_backward_func(
            forward_step_func=_forward_step,
            data_iterator=mb_iter,
            model=[self._ddp],
            num_microbatches=B,
            seq_length=S,
            micro_batch_size=1,
            forward_only=False,
        )
        config.timers = saved_timers

        grad_norm = self._optimizer_step()
        lr = self._sched_step()
        metrics = {
            "loss": loss_accum[0] / max(1, n_rows[0]),
            "lr": lr,
            "grad_norm": grad_norm,
        }
        if ppo_kl[1] > 0:
            metrics["ppo_kl_sum"] = ppo_kl[0]
            metrics["ppo_kl_tok"] = ppo_kl[1]
        if rc_kl[1] > 0:
            metrics["rollout_corr_kl_sum"] = rc_kl[0]
            metrics["rollout_corr_kl_tok"] = rc_kl[1]
        return metrics

    def _engine_update_sft(self, batch: DataProto) -> dict[str, float]:
        """SFT training update: forward → log_probs → NLL → backward.

        Uses global token-mean normalization with cross-DP all-reduce,
        matching the FSDP2 path's ``sft_loss()`` behavior.
        Row-by-row forward only (pack_sequences assumes left-padded input
        but SFT data is right-padded).
        """
        meta = dict(batch.meta)
        t = batch.tensors
        seqs = t["input_ids"]
        am = t.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        loss_masks = t["loss_mask"]
        B, S = seqs.shape
        dp = int(meta.get("dp_size", self.get_data_parallel_size()) or 1)

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        rows = []
        for r in range(B):
            start, L = self._real_block(am[r])
            if L >= 2:
                rows.append((r, start, L))

        # Global token count for token-mean normalization (cross-DP all-reduce).
        local_tokens = sum(
            float(loss_masks[r, start + 1:start + L].sum()) for r, start, L in rows
        )
        num_tokens_t = torch.tensor(local_tokens, device="cuda")
        dp_group = self.get_data_parallel_group()
        if dp_group is not None and dist.is_initialized():
            dist.all_reduce(num_tokens_t, group=dp_group)
        global_num_tokens = max(int(num_tokens_t.item()), 1)

        loss_accum = 0.0
        token_accum = local_tokens

        for r, start, L in rows:
            ids = seqs[r, start:start + L].to("cuda")
            logits = self._forward_logits(ids, model=self._ddp)
            token_lp = self._token_logprob_train(logits[:-1], ids[1:])
            mask = loss_masks[r, start + 1:start + L].to(token_lp.device).float()
            Le = min(token_lp.shape[0], mask.shape[0])
            token_lp = token_lp[:Le]
            mask = mask[:Le]
            if mask.sum() < 1:
                del logits, token_lp
                continue
            loss = -(token_lp * mask).sum() / global_num_tokens * dp
            loss.backward()
            loss_accum += float(-(token_lp.detach() * mask).sum())
            del logits, token_lp, loss

        torch.cuda.empty_cache()
        grad_norm = self._optimizer_step()
        lr = self._sched_step()
        avg_loss = loss_accum / max(token_accum, 1)
        return {
            "loss": avg_loss,
            "sft_loss": avg_loss,
            "num_tokens": token_accum,
            "lr": lr,
            "grad_norm": grad_norm,
        }





    # ---- weight sync: Megatron -> HF named tensors ----
    def get_per_tensor_param(self, **kwargs):
        """Yield ``(hf_name, full_tensor)`` with PP/TP/EP all-gather.

        With TP>1, EP>1, or PP>1 each rank holds only a shard.  We:
        1. TP all-gather dense params / ETP all-gather expert params
        2. EP all-gather fused expert tensors (weight1/weight2)
        3. PP broadcast each stage's params to all PP ranks
        4. Convert the full model to HF key space via ``megatron_to_hf``
        """
        assert self.module is not None
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group() if tp_size > 1 else None
        ep_size = getattr(self, "_ep_size", 1)
        ep_group = (mpu.get_expert_model_parallel_group()
                    if ep_size > 1 else None)
        etp_size = mpu.get_expert_tensor_parallel_world_size()
        etp_group = (mpu.get_expert_tensor_parallel_group()
                     if etp_size > 1 else None)
        pp_size = getattr(self, "_pp_size", 1)
        pp_rank = getattr(self, "_pp_rank", 0)

        # --- Phase 1: TP all-gather local params ---
        local_full: dict[str, torch.Tensor] = {}
        for raw_name, param in self.module.named_parameters():
            name = raw_name
            for pre in ("module.module.", "module."):
                if name.startswith(pre):
                    name = name[len(pre):]
                    break
            p = param.data

            is_tp = getattr(param, "tensor_model_parallel", False)
            if is_tp:
                is_expert = ".experts." in name and ".shared_experts." not in name
                g_size = etp_size if is_expert else tp_size
                g_group = etp_group if is_expert else tp_group
                if g_size > 1 and g_group is not None:
                    parts = [torch.empty_like(p) for _ in range(g_size)]
                    dist.all_gather(parts, p, group=g_group)
                    p = _gather_with_stride(
                        parts,
                        getattr(param, "partition_dim", 0),
                        getattr(param, "partition_stride", 1),
                    )
            local_full[name] = p

        # --- Phase 2: EP all-gather expert tensors ---
        if ep_size > 1 and ep_group is not None:
            expert_keys = [
                k for k in local_full
                if ".experts." in k and ".shared_experts." not in k
            ]
            # For SequentialMLP expert layout, each EP rank exposes
            # local_experts.{0..num_local-1}. We need to rebuild full
            # local_experts.{0..num_experts-1} before megatron_to_hf(ep_size=1).
            num_local_experts = max(1, self._dims.num_experts // ep_size)
            for key in expert_keys:
                local_t = local_full[key].contiguous()
                if not local_t.is_cuda:
                    local_t = local_t.cuda()
                gathered = [torch.empty_like(local_t) for _ in range(ep_size)]
                dist.all_gather(gathered, local_t, group=ep_group)
                if "weight1" in key:
                    local_full[key] = torch.cat(gathered, dim=1).contiguous()
                elif "weight2" in key:
                    local_full[key] = torch.cat(gathered, dim=0).contiguous()
                elif ".local_experts." in key:
                    head, rest = key.split("local_experts.", 1)
                    local_idx_s, tail = rest.split(".", 1)
                    local_idx = int(local_idx_s)
                    # Remove the rank-local entry before rebuilding global
                    # expert keys.  For src_ep_rank=0 the rebuilt key equals
                    # ``key``; deleting afterwards would discard expert 0.
                    del local_full[key]
                    for src_ep_rank, part in enumerate(gathered):
                        global_idx = src_ep_rank * num_local_experts + local_idx
                        gkey = f"{head}local_experts.{global_idx}.{tail}"
                        local_full[gkey] = part.contiguous()
                else:
                    local_full[key] = gathered[0]

        # --- Phase 3: PP broadcast (each stage contributes its layers) ---
        if pp_size > 1:
            pp_group = mpu.get_pipeline_model_parallel_group()
            pp_global_ranks = dist.get_process_group_ranks(pp_group)

            # Each PP rank converts its local params to HF keys (with
            # correct global layer indices), then broadcasts to others.
            local_hf = dict(megatron_to_hf(
                list(local_full.items()), self._dims,
                ep_rank=0, ep_size=1,
                pp_rank=pp_rank, pp_size=pp_size,
                layers_per_pp_rank=self._layers_per_pp_rank,
                use_grouped_mlp=self._use_grouped_mlp,
            ))

            # Exchange param metadata across PP group so every rank
            # knows the full set of keys/shapes/dtypes.
            all_meta: list = [None] * pp_size
            my_meta = {k: (v.shape, v.dtype) for k, v in local_hf.items()}
            dist.all_gather_object(all_meta, (pp_rank, my_meta), group=pp_group)

            full_hf: dict[str, torch.Tensor] = {}
            for src_pp, meta in all_meta:
                src_global = pp_global_ranks[src_pp]
                for key, (shape, dtype) in meta.items():
                    if src_pp == pp_rank:
                        t = local_hf[key]
                        if not t.is_cuda:
                            t = t.cuda()
                    else:
                        t = torch.empty(shape, dtype=dtype, device="cuda")
                    dist.broadcast(t, src=src_global, group=pp_group)
                    full_hf[key] = t

            def _gen():
                for k, v in full_hf.items():
                    yield k, v
            return _gen(), None

        # PP=1 path: convert directly
        gen = megatron_to_hf(
            list(local_full.items()), self._dims,
            ep_rank=0, ep_size=1,
            pp_rank=0, pp_size=1,
            use_grouped_mlp=self._use_grouped_mlp,
        )
        return gen, None

    def _dist_sharded_state_dict(self, is_loading: bool):
        """Build a low-memory, DP-reshardable model and optimizer state."""
        model_state = self.module.sharded_state_dict()
        optimizer_state = self.optimizer.sharded_state_dict(
            model_state,
            is_loading=is_loading,
            metadata={"distrib_optim_sharding_type": "dp_reshardable"},
        )
        return {"model": model_state, "optimizer": optimizer_state}

    def save_dist_checkpoint(self, local_path: str, global_step: int = 0) -> bool:
        """Save directly from each rank's owned optimizer buffers.

        ``dp_reshardable`` avoids the DP-zero gather used by the legacy actor
        checkpoint path, so saving does not materialize full optimizer copies
        on every worker.
        """
        import megatron.core.dist_checkpointing as dc

        state = self._dist_sharded_state_dict(is_loading=False)
        state["global_step"] = int(global_step)
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        os.makedirs(local_path, exist_ok=True)
        dc.save(state, str(local_path))
        if dist.is_initialized():
            dist.barrier()
        return True

    @staticmethod
    def _patch_hybrid_optimizer_checkpoint_load() -> None:
        """Tolerate native-FP32 params in MCore's CPU-offload load hook.

        MCore 0.15's ``HybridDeviceOptimizer`` assumes every state entry has a
        separate FP32 master parameter. Qwen3 MoE also has parameters that are
        already FP32, so those entries are intentionally absent from
        ``param_to_fp32_param`` and the upstream hook raises ``KeyError``.
        """
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

        if getattr(HybridDeviceOptimizer, "_lumenrl_safe_fp32_restore", False):
            return

        def _update_mapped_fp32_params(optimizer) -> None:
            if not optimizer.param_update_in_fp32:
                return
            for param, value in optimizer.state.items():
                fp32_param = optimizer.param_to_fp32_param.get(param)
                master_param = value.get("master_param")
                if fp32_param is not None and master_param is not None:
                    fp32_param.data.copy_(master_param)

        HybridDeviceOptimizer._update_fp32_params_by_new_state = (
            _update_mapped_fp32_params
        )
        HybridDeviceOptimizer._lumenrl_safe_fp32_restore = True

    def load_dist_checkpoint(self, local_path: str) -> int:
        """Restore a checkpoint saved by :meth:`save_dist_checkpoint`."""
        import megatron.core.dist_checkpointing as dc

        self._patch_hybrid_optimizer_checkpoint_load()
        state = self._dist_sharded_state_dict(is_loading=True)
        state["global_step"] = 0
        loaded = dc.load(state, str(local_path))
        self.module.load_state_dict(loaded["model"])
        self.optimizer.load_state_dict(loaded["optimizer"])
        if self.lr_scheduler is not None and loaded.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(loaded["lr_scheduler"])
        if dist.is_initialized():
            dist.barrier()
        return int(loaded.get("global_step", 0))


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    pass


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngine):
    pass
