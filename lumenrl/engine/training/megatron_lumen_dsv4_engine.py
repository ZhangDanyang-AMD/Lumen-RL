# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Megatron-Core training engine for DeepSeek-V4-Flash (Lumen DSV4 spec).

Subclasses :class:`MegatronEngine` to use the Lumen DSV4 layer spec
(MLA attention, Hyper-Connection, compressor/indexer, MoE) instead of
the standard Qwen3 GPT layer spec.  Registered as ``backend="megatron_lumen_dsv4"``.

Key differences from the Qwen3 parent:
  - ``DSV4Dims`` replaces ``Qwen3Dims``
  - ``get_dsv4_spec()`` replaces ``get_gpt_layer_local_spec()``
  - ``hf_to_dsv4_megatron()`` / ``dsv4_megatron_to_hf()`` for weight I/O
  - MoE router gates, expert bias, and tid2eid hash table are frozen
  - FP32 parameters (HC, attn_sink, compressor APE/norm) are preserved
"""

from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from lumenrl.engine.training.base_engine import EngineRegistry
from lumenrl.engine.training.dsv4_megatron_bridge import (
    DSV4Dims,
    DSV4_FLASH_COMPRESS_RATIOS,
    dsv4_megatron_to_hf,
    hf_to_dsv4_megatron,
)
from lumenrl.engine.training.megatron_engine import (
    MegatronEngine,
    _gather_with_stride,
    _shard_with_stride,
)
from lumenrl.engine.training.qwen3_megatron_bridge import (
    _pp_layer_range,
    load_hf_safetensors,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "INFO"))


class MegatronLumenDSV4Engine(MegatronEngine):
    """Megatron-Core GPTModel engine for DeepSeek-V4-Flash (BF16, TP/PP/EP/DP)."""

    def initialize(self) -> None:
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig

        ec = self.engine_config
        tp = int(ec.get("tensor_model_parallel_size", 1))
        pp = int(ec.get("pipeline_model_parallel_size", 1))
        cp = int(ec.get("context_parallel_size", 1))
        ep = int(ec.get("expert_model_parallel_size", 1))
        seed = int(ec.get("seed", 42))
        self._attention_backend = str(ec.get("attention_backend") or "unfused").lower()
        self._use_packed_sequences = bool(ec.get("use_packed_sequences", True))
        self._logprob_chunk_size = int(ec.get("log_probs_chunk_size") or 0)
        self._r3_enabled = bool(ec.get("moe_enable_routing_replay", False))

        etp = ec.get("expert_tensor_parallel_size", None)
        if etp is not None:
            etp = int(etp)

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
                file=sys.stderr, flush=True,
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

        # ---- HF config -> DSV4 dims / TransformerConfig ----
        cfg_path = os.path.join(self.model_name, "config.json")
        with open(cfg_path) as fh:
            hf = json.load(fh)

        # HF config field names differ between model families. DSV4 uses:
        #   head_dim (not kv_lora_rank), qk_rope_head_dim (not qk_pos_emb_head_dim),
        #   n_routed_experts (not num_experts), compress_ratios (not dsv4_compress_ratios),
        #   sliding_window (not dsv4_window_size), etc.
        num_experts = int(
            hf.get("n_routed_experts") or hf.get("num_experts", 0)
            or ec.get("num_experts", 0)
        )
        moe_ffn = int(hf.get("moe_intermediate_size", 0) or hf.get("moe_ffn_hidden_size", 0) or 0)
        n_shared = int(hf.get("n_shared_experts", 0) or 0)
        shared_ffn = int(
            hf.get("shared_expert_intermediate_size", 0)
            or (moe_ffn * n_shared if n_shared else 0)
        )
        head_dim = int(hf.get("head_dim") or hf.get("kv_lora_rank", 512))
        q_lora_rank = int(hf.get("q_lora_rank", 1024))
        qk_pos_emb_head_dim = int(
            hf.get("qk_rope_head_dim") or hf.get("qk_pos_emb_head_dim", 64)
        )
        v_head_dim = int(hf.get("v_head_dim") or head_dim)

        compress_ratios_raw = (
            ec.get("dsv4_compress_ratios")
            or hf.get("compress_ratios")
            or hf.get("dsv4_compress_ratios")
        )
        if compress_ratios_raw is not None:
            if isinstance(compress_ratios_raw, str):
                compress_ratios = [int(x) for x in compress_ratios_raw.split()]
            else:
                compress_ratios = [int(x) for x in compress_ratios_raw]
        else:
            compress_ratios = list(DSV4_FLASH_COMPRESS_RATIOS)

        hc_mult = int(ec.get("dsv4_hc_mult") or hf.get("hc_mult") or hf.get("dsv4_hc_mult", 4))
        o_groups = int(ec.get("dsv4_o_groups") or hf.get("o_groups") or hf.get("dsv4_o_groups", 8))
        o_lora_rank = int(ec.get("dsv4_o_lora_rank") or hf.get("o_lora_rank") or hf.get("dsv4_o_lora_rank", 1024))
        n_hash_layers = int(
            ec.get("dsv4_n_hash_layers") or hf.get("num_hash_layers")
            or hf.get("dsv4_n_hash_layers", 3)
        )
        window_size = int(
            ec.get("dsv4_window_size") or hf.get("sliding_window")
            or hf.get("dsv4_window_size", 128)
        )
        moe_topk = int(hf.get("num_experts_per_tok", ec.get("moe_router_topk", 6)))

        self._dims = DSV4Dims(
            num_layers=hf["num_hidden_layers"],
            hidden=hf["hidden_size"],
            num_heads=hf["num_attention_heads"],
            num_kv_groups=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            head_dim=head_dim,
            ffn=hf.get("intermediate_size", moe_ffn),
            vocab=hf["vocab_size"],
            num_experts=num_experts,
            moe_ffn=moe_ffn,
            shared_expert_ffn=shared_ffn,
            shared_expert_gate=bool(hf.get("shared_expert_gate", False)),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
            hc_mult=hc_mult,
            n_hash_layers=n_hash_layers,
            window_size=window_size,
            compress_ratios=compress_ratios,
            moe_topk=moe_topk,
        )

        # ---- Build MLATransformerConfig directly ----
        # Bypass core_transformer_config_from_args (too many version-dependent args).
        # MLATransformerConfig is a dataclass — pass only the fields it declares.
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        compress_rope_theta = float(
            ec.get("dsv4_compress_rope_theta")
            or hf.get("compress_rope_theta")
            or hf.get("dsv4_compress_rope_theta", 160000)
        )
        first_pp_layers = ec.get("num_layers_in_first_pipeline_stage")
        last_pp_layers = ec.get("num_layers_in_last_pipeline_stage")

        recompute_kwargs = {}
        if ec.get("recompute_granularity"):
            recompute_kwargs["recompute_granularity"] = ec["recompute_granularity"]
            recompute_kwargs["recompute_method"] = ec.get("recompute_method") or "uniform"
            recompute_kwargs["recompute_num_layers"] = int(ec.get("recompute_num_layers") or 1)

        pp_kwargs = {}
        if first_pp_layers is not None:
            pp_kwargs["num_layers_in_first_pipeline_stage"] = int(first_pp_layers)
        if last_pp_layers is not None:
            pp_kwargs["num_layers_in_last_pipeline_stage"] = int(last_pp_layers)
        if pp > 1:
            pp_kwargs["variable_seq_lengths"] = True

        moe_kwargs = {}
        if num_experts > 0:
            moe_kwargs.update(
                num_moe_experts=num_experts,
                moe_ffn_hidden_size=moe_ffn if moe_ffn > 0 else hf.get("intermediate_size", 2048),
                moe_router_topk=moe_topk,
                moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", False)),
                moe_token_dispatcher_type=str(ec.get("moe_token_dispatcher_type", "alltoall")),
                expert_model_parallel_size=ep,
                moe_enable_routing_replay=self._r3_enabled,
            )
            if shared_ffn > 0:
                moe_kwargs["moe_shared_expert_intermediate_size"] = shared_ffn

        tfcfg = MLATransformerConfig(
            # Core
            num_layers=hf["num_hidden_layers"],
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_query_groups=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            ffn_hidden_size=hf.get("intermediate_size", moe_ffn),
            kv_channels=head_dim,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=False,
            normalization="RMSNorm",
            layernorm_epsilon=hf.get("rms_norm_eps", 1e-6),
            qk_layernorm=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_softmax_in_fp32=bool(ec.get("attention_softmax_in_fp32", True)),
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            sequence_parallel=bool(ec.get("sequence_parallel", False)),
            use_cpu_initialization=True,
            perform_initialization=True,
            # MLA
            q_lora_rank=q_lora_rank,
            kv_lora_rank=head_dim,
            qk_head_dim=head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            rotary_base=hf.get("rope_theta", 10000.0),
            rotary_scaling_factor=hf.get("rotary_scaling_factor", 16),
            original_max_position_embeddings=hf.get("original_max_position_embeddings", 65536),
            beta_fast=32,
            beta_slow=1,
            # DSV4 (patch-added fields)
            experimental_attention_variant="dsv4",
            dsv4_hc_mult=hc_mult,
            dsv4_hc_sinkhorn_iters=int(ec.get("dsv4_hc_sinkhorn_iters", 20)),
            dsv4_hc_eps=float(ec.get("dsv4_hc_eps", 1e-6)),
            dsv4_o_groups=o_groups,
            dsv4_o_lora_rank=o_lora_rank,
            dsv4_window_size=window_size,
            dsv4_n_hash_layers=n_hash_layers,
            dsv4_compress_ratios=compress_ratios,
            dsv4_compress_rope_theta=compress_rope_theta,
            dsa_indexer_n_heads=int(ec.get("dsa_indexer_n_heads") or hf.get("index_n_heads") or hf.get("dsa_indexer_n_heads", 64)),
            dsa_indexer_head_dim=int(ec.get("dsa_indexer_head_dim") or hf.get("index_head_dim") or hf.get("dsa_indexer_head_dim", 128)),
            dsa_indexer_topk=int(ec.get("dsa_indexer_topk") or hf.get("index_topk") or hf.get("dsa_indexer_topk", 512)),
            **recompute_kwargs,
            **moe_kwargs,
            **pp_kwargs,
        )
        # miles_dsa_topk_backend is not a dataclass field — set via setattr
        # (get_dsv4_spec reads it from config at runtime)
        tfcfg.miles_dsa_topk_backend = str(ec.get("miles_dsa_topk_backend", "torch"))

        # mock_args for get_dsv4_spec (it only reads miles_dsa_topk_backend)
        mock_args = SimpleNamespace(
            miles_dsa_topk_backend=tfcfg.miles_dsa_topk_backend,
        )

        # ---- Build GPTModel using Lumen's get_dsv4_spec ----
        from lumen.models.dsv4.megatron.spec import get_dsv4_spec
        from megatron.core.models.gpt.gpt_model import GPTModel

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._pp_rank = pp_rank
        self._pp_size = pp_size
        self._tp_rank = mpu.get_tensor_model_parallel_rank()
        self._tp_size = tp

        spec = get_dsv4_spec(mock_args, tfcfg, vp_stage=0)

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
            config=tfcfg,
            transformer_layer_spec=spec,
            vocab_size=hf["vocab_size"],
            max_sequence_length=hf.get("max_position_embeddings", 65536),
            pre_process=(pp_rank == 0),
            post_process=(pp_rank == pp_size - 1),
            parallel_output=False,
            position_embedding_type="rope",
            rotary_base=hf.get("rope_theta", 10000.0),
            share_embeddings_and_output_weights=hf.get("tie_word_embeddings", False),
        )

        # ---- Detect expert format ----
        self._use_grouped_mlp = any(
            ".experts.weight1" in n for n, _ in model.named_parameters()
        )

        # ---- Load HF weights ----
        ep_rank = mpu.get_expert_model_parallel_rank() if num_experts > 0 else 0
        ep_size = mpu.get_expert_model_parallel_world_size() if num_experts > 0 else 1
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        logger.info(
            "MegatronLumenDSV4Engine[%d]: loading HF weights from %s (ep_rank=%d/%d, experts=%d)",
            self._rank(), self.model_name, ep_rank, ep_size, num_experts,
        )
        hf_state = load_hf_safetensors(self.model_name)
        meg_state = hf_to_dsv4_megatron(
            hf_state, self._dims,
            ep_rank=ep_rank, ep_size=ep_size,
            pp_rank=pp_rank, pp_size=pp_size,
            layers_per_pp_rank=self._layers_per_pp_rank,
            use_grouped_mlp=self._use_grouped_mlp,
        )
        del hf_state

        # TP shard: hf_to_dsv4_megatron returns full tensors; the GPTModel's
        # parallel linears only hold 1/tp of each weight. Shard using the
        # partition metadata set by Megatron.
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
            raise RuntimeError(f"Megatron DSV4 load missing keys: {real_missing[:10]} ...")
        if incompat.unexpected_keys:
            logger.warning("Megatron DSV4 load unexpected keys: %s", incompat.unexpected_keys[:6])
        del meg_state
        import gc
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
        # Move to GPU. Preserve FP32 params that have _keep_fp32 marker.
        model = model.cuda()
        for name, param in model.named_parameters():
            if getattr(param, "_keep_fp32", False):
                continue
            if param.dtype != torch.bfloat16 and param.is_floating_point():
                param.data = param.data.bfloat16()
        self.module = model
        self._tfcfg = tfcfg
        _gpu_diag("AFTER model.cuda()")

        # ---- Freeze non-trainable params ----
        # MoE router gates, expert bias, and tid2eid hash table
        for name, param in self.module.named_parameters():
            if "mlp.router.weight" in name:
                param.requires_grad_(False)
            elif "mlp.router.expert_bias" in name:
                param.requires_grad_(False)
            elif "mlp.router.tid2eid" in name:
                param.requires_grad_(False)

        n_params = sum(p.numel() for p in self.module.parameters())
        n_grad = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        print(
            f"[GPU_DIAG rank={self._rank()}] params={n_params:,} "
            f"({n_params * 2 / 2**30:.2f} GiB bf16), requires_grad={n_grad:,}",
            file=sys.stderr, flush=True,
        )

        # ---- Megatron DistributedDataParallel + optimizer ----
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
                "MegatronLumenDSV4Engine: model+distributed-optimizer ready, %d params, "
                "dp_size=%d, MILES-R3=%s",
                n, self.get_data_parallel_size(), self._r3_enabled,
            )

    # ---- weight sync: Megatron -> HF named tensors ----
    def get_per_tensor_param(self, **kwargs):
        """Yield ``(hf_name, full_tensor)`` with PP/TP/EP all-gather.

        Same architecture-agnostic gather logic as the parent (phases 1-2),
        but uses ``dsv4_megatron_to_hf`` for name conversion (phase 3 + PP=1).
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

            local_hf = dict(dsv4_megatron_to_hf(
                list(local_full.items()), self._dims,
                pp_rank=pp_rank, pp_size=pp_size,
                layers_per_pp_rank=self._layers_per_pp_rank,
                use_grouped_mlp=self._use_grouped_mlp,
            ))

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
        gen = dsv4_megatron_to_hf(
            list(local_full.items()), self._dims,
            pp_rank=0, pp_size=1,
            use_grouped_mlp=self._use_grouped_mlp,
        )
        return gen, None


@EngineRegistry.register(model_type="language_model", backend="megatron_lumen_dsv4")
class MegatronLumenDSV4EngineWithLMHead(MegatronLumenDSV4Engine):
    pass


@EngineRegistry.register(model_type="value_model", backend="megatron_lumen_dsv4")
class MegatronLumenDSV4EngineWithValueHead(MegatronLumenDSV4Engine):
    pass
