# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""DeepSeek-V4 (``deepseek_v4``) support for the native Megatron-Core engine.

DSv4 is the first model family LumenRL runs that the engine's inline
``TransformerConfig`` cannot express: it is MLA (so ``num_query_groups`` /
``kv_channels`` are meaningless), its residual stream is 4-dimensional
(hyper-connections), its attention is per-layer heterogeneous (sliding /
compressed+indexed / hyper-compressed), and its router is hash-based on the
first ``num_hash_layers``. All of that lives in ``MLATransformerConfig`` fields
that only exist in a Megatron carrying the DSv4 patch.

Three things are needed and provided here:

``build_dsv4_config``
    HF ``config.json`` -> ``MLATransformerConfig``. This is the LumenRL-native
    twin of the 85 CLI flags in miles' ``deepseek-v4-flash.sh``; every value is
    derived from the HF config rather than hard-coded. ``probe_60`` (see the
    DSv4 runbook) asserts field-for-field equality against what Megatron's own
    ``parse_args`` produces from those flags.

``build_dsv4_spec``
    The per-layer spec, from the DSv4 plugin. Heterogeneous, so
    ``get_gpt_decoder_block_spec`` cannot produce it.

``load_dsv4_dist_checkpoint``
    Model-only ``dist_checkpointing.load``. The weights arrive as a torch_dist
    checkpoint converted offline (FP8 native -> bf16 HF -> torch_dist), because
    DSv4 ships block-quantized FP8 that the HF-safetensors bridge cannot read.

The DSv4 plugin (``miles_plugins``) and the DSv4-patched Megatron are *not*
vendored into this repo yet; both must be on ``sys.path`` before the engine
initializes. See ``docs/`` -- unresolved at time of writing.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import torch
import torch.nn.functional as F

MODEL_TYPE = "deepseek_v4"

# YaRN + MLA defaults that DSv4 does not spell out in its HF config. miles'
# deepseek-v4-flash.sh passes them explicitly; Megatron's own defaults differ,
# so they cannot be left unset.
_KV_LORA_RANK = 512
_NCCL_ALGO_CHOICES = ("Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS")


def is_dsv4(hf: dict) -> bool:
    """Does this HF config describe a DeepSeek-V4 model?"""
    if str(hf.get("model_type", "")) == MODEL_TYPE:
        return True
    return any("DeepseekV4" in a for a in hf.get("architectures", []))


def _require_dsv4_megatron() -> type:
    """Return ``MLATransformerConfig``, or explain why DSv4 cannot be built.

    Stock megatron-core has ``MLATransformerConfig`` but none of the ``dsv4_*``
    fields, and the failure would otherwise surface as an opaque
    ``unexpected keyword argument`` deep in a dataclass constructor.
    """
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    have = {f.name for f in dataclasses.fields(MLATransformerConfig)}
    need = {"dsv4_mode", "dsv4_hc_mult", "dsv4_compress_ratios", "experimental_attention_variant"}
    missing = sorted(need - have)
    if missing:
        import megatron.core

        raise RuntimeError(
            f"megatron-core {megatron.core.__version__} at "
            f"{os.path.dirname(os.path.dirname(megatron.core.__file__))} has no DSv4 support "
            f"(missing config fields: {missing}). Put the DSv4-patched Megatron ahead of it on "
            f"sys.path/PYTHONPATH."
        )
    return MLATransformerConfig


def enable_deterministic_mode() -> None:
    """LumenRL's equivalent of Megatron's ``--deterministic-mode``.

    The engine does not go through Megatron's argument parser, so the three
    side effects of that flag have to be reproduced by hand. Without them DSv4
    forwards disagree with themselves run-to-run on ~1.6% of argmaxes, which is
    larger than the rollout/train gap the whole pipeline is trying to measure.
    (Setting ``config.deterministic_mode`` alone is not enough: it only steers
    Megatron's own kernel choices.)
    """
    algo = os.environ.get("NCCL_ALGO")
    if algo not in _NCCL_ALGO_CHOICES:
        raise RuntimeError(
            f"deterministic mode needs NCCL_ALGO set to one of {_NCCL_ALGO_CHOICES}, got {algo!r}. "
            f"(This image leaves it unset, which also trips Megatron's own assert.)"
        )
    torch.use_deterministic_algorithms(True)


def build_dsv4_config(
    hf: dict,
    ec: dict,
    *,
    tp: int,
    pp: int,
    cp: int,
    ep: int,
    etp: int,
    sp: bool,
    deterministic: bool = True,
) -> Any:
    """Build the ``MLATransformerConfig`` for a DeepSeek-V4 model.

    Mirrors ``core_transformer_config_from_args`` applied to miles'
    ``deepseek-v4-flash.sh`` MODEL_ARGS. Anything that reads from ``hf`` is a
    genuine model property; anything constant is a DSv4 architectural fact that
    the HF config happens not to record (see ``_KV_LORA_RANK``) or a training
    policy LumenRL owns (dtype, parallel sizes, recompute).
    """
    cls = _require_dsv4_megatron()

    num_layers = int(hf["num_hidden_layers"])
    moe_ffn = int(hf["moe_intermediate_size"])
    rope = hf.get("rope_scaling") or {}

    # The HF config keeps the full-model list even in a truncated slice, and
    # Megatron indexes it by local layer -- a 44-entry list on a 4-layer model
    # is not an error, it just has to be cut.
    ratios = [int(r) for r in hf["compress_ratios"][:num_layers]]

    # noaux_tc == "no auxiliary loss, top-k with expert-bias correction".
    expert_bias = str(hf.get("topk_method", "")) == "noaux_tc"

    n_shared = int(hf.get("n_shared_experts", 0) or 0)

    recompute: dict = {}
    if ec.get("recompute_granularity"):
        recompute = dict(
            recompute_granularity=ec["recompute_granularity"],
            recompute_method=ec.get("recompute_method") or "uniform",
            recompute_num_layers=int(ec.get("recompute_num_layers") or 1),
        )

    return cls(
        # ---- shape ----
        num_layers=num_layers,
        hidden_size=int(hf["hidden_size"]),
        num_attention_heads=int(hf["num_attention_heads"]),
        # MLA derives its own head geometry; DSv4's HF ``num_key_value_heads=1``
        # describes the latent KV, not a GQA group count, and feeding it to
        # ``num_query_groups`` would build a 1-group GQA attention instead.
        num_query_groups=None,
        ffn_hidden_size=moe_ffn,
        vocab_size=int(hf["vocab_size"]),
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=bool(hf.get("attention_bias", False)),
        normalization="RMSNorm",
        layernorm_epsilon=float(hf.get("rms_norm_eps", 1e-6)),
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        # SwiGLU is clamped in DSv4 (``swiglu_limit``), and the clamp is not
        # applied to the shared expert.
        activation_func_clamp_value=float(hf["swiglu_limit"]),
        activation_func_clamp_shared_expert=False,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        # Megatron's argument parser defaults these on and the DSv4 reference run
        # inherited them; the ``TransformerConfig`` dataclass defaults are the
        # opposite, so leaving them out would quietly pick different kernels.
        bias_dropout_fusion=True,
        persist_layer_norm=True,
        deallocate_pipeline_outputs=True,
        cp_comm_type="p2p",
        # ---- MLA ----
        multi_latent_attention=True,
        q_lora_rank=int(hf["q_lora_rank"]),
        kv_lora_rank=_KV_LORA_RANK,
        qk_head_dim=int(hf["head_dim"]),
        qk_pos_emb_head_dim=int(hf["qk_rope_head_dim"]),
        v_head_dim=int(hf["head_dim"]),
        rope_type="yarn",
        rotary_base=float(hf.get("rope_theta", 10000)),
        rotary_scaling_factor=float(rope.get("factor", 1)),
        original_max_position_embeddings=int(rope.get("original_max_position_embeddings", 4096)),
        beta_fast=float(rope.get("beta_fast", 32)),
        beta_slow=float(rope.get("beta_slow", 1)),
        apply_rope_fusion=False,
        # ---- MoE ----
        num_moe_experts=int(hf["n_routed_experts"]),
        moe_layer_freq=[1] * num_layers,
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=int(hf["num_experts_per_tok"]),
        moe_shared_expert_intermediate_size=(n_shared * moe_ffn) or None,
        # NOT the Qwen3 argument. DSv4 scores with sqrtsoftplus + expert bias and
        # group-limited top-k, where the "renormalization cancels the softmax
        # denominator" equivalence behind the dense-MoE default does not hold.
        # miles trains with pre_softmax=True; flipping it silently changes the
        # gate scale.
        moe_router_pre_softmax=True,
        moe_router_score_function=str(hf.get("scoring_func", "softmax")),
        moe_router_enable_expert_bias=expert_bias,
        moe_router_topk_scaling_factor=float(hf.get("routed_scaling_factor", 1.0)),
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=float(ec.get("moe_aux_loss_coeff", 0.0) or 0.0),
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", True)),
        moe_permute_fusion=bool(ec.get("moe_permute_fusion", False)),
        # ---- DSv4: hyper-connections, compressed attention, hash routing ----
        experimental_attention_variant="dsv4",  # __post_init__ sets dsv4_mode
        dsv4_hc_mult=int(hf["hc_mult"]),
        dsv4_hc_sinkhorn_iters=int(hf["hc_sinkhorn_iters"]),
        dsv4_hc_eps=float(hf["hc_eps"]),
        dsv4_compress_ratios=ratios,
        dsv4_compress_rope_theta=float(hf["compress_rope_theta"]),
        dsv4_o_groups=int(hf["o_groups"]),
        dsv4_o_lora_rank=int(hf["o_lora_rank"]),
        dsv4_n_hash_layers=int(hf["num_hash_layers"]),
        dsv4_window_size=int(hf["sliding_window"]),
        dsa_indexer_n_heads=int(hf["index_n_heads"]),
        dsa_indexer_head_dim=int(hf["index_head_dim"]),
        dsa_indexer_topk=int(hf["index_topk"]),
        # ---- parallelism / dtype / determinism ----
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        sequence_parallel=sp,
        variable_seq_lengths=(pp > 1),
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        deterministic_mode=deterministic,
        # Without a DDP wrapper ``param.main_grad`` is None and the fused
        # accumulation path dereferences it.
        gradient_accumulation_fusion=False,
        # Weights arrive from a torch_dist checkpoint, so the (slow) CPU init
        # the HF-safetensors path needs buys nothing here.
        use_cpu_initialization=False,
        **recompute,
    )


def build_dsv4_spec(config: Any, *, dsa_topk_backend: str = "torch") -> Any:
    """Per-layer spec for DSv4, from the plugin.

    ``get_dsv4_spec`` wants Megatron's global ``args`` only to read
    ``miles_dsa_topk_backend`` off it, so a stand-in namespace is enough.
    ``'torch'`` selects the tilelang-free DSA indexer.
    """
    from argparse import Namespace

    from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec

    return get_dsv4_spec(
        Namespace(miles_dsa_topk_backend=dsa_topk_backend), config, vp_stage=None
    )


def materialize_dsv4(model: torch.nn.Module) -> None:
    """Move a freshly built DSv4 model onto the current device.

    The hyper-connection parameters and the attention sinks are created with a
    bare ``torch.empty`` and no ``device=``, so they are born on CPU no matter
    what ``use_cpu_initialization`` says. Megatron's own training loop is
    rescued by the ``.cuda()`` inside ``megatron.training.get_model``; this
    engine never calls it.
    """
    model.cuda(torch.cuda.current_device())
    stranded = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if stranded:
        raise RuntimeError(f"DSv4 parameters still on CPU after .cuda(): {stranded[:5]}")


def load_dsv4_dist_checkpoint(model: torch.nn.Module, ckpt_dir: str) -> dict:
    """Load a **model-only** Megatron torch_dist checkpoint into ``model``.

    The engine's own ``load_dist_checkpoint`` is a resume path: it asks for
    ``{'model': ..., 'optimizer': ...}``, but the offline DSv4 converter writes
    ``save_checkpoint(1, model, None, None, 0)`` -- no optimizer. Same
    ``dist_checkpointing.load``, same automatic resharding, model key only.

    Returns a small report so the caller can log/assert on it.
    """
    import megatron.core.dist_checkpointing as dc

    if os.path.isdir(os.path.join(ckpt_dir, "release")):
        ckpt_dir = os.path.join(ckpt_dir, "release")

    loaded = dc.load({"model": model.sharded_state_dict()}, ckpt_dir)
    result = model.load_state_dict(loaded["model"], strict=False)

    # ``_extra_state`` is TE's fp8 bookkeeping; absent from a bf16 checkpoint
    # and irrelevant here.
    missing = [k for k in result.missing_keys if "_extra_state" not in k]
    unexpected = [k for k in result.unexpected_keys if "_extra_state" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"DSv4 dist-checkpoint load mismatch: missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    non_finite = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if non_finite:
        raise RuntimeError(f"DSv4 checkpoint has non-finite parameters: {non_finite[:5]}")

    return {
        "path": ckpt_dir,
        "num_params": sum(p.numel() for p in model.parameters()),
        "num_tensors": len(loaded["model"]),
    }


def keep_fp32_params(model: torch.nn.Module) -> list[str]:
    """Names of parameters the DSv4 plugin marks ``_keep_fp32``.

    The hyper-connection mixing weights are fp32 by construction; a blanket
    ``.to(bfloat16)`` over the module would silently halve their precision.
    """
    return [n for n, p in model.named_parameters() if getattr(p, "_keep_fp32", False)]
