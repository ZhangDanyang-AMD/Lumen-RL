"""The registered :class:`ModelSpec` entries.

Kept separate from ``model_registry`` so the registry stays free of bridge
imports and can be unit-tested on its own. Import order in this file is the
resolution order: DSv4 first (most specific), then MoE, then the dense
catch-all.
"""

from __future__ import annotations

from typing import Any, Mapping

from lumenrl.engine.training import dsv4_megatron_bridge as dsv4
from lumenrl.engine.training.model_registry import (
    MODEL_REGISTRY,
    ModelCaps,
    ModelSpec,
    hf_num_experts,
    resolve_head_dim,
)
from lumenrl.engine.training.qwen3_megatron_bridge import Qwen3Dims
from lumenrl.engine.training.qwen3moe_megatron_bridge import build_moe_dims


def _effective_num_experts(hf: Mapping[str, Any], ec: Mapping[str, Any]) -> int:
    """Expert count after the engine_config override.

    An explicit ``num_experts`` in engine_config wins over the HF config, so a
    dense checkpoint can be driven down the MoE path and vice versa.
    """
    return int(ec.get("num_experts") or hf_num_experts(hf) or 0)


def _dense_dims(hf: Mapping[str, Any]) -> Qwen3Dims:
    return Qwen3Dims(
        num_layers=hf["num_hidden_layers"],
        hidden=hf["hidden_size"],
        num_heads=hf["num_attention_heads"],
        num_kv_groups=hf["num_key_value_heads"],
        head_dim=resolve_head_dim(hf),
        ffn=hf["intermediate_size"],
        vocab=hf["vocab_size"],
    )


# --- DeepSeek-V4 -------------------------------------------------------------
# MLA head geometry, a 4-D hyper-connection residual stream, per-layer
# heterogeneous attention and hash routing on the first layers -- none of which
# the generic TransformerConfig path can describe. Its block-quantized FP8
# weights are also unreadable by the HF-safetensors bridge, so it carries no
# dims: ``build_dims`` is None rather than a stub that would fail later.
DSV4 = MODEL_REGISTRY.register(
    ModelSpec(
        name="deepseek_v4",
        detect=lambda hf, ec: dsv4.is_dsv4(hf),
        caps=ModelCaps(
            has_experts=True,
            supports_hf_bridge=False,
            supports_dynamic_batch=False,
        ),
        build_dims=None,
    )
)


# --- Qwen3-MoE (and any config declaring routed experts) ---------------------
# Qwen3-MoE routing = softmax(all) -> top-k -> renormalize top-k (HF
# ``norm_topk_prob=True``), which is mathematically identical to Megatron's
# ``moe_router_pre_softmax=False``: top-k of logits then a softmax over only the
# top-k, already summing to 1, because the full-softmax denominator cancels under
# renormalization. ``pre_softmax=True`` would leave gate weights un-renormalized
# (sum<1) and diverge from vLLM -> large rollout/train log-prob mismatch.
QWEN3_MOE = MODEL_REGISTRY.register(
    ModelSpec(
        name="qwen3_moe",
        detect=lambda hf, ec: _effective_num_experts(hf, ec) > 1,
        caps=ModelCaps(has_experts=True),
        build_dims=build_moe_dims,
        routing_defaults={"moe_router_pre_softmax": False},
    )
)


# --- Dense catch-all ---------------------------------------------------------
# Standard GQA + SwiGLU decoder. Last so it only sees configs nothing else
# claimed; resolve() raises if this entry is ever removed.
QWEN3_DENSE = MODEL_REGISTRY.register(
    ModelSpec(
        name="qwen3_dense",
        detect=lambda hf, ec: True,
        caps=ModelCaps(has_experts=False),
        build_dims=_dense_dims,
    )
)
