"""The registered :class:`ModelSpec` entries.

Kept separate from ``model_registry`` so the registry stays free of bridge
imports and can be unit-tested on its own. Import order in this file is the
resolution order, most specific first: DSv4, then the DeepSeek-V3 family,
then any other config declaring routed experts, then the dense catch-all.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping

from lumenrl.engine.training import dsv3_megatron_bridge as dsv3
from lumenrl.engine.training import dsv4_megatron_bridge as dsv4
from lumenrl.engine.training.model_registry import (
    MODEL_REGISTRY,
    ModelCaps,
    ModelSpec,
    hf_num_experts,
    resolve_head_dim,
)
from lumenrl.engine.training.qwen3_megatron_bridge import Qwen3Dims, megatron_to_hf
from lumenrl.engine.training.qwen3moe_megatron_bridge import (
    build_moe_dims,
    megatron_to_hf_moe,
)


def _effective_num_experts(hf: Mapping[str, Any], ec: Mapping[str, Any]) -> int:
    """Expert count after the engine_config override.

    An explicit ``num_experts`` in engine_config wins over the HF config, so a
    dense checkpoint can be driven down the MoE path and vice versa.
    """
    return int(ec.get("num_experts") or hf_num_experts(hf) or 0)


def _export_dsv4(engine):
    """DSv4 gathers like Qwen3-MoE but must land on the *checkpoint* names.

    The rollout side feeds these straight to vLLM's own ``load_weights``, so the
    renaming, not the gather, is what differs.
    """
    named = itertools.chain(
        engine._full_megatron_named_params_moe(),
        engine._dsv4_router_bias_buffers(),
    )
    return dsv4.megatron_to_dsv4_native(named)


def _export_moe(engine):
    return megatron_to_hf_moe(engine._full_megatron_named_params_moe(), engine._dims)


def _export_dense(engine):
    return megatron_to_hf(engine._full_megatron_named_params(), engine._dims, te=True)


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
            builds_own_config=True,
            # Its topology check must run before every forward, and the generic
            # non-pipelined shortcut cannot host it.
            requires_pipeline_forward=True,
            packed_stream_is_single_sequence=True,
        ),
        build_dims=None,
        # Resolved at call time, not import time: the DSv4 construction functions
        # ship with the DSv4 branch (``dev/vllm-fsdp-dapo`` / ``dev/OPD``), while
        # ``main`` carries only the call sites. Binding them eagerly here would make
        # importing this module fail on ``main`` for every model, so the indirection
        # is deliberate and should stay even after the branches converge.
        #
        # build_config: field-for-field equal to what Megatron's own parser produces
        #   from miles' deepseek-v4-flash.sh, the config every existing DSv4
        #   numerical reference was measured on.
        # build_layer_spec: heterogeneous per layer (sliding / compressed+indexed /
        #   hyper-compressed), so no block-spec builder can produce it.
        build_config=lambda *a, **kw: dsv4.build_dsv4_config(*a, **kw),
        build_layer_spec=lambda *a, **kw: dsv4.build_dsv4_spec(*a, **kw),
        sequence_alignment=lambda tfcfg: dsv4.sequence_alignment(tfcfg),
        export_weights=_export_dsv4,
    )
)


# --- DeepSeek-V3 family (V3 / V3.1 / R1, and the Kimi K2 line) ---------------
# Registered before qwen3_moe: a DSv3 config declares routed experts, so the
# generic MoE entry would otherwise claim it and build a plain TransformerConfig
# with fused QKV -- wrong for MLA. Detection is on ``architectures`` so DSv4,
# which also has MLA fields, is not caught here (it is matched earlier anyway).
DSV3 = MODEL_REGISTRY.register(
    ModelSpec(
        name="deepseek_v3",
        detect=lambda hf, ec: dsv3.is_dsv3(hf),
        caps=ModelCaps(
            has_experts=True,
            # Construction needs MLATransformerConfig, not TransformerConfig.
            builds_own_config=True,
            # No weight bridge yet -- see export_weights below. The HF-bridge
            # capability stays True because the intent is to use it once written;
            # what is missing is the MLA mapping, not the mechanism.
            supports_hf_bridge=True,
        ),
        build_dims=dsv3.build_dsv3_dims,
        build_config=dsv3.build_dsv3_config,
        # MLA is a parameter of the stock TE builder, so no custom layer spec:
        # the engine's generic MoE branch produces the right thing once
        # ``multi_latent_attention`` is set on the config.
        build_layer_spec=None,
        routing_defaults={"moe_router_pre_softmax": False},
        export_weights=dsv3.export_weights_not_implemented,
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
        export_weights=_export_moe,
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
        export_weights=_export_dense,
    )
)
