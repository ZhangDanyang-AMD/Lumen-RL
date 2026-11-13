"""DeepSeek-V4-Flash <-> Megatron-Core GPTModel weight conversion.

Maps Hugging Face DeepSeek-V4-Flash weights to a Megatron-Core ``GPTModel``
built with the Lumen DSV4 layer spec (MLA attention, Hyper-Connection,
compressor/indexer, MoE), and back (for rollout weight sync to vLLM).

DSV4 uses Multi-Latent Attention (MLA) instead of standard GQA -- there is
no fused QKV projection.  Each attention projection (wq_a, wq_b, wkv, wo_a,
wo_b) is mapped individually.  Hyper-Connection parameters and compressor /
indexer weights are conditional on per-layer ``compress_ratios``.

MoE expert layout uses Lumen's GroupedMLP per-expert indexed format:
``linear_fc1.weight{E}`` (gate+up fused) and ``linear_fc2.weight{E}`` (down),
with router, expert bias, hash table (tid2eid), and shared expert.

The second half serves the native Megatron-Core engine, which builds DSv4 as an
``MLATransformerConfig`` with the DSv4 plugin's per-layer spec instead of the
Lumen layer spec:

``build_dsv4_config``
    HF ``config.json`` -> ``MLATransformerConfig``, every value derived from the
    HF config; the LumenRL-native twin of the CLI flags in miles'
    ``deepseek-v4-flash.sh``.
``build_dsv4_spec``
    The heterogeneous per-layer spec, from the DSv4 plugin (``miles_plugins``).
``load_dsv4_dist_checkpoint``
    Model-only ``dist_checkpointing.load`` of the torch_dist checkpoint converted
    offline, since DSv4 ships block-quantized FP8 the HF bridge cannot read.
``megatron_to_dsv4_native``
    The trainer's parameters under the checkpoint's own names, for the rollout.

The plugin and the DSv4-patched Megatron are not vendored here; both must be on
``sys.path`` (examples/DSv4_DAPO puts them there).
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from lumenrl.engine.training.bridges.core import pp_layer_range, strip_module_prefix
from lumenrl.engine.training.bridges.gpt import GPTDims

# Default compress_ratios for the 43-layer Flash model.
# Layers 0,1,42: ratio=0 (no compressor/indexer)
# Even layers 2,4,...,40: ratio=4 (CSA -- compressor + indexer)
# Odd layers 3,5,...,41: ratio=128 (HCA -- compressor only, no indexer)
DSV4_FLASH_COMPRESS_RATIOS: list[int] = [0, 0] + [4, 128] * 20 + [0]  # 43 values

def is_dsv4(hf: dict) -> bool:
    """True when ``config.json`` is a DeepSeek-V4 family (MLA + hyper-connections).

    ``MegatronNativeEngine.__init__`` asks this of every model it builds, so a
    Qwen3 run reaches it too: it is how the engine decides whether to take the
    DSv4 path at all.
    """
    model_type = str(hf.get("model_type") or "").lower()
    if "deepseek_v4" in model_type or "dsv4" in model_type:
        return True
    architectures = hf.get("architectures") or []
    return any("deepseekv4" in str(name).lower() for name in architectures)


@dataclass
class DSV4Dims(GPTDims):
    """GPTDims extended with DSV4-specific MLA / HC / compressor fields."""

    q_lora_rank: int = 1024
    kv_lora_rank: int = 512          # also serves as head_dim for MLA
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 512
    o_groups: int = 8
    o_lora_rank: int = 1024
    hc_mult: int = 4
    n_hash_layers: int = 3
    window_size: int = 128
    compress_ratios: list[int] = field(default_factory=lambda: list(DSV4_FLASH_COMPRESS_RATIOS))
    moe_topk: int = 6


def _normalize_hf_keys(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize HF weight keys to deepseek-ai format.

    RedHatAI/DeepSeek-V4-Flash-BF16 uses a simplified naming:
        layers.{L}.attn.*    -> model.layers.{L}.self_attn.*
        layers.{L}.attn_norm -> model.layers.{L}.input_layernorm
        layers.{L}.ffn_norm  -> model.layers.{L}.post_attention_layernorm
        embed.weight         -> model.embed_tokens.weight
        head.weight          -> lm_head.weight
        norm.weight          -> model.norm.weight
        hc_head_*            -> model.hc_head_*

    If keys already use the deepseek-ai format (model.layers.*), returns as-is.
    """
    if any(k.startswith("model.layers.") for k in hf):
        return hf  # already in deepseek-ai format

    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in hf.items():
        new_key = key
        if key == "embed.weight":
            new_key = "model.embed_tokens.weight"
        elif key == "head.weight":
            new_key = "lm_head.weight"
        elif key == "norm.weight":
            new_key = "model.norm.weight"
        elif key.startswith("hc_head_"):
            new_key = f"model.{key}"
        elif key.startswith("layers."):
            new_key = "model." + key
            # attn -> self_attn (but NOT attn_norm or attn_sink inside self_attn)
            new_key = re.sub(r"\.attn\.", ".self_attn.", new_key)
            new_key = new_key.replace(".attn_norm.", ".input_layernorm.")
            new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")
            # MoE: ffn.* -> mlp.*
            new_key = re.sub(r"\.ffn\.gate\.tid2eid", ".mlp.topk.tid2eid", new_key)
            new_key = re.sub(r"\.ffn\.gate\.bias", ".mlp.gate.e_score_correction_bias", new_key)
            new_key = re.sub(r"\.ffn\.gate\.weight", ".mlp.gate.weight", new_key)
            # Experts: ffn.experts.{E}.w1 -> mlp.experts.{E}.gate_proj
            #          ffn.experts.{E}.w2 -> mlp.experts.{E}.down_proj
            #          ffn.experts.{E}.w3 -> mlp.experts.{E}.up_proj
            new_key = re.sub(r"\.ffn\.experts\.(\d+)\.w1\.", r".mlp.experts.\1.gate_proj.", new_key)
            new_key = re.sub(r"\.ffn\.experts\.(\d+)\.w2\.", r".mlp.experts.\1.down_proj.", new_key)
            new_key = re.sub(r"\.ffn\.experts\.(\d+)\.w3\.", r".mlp.experts.\1.up_proj.", new_key)
            # Shared experts: ffn.shared_experts.w1 -> mlp.shared_experts.gate_proj
            new_key = new_key.replace(".ffn.shared_experts.w1.", ".mlp.shared_experts.gate_proj.")
            new_key = new_key.replace(".ffn.shared_experts.w2.", ".mlp.shared_experts.down_proj.")
            new_key = new_key.replace(".ffn.shared_experts.w3.", ".mlp.shared_experts.up_proj.")
        mapped[new_key] = tensor
    return mapped


def _expert_bias_or_zeros(
    hf: Mapping[str, torch.Tensor],
    layer_prefix: str,
    num_experts: int,
) -> torch.Tensor:
    bias_key = layer_prefix + "mlp.gate.e_score_correction_bias"
    if bias_key in hf:
        return hf[bias_key].float()
    router_weight = hf[layer_prefix + "mlp.gate.weight"]
    return torch.zeros(
        num_experts,
        dtype=torch.float32,
        device=router_weight.device,
    )


def denormalize_redhat_key(key: str) -> str:
    """Map a normalized DSV4 key back to the RedHat checkpoint/vLLM format."""
    if key == "model.embed_tokens.weight":
        return "embed.weight"
    if key == "lm_head.weight":
        return "head.weight"
    if key == "model.norm.weight":
        return "norm.weight"
    if key.startswith("model.hc_head_"):
        return key.removeprefix("model.")
    if not key.startswith("model.layers."):
        return key

    key = key.removeprefix("model.")
    key = key.replace(".input_layernorm.", ".attn_norm.")
    key = key.replace(".post_attention_layernorm.", ".ffn_norm.")
    key = key.replace(".self_attn.", ".attn.")
    key = key.replace(".mlp.topk.tid2eid", ".ffn.gate.tid2eid")
    key = key.replace(
        ".mlp.gate.e_score_correction_bias",
        ".ffn.gate.bias",
    )
    key = key.replace(".mlp.gate.weight", ".ffn.gate.weight")
    key = re.sub(
        r"\.mlp\.experts\.(\d+)\.gate_proj\.",
        r".ffn.experts.\1.w1.",
        key,
    )
    key = re.sub(
        r"\.mlp\.experts\.(\d+)\.down_proj\.",
        r".ffn.experts.\1.w2.",
        key,
    )
    key = re.sub(
        r"\.mlp\.experts\.(\d+)\.up_proj\.",
        r".ffn.experts.\1.w3.",
        key,
    )
    key = key.replace(".mlp.shared_experts.gate_proj.", ".ffn.shared_experts.w1.")
    key = key.replace(".mlp.shared_experts.down_proj.", ".ffn.shared_experts.w2.")
    key = key.replace(".mlp.shared_experts.up_proj.", ".ffn.shared_experts.w3.")
    return key


def hf_to_megatron(
    hf: dict[str, torch.Tensor],
    d: DSV4Dims,
    ep_rank: int = 0,
    ep_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: Optional[list[int]] = None,
    use_grouped_mlp: bool = True,  # ignored — kept for API compat
) -> dict[str, torch.Tensor]:
    """Return a Megatron GPTModel state_dict from HF DeepSeek-V4-Flash weights.

    For MoE models (``d.num_experts > 0``), only the local experts for the
    given ``ep_rank`` are included.

    With PP > 1, only the layers and embedding/output owned by this PP rank
    are included.  Megatron's decoder layers are numbered locally (0-based
    within the stage), while HF layers use the global index.

    FP32 parameters (HC params, attn_sink, compressor APE/norm) are preserved
    without casting to bf16.

    Supports both deepseek-ai (``model.layers.{L}.self_attn.*``) and RedHatAI
    (``layers.{L}.attn.*``) key naming via automatic normalization.
    """
    hf = _normalize_hf_keys(hf)
    m: dict[str, torch.Tensor] = {}
    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    # -- non-layer params --
    if is_first_pp:
        m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    if is_last_pp:
        m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
        m["output_layer.weight"] = hf["lm_head.weight"]

    # -- HC head params (FP32, on last PP rank) --
    if is_last_pp:
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            hf_key = f"model.{suffix}"
            meg_key = f"decoder.hc_head_params.{suffix}"
            if hf_key in hf:
                m[meg_key] = hf[hf_key].float()

    # -- MoE expert setup --
    is_moe = d.num_experts > 0
    if is_moe:
        num_local = d.num_experts // ep_size
        expert_offset = ep_rank * num_local

    layer_offset, num_local_layers = pp_layer_range(
        d, pp_rank, pp_size, layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        hp = f"model.layers.{global_i}."
        mp = f"decoder.layers.{local_i}."
        compress_ratio = d.compress_ratios[global_i] if d.compress_ratios else 0

        # -- input layernorm --
        m[mp + "input_layernorm.weight"] = hf[hp + "input_layernorm.weight"]

        # -- MLA attention projections (no fused QKV) --
        m[mp + "self_attention.wq_a.weight"] = hf[hp + "self_attn.wq_a.weight"]
        q_norm_w = hf[hp + "self_attn.q_norm.weight"]
        m[mp + "self_attention.q_norm.weight"] = q_norm_w
        m[mp + "self_attention.q_norm._norm.weight"] = q_norm_w
        m[mp + "self_attention.wq_b.weight"] = hf[hp + "self_attn.wq_b.weight"]
        m[mp + "self_attention.wkv.weight"] = hf[hp + "self_attn.wkv.weight"]
        kv_norm_w = hf[hp + "self_attn.kv_norm.weight"]
        m[mp + "self_attention.kv_norm.weight"] = kv_norm_w
        m[mp + "self_attention.kv_norm._norm.weight"] = kv_norm_w
        m[mp + "self_attention.wo_a.weight"] = hf[hp + "self_attn.wo_a.weight"]
        m[mp + "self_attention.wo_b.weight"] = hf[hp + "self_attn.wo_b.weight"]

        # -- attn_sink (FP32, TP-sharded dim0) --
        sink_key = hp + "self_attn.attn_sink"
        if sink_key in hf:
            m[mp + "self_attention.attn_sink"] = hf[sink_key].float()

        # -- pre-MLP layernorm --
        m[mp + "pre_mlp_layernorm.weight"] = hf[hp + "post_attention_layernorm.weight"]

        # -- Hyper-Connection per-layer params (FP32, duplicated) --
        for hc_suffix in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                          "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            hf_hc_key = hp + hc_suffix
            if hf_hc_key in hf:
                m[mp + hc_suffix] = hf[hf_hc_key].float()

        # -- Compressor params (only for layers with compress_ratio > 0) --
        if compress_ratio > 0:
            comp_hf = hp + "self_attn.compressor."
            comp_mg = mp + "self_attention.compressor."
            m[comp_mg + "ape"] = hf[comp_hf + "ape"].float()
            m[comp_mg + "wkv.weight"] = hf[comp_hf + "wkv.weight"]
            m[comp_mg + "wgate.weight"] = hf[comp_hf + "wgate.weight"]
            m[comp_mg + "norm.weight"] = hf[comp_hf + "norm.weight"].float()

        # -- Indexer params (only for compress_ratio == 4 layers) --
        if compress_ratio == 4:
            # Indexer linear params
            idx_hf = hp + "self_attn.indexer."
            idx_mg = mp + "self_attention.indexer."
            m[idx_mg + "linear_wq_b.weight"] = hf[idx_hf + "wq_b.weight"]
            m[idx_mg + "linear_weights_proj.weight"] = hf[idx_hf + "weights_proj.weight"]

            # Indexer's own compressor
            ic_hf = idx_hf + "compressor."
            ic_mg = idx_mg + "compressor."
            m[ic_mg + "ape"] = hf[ic_hf + "ape"].float()
            m[ic_mg + "wkv.weight"] = hf[ic_hf + "wkv.weight"]
            m[ic_mg + "wgate.weight"] = hf[ic_hf + "wgate.weight"]
            m[ic_mg + "norm.weight"] = hf[ic_hf + "norm.weight"].float()

        # -- MoE params --
        if is_moe:
            # Router
            m[mp + "mlp.router.weight"] = hf[hp + "mlp.gate.weight"]

            # Expert bias (e_score_correction_bias)
            m[mp + "mlp.router.expert_bias"] = _expert_bias_or_zeros(
                hf,
                hp,
                d.num_experts,
            )

            # Hash table (tid2eid) -- non-trainable
            tid2eid_key = hp + "mlp.topk.tid2eid"
            if tid2eid_key in hf:
                m[mp + "mlp.router.tid2eid"] = hf[tid2eid_key]

            # Experts — Lumen's GroupedMLP uses per-expert indexed format:
            #   mlp.experts.linear_fc1.weight{E}  (gate+up fused, [2*moe_ffn, hidden])
            #   mlp.experts.linear_fc2.weight{E}  (down, [hidden, moe_ffn])
            for local_e, e in enumerate(range(expert_offset, expert_offset + num_local)):
                gate = hf[hp + f"mlp.experts.{e}.gate_proj.weight"]
                up = hf[hp + f"mlp.experts.{e}.up_proj.weight"]
                down = hf[hp + f"mlp.experts.{e}.down_proj.weight"]
                m[mp + f"mlp.experts.linear_fc1.weight{local_e}"] = torch.cat(
                    [gate, up], dim=0,
                ).contiguous()
                m[mp + f"mlp.experts.linear_fc2.weight{local_e}"] = down

            # Shared expert
            if d.shared_expert_ffn > 0:
                shared_hp = hp + "mlp.shared_experts."
                if (shared_hp + "gate_proj.weight") in hf:
                    m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                        [hf[shared_hp + "gate_proj.weight"],
                         hf[shared_hp + "up_proj.weight"]], dim=0,
                    ).contiguous()
                    m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                        shared_hp + "down_proj.weight"
                    ]
                # Shared expert with separate naming (shared_expert vs shared_experts)
                shared_hp_alt = hp + "mlp.shared_expert."
                if (shared_hp_alt + "gate_proj.weight") in hf:
                    m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                        [hf[shared_hp_alt + "gate_proj.weight"],
                         hf[shared_hp_alt + "up_proj.weight"]], dim=0,
                    ).contiguous()
                    m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                        shared_hp_alt + "down_proj.weight"
                    ]
        else:
            # Dense MLP fallback
            m[mp + "mlp.linear_fc1.weight"] = torch.cat(
                [hf[hp + "mlp.gate_proj.weight"], hf[hp + "mlp.up_proj.weight"]], dim=0,
            ).contiguous()
            m[mp + "mlp.linear_fc2.weight"] = hf[hp + "mlp.down_proj.weight"]

    return m


def megatron_to_hf(
    named_params,
    d: DSV4Dims,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: Optional[list[int]] = None,
    use_grouped_mlp: bool = True,  # ignored — kept for API compat
):
    """Yield ``(hf_name, tensor)`` from Megatron GPTModel named params.

    Inverse of ``hf_to_megatron``.  Used for weight sync to vLLM.

    With PP > 1, maps local decoder layer indices back to global HF
    layer indices.
    """
    if isinstance(named_params, Mapping):
        md = named_params
    else:
        md: dict[str, torch.Tensor] = {}
        for name, t in named_params:
            md[strip_module_prefix(name)] = t

    def get(n):
        return md[n]

    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    # -- non-layer params --
    if is_first_pp:
        yield "model.embed_tokens.weight", get("embedding.word_embeddings.weight")
    if is_last_pp:
        yield "model.norm.weight", get("decoder.final_layernorm.weight")
        yield "lm_head.weight", get("output_layer.weight")

    # -- HC head params (FP32) --
    if is_last_pp:
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            meg_key = f"decoder.hc_head_params.{suffix}"
            if meg_key in md:
                yield f"model.{suffix}", get(meg_key)

    # -- MoE expert setup --
    is_moe = d.num_experts > 0
    if is_moe:
        num_local = d.num_experts  # after EP all-gather, we have all experts
        expert_offset = 0

    layer_offset, num_local_layers = pp_layer_range(
        d, pp_rank, pp_size, layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        mp = f"decoder.layers.{local_i}."
        hp = f"model.layers.{global_i}."
        compress_ratio = d.compress_ratios[global_i] if d.compress_ratios else 0

        # -- input layernorm --
        yield hp + "input_layernorm.weight", get(mp + "input_layernorm.weight")

        # -- MLA attention projections --
        yield hp + "self_attn.wq_a.weight", get(mp + "self_attention.wq_a.weight")
        yield hp + "self_attn.q_norm.weight", get(mp + "self_attention.q_norm.weight")
        yield hp + "self_attn.wq_b.weight", get(mp + "self_attention.wq_b.weight")
        yield hp + "self_attn.wkv.weight", get(mp + "self_attention.wkv.weight")
        yield hp + "self_attn.kv_norm.weight", get(mp + "self_attention.kv_norm.weight")
        yield hp + "self_attn.wo_a.weight", get(mp + "self_attention.wo_a.weight")
        yield hp + "self_attn.wo_b.weight", get(mp + "self_attention.wo_b.weight")

        # -- attn_sink (FP32) --
        sink_key = mp + "self_attention.attn_sink"
        if sink_key in md:
            yield hp + "self_attn.attn_sink", get(sink_key)

        # -- pre-MLP layernorm --
        yield hp + "post_attention_layernorm.weight", get(mp + "pre_mlp_layernorm.weight")

        # -- Hyper-Connection per-layer params (FP32) --
        for hc_suffix in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                          "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            meg_hc_key = mp + hc_suffix
            if meg_hc_key in md:
                yield hp + hc_suffix, get(meg_hc_key)

        # -- Compressor params --
        if compress_ratio > 0:
            comp_mg = mp + "self_attention.compressor."
            comp_hf = hp + "self_attn.compressor."
            if (comp_mg + "ape") in md:
                yield comp_hf + "ape", get(comp_mg + "ape")
                yield comp_hf + "wkv.weight", get(comp_mg + "wkv.weight")
                yield comp_hf + "wgate.weight", get(comp_mg + "wgate.weight")
                yield comp_hf + "norm.weight", get(comp_mg + "norm.weight")

        # -- Indexer params --
        if compress_ratio == 4:
            idx_mg = mp + "self_attention.indexer."
            idx_hf = hp + "self_attn.indexer."
            if (idx_mg + "linear_wq_b.weight") in md:
                yield idx_hf + "wq_b.weight", get(idx_mg + "linear_wq_b.weight")
                yield idx_hf + "weights_proj.weight", get(idx_mg + "linear_weights_proj.weight")

            # Indexer's own compressor
            ic_mg = idx_mg + "compressor."
            ic_hf = idx_hf + "compressor."
            if (ic_mg + "ape") in md:
                yield ic_hf + "ape", get(ic_mg + "ape")
                yield ic_hf + "wkv.weight", get(ic_mg + "wkv.weight")
                yield ic_hf + "wgate.weight", get(ic_mg + "wgate.weight")
                yield ic_hf + "norm.weight", get(ic_mg + "norm.weight")

        # -- MoE params --
        if is_moe:
            # Router
            yield hp + "mlp.gate.weight", get(mp + "mlp.router.weight")

            # Expert bias
            ebias_key = mp + "mlp.router.expert_bias"
            if ebias_key in md:
                yield hp + "mlp.gate.e_score_correction_bias", get(ebias_key)

            # Hash table (tid2eid)
            tid2eid_key = mp + "mlp.router.tid2eid"
            if tid2eid_key in md:
                yield hp + "mlp.topk.tid2eid", get(tid2eid_key)

            # Experts — Lumen's GroupedMLP per-expert indexed format:
            #   mlp.experts.linear_fc1.weight{E} -> gate_proj + up_proj
            #   mlp.experts.linear_fc2.weight{E} -> down_proj
            for local_idx in range(d.num_experts):
                ep = hp + f"mlp.experts.{local_idx}."
                fc1 = get(mp + f"mlp.experts.linear_fc1.weight{local_idx}")
                gate_w = fc1[:d.moe_ffn].contiguous()
                up_w = fc1[d.moe_ffn:].contiguous()
                yield ep + "gate_proj.weight", gate_w
                yield ep + "up_proj.weight", up_w
                yield ep + "down_proj.weight", get(
                    mp + f"mlp.experts.linear_fc2.weight{local_idx}"
                )

            # Shared expert
            if d.shared_expert_ffn > 0:
                sfc1_key = mp + "mlp.shared_experts.linear_fc1.weight"
                if sfc1_key in md:
                    sfc1 = get(sfc1_key)
                    s_gate, s_up = sfc1[:d.shared_expert_ffn], sfc1[d.shared_expert_ffn:]
                    yield hp + "mlp.shared_experts.gate_proj.weight", s_gate.contiguous()
                    yield hp + "mlp.shared_experts.up_proj.weight", s_up.contiguous()
                    yield hp + "mlp.shared_experts.down_proj.weight", get(
                        mp + "mlp.shared_experts.linear_fc2.weight"
                    )
        else:
            # Dense MLP fallback
            fc1 = get(mp + "mlp.linear_fc1.weight")
            gate, up = fc1[:d.ffn], fc1[d.ffn:]
            yield hp + "mlp.gate_proj.weight", gate.contiguous()
            yield hp + "mlp.up_proj.weight", up.contiguous()
            yield hp + "mlp.down_proj.weight", get(mp + "mlp.linear_fc2.weight")


# ---- native Megatron-Core engine: MLATransformerConfig and the DSv4 plugin ----

MODEL_TYPE = "deepseek_v4"

# YaRN + MLA defaults that DSv4 does not spell out in its HF config. miles'
# deepseek-v4-flash.sh passes them explicitly; Megatron's own defaults differ,
# so they cannot be left unset.
_KV_LORA_RANK = 512
_NCCL_ALGO_CHOICES = ("Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS")


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
    max_tokens_per_gpu: int = 0,
    deterministic: bool = True,
) -> Any:
    """Build the ``MLATransformerConfig`` for a DeepSeek-V4 model.

    Mirrors ``core_transformer_config_from_args`` applied to miles'
    ``deepseek-v4-flash.sh`` MODEL_ARGS. Anything that reads from ``hf`` is a
    genuine model property; anything constant is a DSv4 architectural fact that
    the HF config happens not to record (see ``_KV_LORA_RANK``) or a training
    policy LumenRL owns (dtype, parallel sizes, recompute).
    """
    from lumenrl.engine.training.megatron_base_engine import moe_dispatcher_kwargs

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
        # The checkpoint's expert bias is part of DSv4's routing policy and
        # reaches the rollout as a checkpoint-static tensor; Megatron's default
        # rate (1e-3) would move it every optimizer step.
        moe_router_bias_update_rate=float(ec.get("moe_router_bias_update_rate") or 0.0),
        moe_router_topk_scaling_factor=float(hf.get("routed_scaling_factor", 1.0)),
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=float(ec.get("moe_aux_loss_coeff", 0.0) or 0.0),
        # Plumbed because it was NOT, and its absence made an experiment silently
        # meaningless. The engine's own ``moe_router_dtype`` handling lives in the
        # ``elif self._is_moe`` branch that DSv4 never takes, so a run that asked
        # for fp32 routing got bf16 and looked like evidence that fp32 changes
        # nothing. Megatron warns about exactly this config ("a large number of
        # experts (>=32) without fp32 routing"), and with 256 experts scored in
        # bf16 a last-bit difference can flip a top-k choice. Default stays None
        # so this changes no existing run.
        moe_router_dtype=(str(ec["moe_router_dtype"])
                          if ec.get("moe_router_dtype") else None),
        **moe_dispatcher_kwargs(ec, tp=tp, cp=cp, sp=sp, max_tokens_per_gpu=max_tokens_per_gpu),
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


def sequence_alignment(config: Any) -> int:
    """Multiple that every forward's sequence length must be a multiple of.

    ``DeepseekV4Compressor.forward_raw`` asserts ``seqlen >= ratio`` and
    ``seqlen % ratio == 0`` (``ops/compressor.py:131``) on every compressed
    layer, so a forward whose length is not a multiple of the largest
    ``dsv4_compress_ratios`` entry dies in the assert. RL sequences are whatever
    the rollout produced, so the caller has to pad up to this.

    Padding goes at the *end*: attention is causal, the compressor groups by
    ``(p+1)//ratio`` and the indexer's span is a causal prefix, so no real
    position can see a trailing pad token.
    """
    ratios = getattr(config, "dsv4_compress_ratios", None) or [0]
    return max(max(int(r) for r in ratios), 1)


def keep_fp32_params(model: torch.nn.Module) -> list[str]:
    """Names of parameters the DSv4 plugin marks ``_keep_fp32``.

    The hyper-connection mixing weights are fp32 by construction; a blanket
    ``.to(bfloat16)`` over the module would silently halve their precision.
    """
    return [n for n, p in model.named_parameters() if getattr(p, "_keep_fp32", False)]


# ---------------------------------------------------------------------------
# Megatron -> DeepSeek-V4 checkpoint names, for rollout weight sync.
#
# The rollout side hands vLLM's own ``model.load_weights`` a list of
# ``(name, tensor)``, so these must be the names the *released* checkpoint uses:
# ``layers.0.attn.wq_a.weight``, ``layers.0.attn_norm.weight``, ``embed.weight``.
# Two other namings are nearby and both wrong here -- vLLM's internal module
# names (which fuse ``wq_a`` and ``wkv`` into ``fused_wqa_wkv``), and the
# DeepSeek-HF rewrite (``model.layers.0.self_attn.wq_a.weight``) that
# ``fp8_cast_bf16`` emits and SGLang's ``remap_weight_name_to_dpsk_hf_format``
# converts to. Sending the HF rewrite gets ``KeyError:
# 'layers.0.input_layernorm.weight'`` out of vLLM's loader.
# ---------------------------------------------------------------------------

_ATTN_SAME = (
    "attn_sink", "q_norm.weight", "kv_norm.weight",
    "wq_a.weight", "wq_b.weight", "wkv.weight", "wo_a.weight", "wo_b.weight",
    "compressor.ape", "compressor.norm.weight",
    "compressor.wgate.weight", "compressor.wkv.weight",
    "indexer.compressor.ape", "indexer.compressor.norm.weight",
    "indexer.compressor.wgate.weight", "indexer.compressor.wkv.weight",
)
# Megatron wraps the indexer's two projections in a ``linear_`` prefix.
_ATTN_RENAMED = {
    "indexer.linear_wq_b.weight": "indexer.wq_b.weight",
    "indexer.linear_weights_proj.weight": "indexer.weights_proj.weight",
}
_TOP_LEVEL = {
    "embedding.word_embeddings.weight": "embed.weight",
    "output_layer.weight": "head.weight",
    "decoder.final_layernorm.weight": "norm.weight",
    "decoder.hc_head_params.hc_head_fn": "hc_head_fn",
    "decoder.hc_head_params.hc_head_base": "hc_head_base",
    "decoder.hc_head_params.hc_head_scale": "hc_head_scale",
}
_LAYER_SAME_PREFIXES = ("hc_attn_", "hc_ffn_")

_MCORE_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.(.*)$")
_GROUPED_EXPERT = re.compile(r"^mlp\.experts\.linear_fc([12])\.weight(\d+)$")


def megatron_to_dsv4_native(named_params):
    """Yield ``(checkpoint_name, tensor)`` from GLOBAL-indexed Megatron params.

    ``named_params`` is ``(megatron_name, tensor)`` with global layer numbers and
    global expert indices, i.e. what ``_full_megatron_named_params_moe`` returns
    after its TP/EP gather. Any ``module.`` wrapper prefix is stripped.

    The one structural change is ``linear_fc1``: Megatron keeps SwiGLU's gate and
    up projections fused along dim 0, the checkpoint keeps them apart as ``w1``
    (gate) and ``w3`` (up), with ``w2`` for down.
    """
    for name, t in named_params:
        for pre in ("module.module.", "module."):
            if name.startswith(pre):
                name = name[len(pre):]

        if name in _TOP_LEVEL:
            yield _TOP_LEVEL[name], t
            continue

        m = _MCORE_LAYER.match(name)
        if m is None:
            raise KeyError(f"no DSv4 checkpoint name for megatron param {name!r}")
        layer, rest = m.group(1), m.group(2)
        pfx = f"layers.{layer}"

        if rest.startswith(_LAYER_SAME_PREFIXES):
            yield f"{pfx}.{rest}", t
        elif rest == "input_layernorm.weight":
            yield f"{pfx}.attn_norm.weight", t
        elif rest == "pre_mlp_layernorm.weight":
            yield f"{pfx}.ffn_norm.weight", t
        elif rest.startswith("self_attention."):
            sub = rest[len("self_attention."):]
            sub = _ATTN_RENAMED.get(sub, sub)
            if sub not in _ATTN_SAME and sub not in _ATTN_RENAMED.values():
                raise KeyError(f"no DSv4 checkpoint name for attention param {name!r}")
            yield f"{pfx}.attn.{sub}", t
        elif rest == "mlp.router.weight":
            yield f"{pfx}.ffn.gate.weight", t
        elif rest == "mlp.router.tid2eid":
            # Hash routing: a token-id -> expert-id table, not a learned weight.
            yield f"{pfx}.ffn.gate.tid2eid", t
        elif rest == "mlp.router.expert_bias":
            yield f"{pfx}.ffn.gate.bias", t
        elif rest == "mlp.shared_experts.linear_fc1.weight":
            gate, up = t.chunk(2, dim=0)
            yield f"{pfx}.ffn.shared_experts.w1.weight", gate
            yield f"{pfx}.ffn.shared_experts.w3.weight", up
        elif rest == "mlp.shared_experts.linear_fc2.weight":
            yield f"{pfx}.ffn.shared_experts.w2.weight", t
        else:
            em = _GROUPED_EXPERT.match(rest)
            if em is None:
                raise KeyError(f"no DSv4 checkpoint name for megatron param {name!r}")
            which_fc, e = em.group(1), int(em.group(2))
            base = f"{pfx}.ffn.experts.{e}"
            if which_fc == "1":
                gate, up = t.chunk(2, dim=0)
                yield f"{base}.w1.weight", gate
                yield f"{base}.w3.weight", up
            else:
                yield f"{base}.w2.weight", t
