"""Qwen3-MoE <-> Megatron-Core GPTModel weight conversion.

Companion to ``qwen3_megatron_bridge`` (dense). Handles the Mixture-of-Experts
Qwen3 family (e.g. ``Qwen3-30B-A3B``): a routed-expert MLP replaces the dense
SwiGLU MLP, and the pre-MLP RMSNorm becomes a **standalone**
``decoder.layers.{L}.pre_mlp_layernorm.weight`` (unlike the dense TE spec, which
fuses it into ``mlp.linear_fc1.layer_norm_weight``). The attention block --
fused ``self_attention.linear_qkv`` (+ fused input RMSNorm), q/k layernorm,
``linear_proj`` -- is identical to the dense bridge.

Megatron expert layout (from ``get_gpt_decoder_block_spec``):
  * router:     ``decoder.layers.{L}.mlp.router.weight``          [E, hidden]
  * grouped:    ``...mlp.experts.linear_fc1.weight{e}``           [2*moe_ffn, hidden]
                ``...mlp.experts.linear_fc2.weight{e}``           [hidden, moe_ffn]
  * sequential: ``...mlp.experts.local_experts.{e}.linear_fc1.weight`` / ``...fc2.weight``
where ``linear_fc1`` is the fused SwiGLU ``[gate; up]`` and ``{e}`` is the
expert index **local to this EP rank** (each EP rank owns ``E / ep_size``
experts). HF stores per-expert ``gate_proj`` / ``up_proj`` / ``down_proj`` and a
router ``mlp.gate.weight``.

The engine converts full HF weights to a global-indexed Megatron dict here, then
slices per (TP, PP, EP, ETP) rank around these full-tensor conversions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from lumenrl.engine.training.qwen3_megatron_bridge import (
    Qwen3Dims,
    _hf_qkv_to_megatron,
    _megatron_qkv_to_hf,
)


@dataclass
class Qwen3MoEDims(Qwen3Dims):
    """Dense ``Qwen3Dims`` + MoE-specific sizes.

    ``ffn`` is unused for pure-routed Qwen3-MoE (no dense MLP); ``moe_ffn`` is the
    per-expert intermediate size (HF ``moe_intermediate_size``). ``shared_ffn`` > 0
    enables a shared expert (HF ``shared_expert_intermediate_size``); Qwen3-30B-A3B
    has none, so it defaults to 0.
    """

    num_experts: int = 128
    moe_ffn: int = 768
    shared_ffn: int = 0


# expert param keys (both grouped-gemm and sequential naming)
_EXP_GROUPED = re.compile(r"^(.*\.mlp\.experts\.linear_fc([12]))\.weight(\d+)$")
_EXP_SEQ = re.compile(r"^(.*\.mlp\.experts\.local_experts\.)(\d+)(\.linear_fc([12])\.weight)$")


def _expert_local_index(name: str) -> tuple[int, str] | None:
    """Return ``(local_expert_index, which_fc)`` for an expert param, else None.

    ``which_fc`` is ``"1"`` (fused gate;up) or ``"2"`` (down).
    """
    m = _EXP_GROUPED.match(name)
    if m:
        return int(m.group(3)), m.group(2)
    m = _EXP_SEQ.match(name)
    if m:
        return int(m.group(2)), m.group(4)
    return None


def _relabel_expert_index(name: str, new_e: int) -> str:
    """Rewrite an expert param name to use expert index ``new_e`` (local->global)."""
    m = _EXP_GROUPED.match(name)
    if m:
        return f"{m.group(1)}.weight{new_e}"
    m = _EXP_SEQ.match(name)
    if m:
        return f"{m.group(1)}{new_e}{m.group(3)}"
    return name


def _non_expert_hf_to_megatron(hf: dict, d: Qwen3MoEDims) -> dict[str, torch.Tensor]:
    """Global-named Megatron tensors for every NON-expert param (TE spec).

    Attention/embedding/output identical to the dense TE bridge; the MoE pre-MLP
    RMSNorm is emitted as the standalone ``pre_mlp_layernorm.weight`` and the
    router as ``mlp.router.weight``.
    """
    m: dict[str, torch.Tensor] = {}
    m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
    # tied-embedding checkpoints may omit lm_head.weight (see dense bridge)
    m["output_layer.weight"] = hf.get("lm_head.weight", hf["model.embed_tokens.weight"])
    for i in range(d.num_layers):
        hp = f"model.layers.{i}."
        mp = f"decoder.layers.{i}."
        m[mp + "self_attention.linear_qkv.layer_norm_weight"] = hf[hp + "input_layernorm.weight"]
        m[mp + "self_attention.linear_qkv.weight"] = _hf_qkv_to_megatron(
            hf[hp + "self_attn.q_proj.weight"],
            hf[hp + "self_attn.k_proj.weight"],
            hf[hp + "self_attn.v_proj.weight"], d,
        )
        m[mp + "self_attention.q_layernorm.weight"] = hf[hp + "self_attn.q_norm.weight"]
        m[mp + "self_attention.k_layernorm.weight"] = hf[hp + "self_attn.k_norm.weight"]
        m[mp + "self_attention.linear_proj.weight"] = hf[hp + "self_attn.o_proj.weight"]
        # MoE: standalone pre-mlp RMSNorm (NOT fused into a linear).
        m[mp + "pre_mlp_layernorm.weight"] = hf[hp + "post_attention_layernorm.weight"]
        m[mp + "mlp.router.weight"] = hf[hp + "mlp.gate.weight"]
        # optional shared expert (fused gate;up + down + gate score)
        if d.shared_ffn > 0 and (hp + "mlp.shared_expert.gate_proj.weight") in hf:
            m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                [hf[hp + "mlp.shared_expert.gate_proj.weight"],
                 hf[hp + "mlp.shared_expert.up_proj.weight"]], dim=0,
            ).contiguous()
            m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[hp + "mlp.shared_expert.down_proj.weight"]
            if (hp + "mlp.shared_expert_gate.weight") in hf:
                m[mp + "mlp.shared_experts.gate_weight"] = hf[hp + "mlp.shared_expert_gate.weight"]
    return m


def hf_expert_fc1(hf: dict, d: Qwen3MoEDims, layer: int, global_e: int) -> torch.Tensor:
    """Fused SwiGLU ``[gate; up]`` for one global expert -> [2*moe_ffn, hidden]."""
    hp = f"model.layers.{layer}.mlp.experts.{global_e}."
    return torch.cat([hf[hp + "gate_proj.weight"], hf[hp + "up_proj.weight"]], dim=0).contiguous()


def hf_expert_fc2(hf: dict, d: Qwen3MoEDims, layer: int, global_e: int) -> torch.Tensor:
    """Down projection for one global expert -> [hidden, moe_ffn]."""
    hp = f"model.layers.{layer}.mlp.experts.{global_e}."
    return hf[hp + "down_proj.weight"].contiguous()


def megatron_to_hf_moe(named_params, d: Qwen3MoEDims):
    """Yield ``(hf_name, tensor)`` from GLOBAL-indexed Megatron named params.

    ``named_params`` is an iterable of ``(megatron_name, tensor)`` where expert
    params use GLOBAL expert indices (grouped ``...experts.linear_fc{1,2}.weight{E}``
    or sequential ``...experts.local_experts.{E}...``) and layers use GLOBAL layer
    numbers. Any ``module.``/``module.module.`` prefix is stripped.
    """
    md: dict[str, torch.Tensor] = {}
    for name, t in named_params:
        for pre in ("module.module.", "module."):
            if name.startswith(pre):
                name = name[len(pre):]
                break
        md[name] = t

    layer_re = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")

    def emit_expert(layer, which_fc, global_e, param):
        if which_fc == "1":
            gate, up = param.chunk(2, dim=0)
            hp = f"model.layers.{layer}.mlp.experts.{global_e}."
            return [(hp + "gate_proj.weight", gate.contiguous()),
                    (hp + "up_proj.weight", up.contiguous())]
        return [(f"model.layers.{layer}.mlp.experts.{global_e}.down_proj.weight", param)]

    for name, t in md.items():
        if name == "embedding.word_embeddings.weight":
            yield "model.embed_tokens.weight", t
            continue
        if name == "decoder.final_layernorm.weight":
            yield "model.norm.weight", t
            continue
        if name == "output_layer.weight":
            yield "lm_head.weight", t
            continue
        mm = layer_re.match(name)
        if not mm:
            continue
        layer, rest = int(mm.group(1)), mm.group(2)

        exp = _expert_local_index(name)
        if exp is not None:
            local_e, which_fc = exp  # here "local" index == GLOBAL (engine relabels)
            for k, v in emit_expert(layer, which_fc, local_e, t):
                yield k, v
            continue

        hp = f"model.layers.{layer}."
        if rest == "self_attention.linear_qkv.layer_norm_weight":
            yield hp + "input_layernorm.weight", t
        elif rest == "self_attention.linear_qkv.weight":
            q, k, v = _megatron_qkv_to_hf(t, d)
            yield hp + "self_attn.q_proj.weight", q
            yield hp + "self_attn.k_proj.weight", k
            yield hp + "self_attn.v_proj.weight", v
        elif rest == "self_attention.q_layernorm.weight":
            yield hp + "self_attn.q_norm.weight", t
        elif rest == "self_attention.k_layernorm.weight":
            yield hp + "self_attn.k_norm.weight", t
        elif rest == "self_attention.linear_proj.weight":
            yield hp + "self_attn.o_proj.weight", t
        elif rest == "pre_mlp_layernorm.weight":
            yield hp + "post_attention_layernorm.weight", t
        elif rest == "mlp.router.weight":
            yield hp + "mlp.gate.weight", t
        elif rest == "mlp.shared_experts.linear_fc1.weight":
            gate, up = t.chunk(2, dim=0)
            yield hp + "mlp.shared_expert.gate_proj.weight", gate.contiguous()
            yield hp + "mlp.shared_expert.up_proj.weight", up.contiguous()
        elif rest == "mlp.shared_experts.linear_fc2.weight":
            yield hp + "mlp.shared_expert.down_proj.weight", t
        elif rest == "mlp.shared_experts.gate_weight":
            yield hp + "mlp.shared_expert_gate.weight", t


def build_moe_dims(hf_cfg: dict) -> Qwen3MoEDims:
    """Construct ``Qwen3MoEDims`` from a HF Qwen3-MoE ``config.json`` dict."""
    head_dim = hf_cfg.get("head_dim", hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"])
    num_experts = (
        hf_cfg.get("num_experts")
        or hf_cfg.get("n_routed_experts")
        or hf_cfg.get("num_local_experts")
    )
    moe_ffn = hf_cfg.get("moe_intermediate_size") or hf_cfg.get("intermediate_size")
    shared_ffn = int(hf_cfg.get("shared_expert_intermediate_size", 0) or 0)
    return Qwen3MoEDims(
        num_layers=hf_cfg["num_hidden_layers"], hidden=hf_cfg["hidden_size"],
        num_heads=hf_cfg["num_attention_heads"], num_kv_groups=hf_cfg["num_key_value_heads"],
        head_dim=head_dim, ffn=hf_cfg.get("intermediate_size", 0), vocab=hf_cfg["vocab_size"],
        num_experts=int(num_experts), moe_ffn=int(moe_ffn), shared_ffn=shared_ffn,
    )
