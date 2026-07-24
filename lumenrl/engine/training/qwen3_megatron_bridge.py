"""Qwen3 <-> Megatron-Core GPTModel weight conversion (TP=1).

Maps Hugging Face ``Qwen3ForCausalLM`` weights to the full Megatron-Core tensor
layout used by the TransformerEngine spec, and back for rollout weight sync.
The native engine applies TP/PP sharding around these full-tensor conversions.

Megatron GQA ``linear_qkv`` layout is interleaved per KV group:
``[g0: q0..q_{r-1}, k0, v0, g1: ...]`` with ``r = num_heads / num_kv_groups``.
``linear_fc1`` is the fused SwiGLU ``[gate; up]`` (gate first).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Qwen3Dims:
    num_layers: int = 36
    hidden: int = 4096
    num_heads: int = 32
    num_kv_groups: int = 8
    head_dim: int = 128
    ffn: int = 12288
    vocab: int = 151936


def _hf_qkv_to_megatron(q, k, v, d: Qwen3Dims) -> torch.Tensor:
    r = d.num_heads // d.num_kv_groups
    q = q.reshape(d.num_kv_groups, r, d.head_dim, d.hidden)
    k = k.reshape(d.num_kv_groups, 1, d.head_dim, d.hidden)
    v = v.reshape(d.num_kv_groups, 1, d.head_dim, d.hidden)
    qkv = torch.cat([q, k, v], dim=1)  # (groups, r+2, head_dim, hidden)
    return qkv.reshape(-1, d.hidden).contiguous()


def _megatron_qkv_to_hf(qkv, d: Qwen3Dims):
    r = d.num_heads // d.num_kv_groups
    qkv = qkv.reshape(d.num_kv_groups, r + 2, d.head_dim, d.hidden)
    q = qkv[:, :r].reshape(d.num_heads * d.head_dim, d.hidden).contiguous()
    k = qkv[:, r:r + 1].reshape(d.num_kv_groups * d.head_dim, d.hidden).contiguous()
    v = qkv[:, r + 1:r + 2].reshape(d.num_kv_groups * d.head_dim, d.hidden).contiguous()
    return q, k, v


def hf_to_megatron(hf: dict[str, torch.Tensor], d: Qwen3Dims, te: bool = False) -> dict[str, torch.Tensor]:
    """Return a Megatron GPTModel state_dict (bf16 tensors on CPU).

    ``te=True`` targets the TransformerEngine layer spec, where the input/pre-mlp
    RMSNorms are fused into the following linear (``linear_qkv.layer_norm_weight``
    / ``linear_fc1.layer_norm_weight``); ``te=False`` targets the local spec
    (standalone ``input_layernorm`` / ``pre_mlp_layernorm``). All other keys match.
    """
    in_ln = "self_attention.linear_qkv.layer_norm_weight" if te else "input_layernorm.weight"
    mlp_ln = "mlp.linear_fc1.layer_norm_weight" if te else "pre_mlp_layernorm.weight"
    m: dict[str, torch.Tensor] = {}
    m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
    m["output_layer.weight"] = hf["lm_head.weight"]
    for i in range(d.num_layers):
        hp = f"model.layers.{i}."
        mp = f"decoder.layers.{i}."
        m[mp + in_ln] = hf[hp + "input_layernorm.weight"]
        m[mp + "self_attention.linear_qkv.weight"] = _hf_qkv_to_megatron(
            hf[hp + "self_attn.q_proj.weight"],
            hf[hp + "self_attn.k_proj.weight"],
            hf[hp + "self_attn.v_proj.weight"], d,
        )
        m[mp + "self_attention.q_layernorm.weight"] = hf[hp + "self_attn.q_norm.weight"]
        m[mp + "self_attention.k_layernorm.weight"] = hf[hp + "self_attn.k_norm.weight"]
        m[mp + "self_attention.linear_proj.weight"] = hf[hp + "self_attn.o_proj.weight"]
        m[mp + mlp_ln] = hf[hp + "post_attention_layernorm.weight"]
        m[mp + "mlp.linear_fc1.weight"] = torch.cat(
            [hf[hp + "mlp.gate_proj.weight"], hf[hp + "mlp.up_proj.weight"]], dim=0
        ).contiguous()
        m[mp + "mlp.linear_fc2.weight"] = hf[hp + "mlp.down_proj.weight"]
    return m


def megatron_to_hf(named_params, d: Qwen3Dims, te: bool = False):
    """Yield (hf_name, tensor) from Megatron GPTModel named params (TP=1).

    ``named_params`` is an iterable of ``(megatron_name, tensor)``. Names may be
    prefixed (e.g. ``module.``); the prefix is stripped. ``te=True`` reads the
    fused TE layernorm names (see ``hf_to_megatron``).
    """
    in_ln = "self_attention.linear_qkv.layer_norm_weight" if te else "input_layernorm.weight"
    mlp_ln = "mlp.linear_fc1.layer_norm_weight" if te else "pre_mlp_layernorm.weight"
    md = {}
    for name, t in named_params:
        # strip DDP/Float16Module wrappers
        for pre in ("module.module.", "module."):
            if name.startswith(pre):
                name = name[len(pre):]
                break
        md[name] = t

    def get(n):
        return md[n]

    yield "model.embed_tokens.weight", get("embedding.word_embeddings.weight")
    yield "model.norm.weight", get("decoder.final_layernorm.weight")
    yield "lm_head.weight", get("output_layer.weight")
    for i in range(d.num_layers):
        mp = f"decoder.layers.{i}."
        hp = f"model.layers.{i}."
        yield hp + "input_layernorm.weight", get(mp + in_ln)
        q, k, v = _megatron_qkv_to_hf(get(mp + "self_attention.linear_qkv.weight"), d)
        yield hp + "self_attn.q_proj.weight", q
        yield hp + "self_attn.k_proj.weight", k
        yield hp + "self_attn.v_proj.weight", v
        yield hp + "self_attn.q_norm.weight", get(mp + "self_attention.q_layernorm.weight")
        yield hp + "self_attn.k_norm.weight", get(mp + "self_attention.k_layernorm.weight")
        yield hp + "self_attn.o_proj.weight", get(mp + "self_attention.linear_proj.weight")
        yield hp + "post_attention_layernorm.weight", get(mp + mlp_ln)
        fc1 = get(mp + "mlp.linear_fc1.weight")
        gate, up = fc1[: d.ffn], fc1[d.ffn:]
        yield hp + "mlp.gate_proj.weight", gate.contiguous()
        yield hp + "mlp.up_proj.weight", up.contiguous()
        yield hp + "mlp.down_proj.weight", get(mp + "mlp.linear_fc2.weight")


def load_hf_safetensors(model_dir: str) -> dict[str, torch.Tensor]:
    """Load a full HF state dict from a sharded safetensors directory."""
    import glob
    import os

    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    for f in files:
        state.update(load_file(f))
    return state
