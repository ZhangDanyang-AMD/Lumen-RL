"""Qwen3 <-> Megatron-Core GPTModel weight conversion.

Maps Hugging Face ``Qwen3ForCausalLM`` / ``Qwen3MoeForCausalLM`` weights to a
Megatron-Core ``GPTModel`` built with a local (non-TE) layer spec, and back
(for rollout weight sync).

Dense (Qwen3): supports TP/PP/EP.
MoE (Qwen3-30B-A3B): EP-aware loading/syncing. Each EP rank handles its
local subset of experts. GroupedMLP stores all local experts in fused
``weight1`` (gate+up) and ``weight2`` (down) tensors.

Megatron GQA ``linear_qkv`` layout is interleaved per KV group:
``[g0: q0..q_{r-1}, k0, v0, g1: ...]`` with ``r = num_heads / num_kv_groups``.
``linear_fc1`` is the fused SwiGLU ``[gate; up]`` (gate first).

``te=True`` targets the TransformerEngine layer spec, where the input/pre-mlp
RMSNorms are fused into the following linear (``linear_qkv.layer_norm_weight``
/ ``linear_fc1.layer_norm_weight``); ``te=False`` targets the local spec
(standalone ``input_layernorm`` / ``pre_mlp_layernorm``). This only affects
dense layernorm keys and composes cleanly with MoE expert keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    num_experts: int = 0
    moe_ffn: int = 0
    shared_expert_ffn: int = 0
    shared_expert_gate: bool = False


def _pp_layer_range(
    d: Qwen3Dims,
    pp_rank: int,
    pp_size: int,
    layers_per_pp_rank: Optional[list[int]],
) -> tuple[int, int]:
    """Return ``(global_layer_offset, num_local_layers)`` for this PP rank."""
    if pp_size <= 1:
        return 0, d.num_layers
    if layers_per_pp_rank is not None:
        assert len(layers_per_pp_rank) == pp_size
        offset = sum(layers_per_pp_rank[:pp_rank])
        return offset, layers_per_pp_rank[pp_rank]
    per_stage = d.num_layers // pp_size
    return pp_rank * per_stage, per_stage


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


def hf_to_megatron(
    hf: dict[str, torch.Tensor],
    d: Qwen3Dims,
    te: bool = False,
    ep_rank: int = 0,
    ep_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: Optional[list[int]] = None,
    use_grouped_mlp: bool = True,
) -> dict[str, torch.Tensor]:
    """Return a Megatron GPTModel state_dict (bf16 tensors on CPU).

    For MoE models (``d.num_experts > 0``), only the local experts for the
    given ``ep_rank`` are included in the returned state dict.

    With PP > 1, only the layers and embedding/output owned by this PP rank
    are included.  Megatron's decoder layers are numbered locally (0-based
    within the stage), while HF layers use the global index.

    ``te=True`` targets the TransformerEngine layer spec, where the input/pre-mlp
    RMSNorms are fused into the following linear (``linear_qkv.layer_norm_weight``
    / ``linear_fc1.layer_norm_weight``); ``te=False`` targets the local spec
    (standalone ``input_layernorm`` / ``pre_mlp_layernorm``). This only affects
    dense layernorm keys.

    ``use_grouped_mlp``: when True, fuse experts into GroupedMLP weight1/weight2;
    when False, use per-expert SequentialMLP local_experts.{i}.linear_fc1/fc2.
    """
    in_ln = "self_attention.linear_qkv.layer_norm_weight" if te else "input_layernorm.weight"
    mlp_ln = "mlp.linear_fc1.layer_norm_weight" if te else "pre_mlp_layernorm.weight"
    m: dict[str, torch.Tensor] = {}
    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    if is_first_pp:
        m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    if is_last_pp:
        m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
        # Tied-embedding checkpoints (tie_word_embeddings=true, e.g. Qwen3-0.6B/
        # 1.7B/4B) may omit ``lm_head.weight``; the output layer IS the
        # embedding. When GPTModel is built with
        # share_embeddings_and_output_weights=True at PP=1 it allocates no
        # ``output_layer.weight`` param, and load_state_dict(strict=False)
        # drops this key; at PP>1 the last stage owns a real (synced) copy.
        lm_head = hf.get("lm_head.weight")
        if lm_head is None:
            lm_head = hf["model.embed_tokens.weight"]
        m["output_layer.weight"] = lm_head

    is_moe = d.num_experts > 0
    if is_moe:
        num_local = d.num_experts // ep_size
        expert_offset = ep_rank * num_local

    layer_offset, num_local_layers = _pp_layer_range(
        d, pp_rank, pp_size, layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        hp = f"model.layers.{global_i}."
        mp = f"decoder.layers.{local_i}."

        # -- attention (shared between dense and MoE) --
        m[mp + in_ln] = hf[hp + "input_layernorm.weight"]
        m[mp + "self_attention.linear_qkv.weight"] = _hf_qkv_to_megatron(
            hf[hp + "self_attn.q_proj.weight"],
            hf[hp + "self_attn.k_proj.weight"],
            hf[hp + "self_attn.v_proj.weight"], d,
        )
        if hp + "self_attn.q_norm.weight" in hf:
            m[mp + "self_attention.q_layernorm.weight"] = hf[hp + "self_attn.q_norm.weight"]
            m[mp + "self_attention.k_layernorm.weight"] = hf[hp + "self_attn.k_norm.weight"]
        m[mp + "self_attention.linear_proj.weight"] = hf[hp + "self_attn.o_proj.weight"]
        m[mp + mlp_ln] = hf[hp + "post_attention_layernorm.weight"]

        if is_moe:
            # -- MoE: router --
            m[mp + "mlp.router.weight"] = hf[hp + "mlp.gate.weight"]

            # -- MoE: experts --
            if use_grouped_mlp:
                # GroupedMLP: fuse all local experts into weight1/weight2
                gate_ups = []
                downs = []
                for e in range(expert_offset, expert_offset + num_local):
                    gate = hf[hp + f"mlp.experts.{e}.gate_proj.weight"]
                    up = hf[hp + f"mlp.experts.{e}.up_proj.weight"]
                    down = hf[hp + f"mlp.experts.{e}.down_proj.weight"]
                    gate_ups.append(torch.cat([gate, up], dim=0).t())
                    downs.append(down.t())
                m[mp + "mlp.experts.weight1"] = torch.cat(gate_ups, dim=1).contiguous()
                m[mp + "mlp.experts.weight2"] = torch.cat(downs, dim=0).contiguous()
            else:
                # SequentialMLP: per-expert linear_fc1/linear_fc2
                for local_i_e, e in enumerate(range(expert_offset, expert_offset + num_local)):
                    gate = hf[hp + f"mlp.experts.{e}.gate_proj.weight"]
                    up = hf[hp + f"mlp.experts.{e}.up_proj.weight"]
                    down = hf[hp + f"mlp.experts.{e}.down_proj.weight"]
                    ep_prefix = mp + f"mlp.experts.local_experts.{local_i_e}."
                    m[ep_prefix + "linear_fc1.weight"] = torch.cat([gate, up], dim=0).contiguous()
                    m[ep_prefix + "linear_fc2.weight"] = down

            # -- MoE: shared expert (optional) --
            if d.shared_expert_ffn > 0:
                m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                    [hf[hp + "mlp.shared_expert.gate_proj.weight"],
                     hf[hp + "mlp.shared_expert.up_proj.weight"]], dim=0,
                ).contiguous()
                m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                    hp + "mlp.shared_expert.down_proj.weight"
                ]
                if d.shared_expert_gate:
                    m[mp + "mlp.shared_experts.gate_weight"] = hf[
                        hp + "mlp.shared_expert_gate.weight"
                    ]
        else:
            # -- Dense MLP --
            m[mp + "mlp.linear_fc1.weight"] = torch.cat(
                [hf[hp + "mlp.gate_proj.weight"], hf[hp + "mlp.up_proj.weight"]], dim=0
            ).contiguous()
            m[mp + "mlp.linear_fc2.weight"] = hf[hp + "mlp.down_proj.weight"]
    return m


def megatron_to_hf(
    named_params,
    d: Qwen3Dims,
    te: bool = False,
    ep_rank: int = 0,
    ep_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: Optional[list[int]] = None,
    use_grouped_mlp: bool = True,
):
    """Yield (hf_name, tensor) from Megatron GPTModel named params.

    For MoE models, yields only the local experts for this EP rank
    with their global expert indices.

    With PP > 1, maps local decoder layer indices back to global HF
    layer indices.

    ``te=True`` reads the fused TE layernorm names (see ``hf_to_megatron``).
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

    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    if is_first_pp:
        yield "model.embed_tokens.weight", get("embedding.word_embeddings.weight")
    if is_last_pp:
        yield "model.norm.weight", get("decoder.final_layernorm.weight")
        # With share_embeddings_and_output_weights=True (tied-embedding HF
        # configs) and PP=1, GPTModel never allocates ``output_layer.weight``
        # (skip_weight_param_allocation) -- the output projection reads the
        # word embedding at forward time. Emit ``lm_head.weight`` as that same
        # tensor: rollout backends either skip it when tied (vLLM) or alias
        # lm_head storage to embed_tokens (ATOM), so the value must match.
        out_w = md.get("output_layer.weight")
        if out_w is None:
            if not is_first_pp:
                raise KeyError(
                    "output_layer.weight missing on last PP stage and word "
                    "embeddings live on another stage; cannot emit lm_head.weight"
                )
            out_w = get("embedding.word_embeddings.weight")
        yield "lm_head.weight", out_w

    is_moe = d.num_experts > 0
    if is_moe:
        num_local = d.num_experts // ep_size
        expert_offset = ep_rank * num_local

    layer_offset, num_local_layers = _pp_layer_range(
        d, pp_rank, pp_size, layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        mp = f"decoder.layers.{local_i}."
        hp = f"model.layers.{global_i}."

        # -- attention --
        yield hp + "input_layernorm.weight", get(mp + in_ln)
        q, k, v = _megatron_qkv_to_hf(get(mp + "self_attention.linear_qkv.weight"), d)
        yield hp + "self_attn.q_proj.weight", q
        yield hp + "self_attn.k_proj.weight", k
        yield hp + "self_attn.v_proj.weight", v
        qln_key = mp + "self_attention.q_layernorm.weight"
        if qln_key in md:
            yield hp + "self_attn.q_norm.weight", get(qln_key)
            yield hp + "self_attn.k_norm.weight", get(mp + "self_attention.k_layernorm.weight")
        yield hp + "self_attn.o_proj.weight", get(mp + "self_attention.linear_proj.weight")
        yield hp + "post_attention_layernorm.weight", get(mp + mlp_ln)

        if is_moe:
            # -- MoE: router --
            yield hp + "mlp.gate.weight", get(mp + "mlp.router.weight")

            # -- MoE: experts --
            if use_grouped_mlp:
                w1 = get(mp + "mlp.experts.weight1")
                w2 = get(mp + "mlp.experts.weight2")
                chunk_fc1 = 2 * d.moe_ffn
                chunk_fc2 = d.moe_ffn
                for local_idx in range(num_local):
                    global_idx = expert_offset + local_idx
                    ep = hp + f"mlp.experts.{global_idx}."
                    w1_e = w1[:, local_idx * chunk_fc1:(local_idx + 1) * chunk_fc1].t()
                    gate_w = w1_e[:d.moe_ffn].contiguous()
                    up_w = w1_e[d.moe_ffn:].contiguous()
                    yield ep + "gate_proj.weight", gate_w
                    yield ep + "up_proj.weight", up_w
                    w2_e = w2[local_idx * chunk_fc2:(local_idx + 1) * chunk_fc2]
                    yield ep + "down_proj.weight", w2_e.t().contiguous()
            else:
                for local_idx in range(num_local):
                    global_idx = expert_offset + local_idx
                    ep = hp + f"mlp.experts.{global_idx}."
                    lp = mp + f"mlp.experts.local_experts.{local_idx}."
                    fc1 = get(lp + "linear_fc1.weight")
                    gate_w = fc1[:d.moe_ffn].contiguous()
                    up_w = fc1[d.moe_ffn:].contiguous()
                    yield ep + "gate_proj.weight", gate_w
                    yield ep + "up_proj.weight", up_w
                    yield ep + "down_proj.weight", get(lp + "linear_fc2.weight")

            # -- MoE: shared expert (optional) --
            if d.shared_expert_ffn > 0:
                sfc1 = get(mp + "mlp.shared_experts.linear_fc1.weight")
                s_gate, s_up = sfc1[:d.shared_expert_ffn], sfc1[d.shared_expert_ffn:]
                yield hp + "mlp.shared_expert.gate_proj.weight", s_gate.contiguous()
                yield hp + "mlp.shared_expert.up_proj.weight", s_up.contiguous()
                yield hp + "mlp.shared_expert.down_proj.weight", get(
                    mp + "mlp.shared_experts.linear_fc2.weight"
                )
                if d.shared_expert_gate:
                    yield hp + "mlp.shared_expert_gate.weight", get(
                        mp + "mlp.shared_experts.gate_weight"
                    )
        else:
            # -- Dense MLP --
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
