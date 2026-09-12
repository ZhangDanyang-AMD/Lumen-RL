"""DeepSeek-V3 family support for the Megatron training path.

Covers every checkpoint that reports ``DeepseekV3ForCausalLM`` -- DeepSeek V3 /
V3.1 / R1 and the Kimi K2 line, which reuses the same architecture class.

What makes this family different from Qwen3-MoE is **MLA**: queries and
key/values go through low-rank projections rather than a single fused QKV, so
the attention block has a different parameter set and Megatron needs
``MLATransformerConfig`` instead of ``TransformerConfig``. Megatron-core 0.18.2
implements MLA natively (``MLASelfAttention``, and ``multi_latent_attention``
is a parameter of the TE layer-spec builder), so this is configuration rather
than new modules.

Routing also differs: DeepSeek uses a sigmoid score function with the
``noaux_tc`` bias-corrected top-k and a routed scaling factor, where Qwen3-MoE
uses softmax with no bias.

The weight bridge below was written against a constructed model's real
parameter names rather than from the HF/Megatron docs, because two mappings
are not guessable: the LoRA layernorms are fused into the up-projections, and
the post-attention norm is spelled differently on dense and MoE layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from lumenrl.engine.training.qwen3_megatron_bridge import Qwen3Dims

__all__ = [
    "DSV3Dims",
    "is_dsv3",
    "build_dsv3_dims",
    "build_dsv3_config",
    "megatron_to_hf_dsv3",
    "hf_to_dsv3_megatron",
    "dsv3_router_bias_buffers",
]

_ARCHITECTURES = {"DeepseekV3ForCausalLM"}


def is_dsv3(hf: Mapping[str, Any]) -> bool:
    """True for DeepSeek-V3-class checkpoints, including the Kimi K2 line.

    Keyed on ``architectures`` rather than on the presence of ``kv_lora_rank``,
    because DSv4 also has MLA fields and must not be claimed by this entry.
    """
    arch = hf.get("architectures") or []
    return bool(set(arch) & _ARCHITECTURES)


@dataclass
class DSV3Dims(Qwen3Dims):
    """Qwen3Dims extended with the MLA and DeepSeek-MoE geometry.

    Follows ``DSV4Dims`` in extending the common dataclass so the shared
    non-attention parts of a bridge stay reusable.
    """

    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    moe_topk: int = 8
    n_shared_experts: int = 1
    first_k_dense_replace: int = 1
    routed_scaling_factor: float = 1.0


def build_dsv3_dims(hf: Mapping[str, Any]) -> DSV3Dims:
    n_shared = int(hf.get("n_shared_experts") or 0)
    moe_ffn = int(hf.get("moe_intermediate_size") or 0)
    return DSV3Dims(
        num_layers=int(hf["num_hidden_layers"]),
        hidden=int(hf["hidden_size"]),
        num_heads=int(hf["num_attention_heads"]),
        # MLA has no grouped-query sharing: every head carries its own latent.
        num_kv_groups=int(hf.get("num_key_value_heads") or hf["num_attention_heads"]),
        head_dim=int(hf.get("qk_nope_head_dim") or 128),
        ffn=int(hf["intermediate_size"]),
        vocab=int(hf["vocab_size"]),
        num_experts=int(hf.get("n_routed_experts") or 0),
        moe_ffn=moe_ffn,
        shared_expert_ffn=moe_ffn * n_shared,
        q_lora_rank=int(hf.get("q_lora_rank") or 0),
        kv_lora_rank=int(hf["kv_lora_rank"]),
        qk_nope_head_dim=int(hf["qk_nope_head_dim"]),
        qk_rope_head_dim=int(hf["qk_rope_head_dim"]),
        v_head_dim=int(hf["v_head_dim"]),
        moe_topk=int(hf.get("num_experts_per_tok") or 8),
        n_shared_experts=n_shared,
        first_k_dense_replace=int(hf.get("first_k_dense_replace") or 0),
        routed_scaling_factor=float(hf.get("routed_scaling_factor") or 1.0),
    )


def _moe_layer_freq(hf: Mapping[str, Any]) -> list[int]:
    """Per-layer dense/MoE pattern.

    DeepSeek keeps the first ``first_k_dense_replace`` layers dense. Megatron
    accepts an explicit per-layer list, which is exactly this.
    """
    n = int(hf["num_hidden_layers"])
    dense = int(hf.get("first_k_dense_replace") or 0)
    return [0] * dense + [1] * (n - dense)


def build_dsv3_config(
    hf: Mapping[str, Any],
    ec: Mapping[str, Any],
    *,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    ep: int = 1,
    etp: Optional[int] = None,
    sp: bool = False,
    deterministic: Optional[bool] = None,
    max_tokens_per_gpu: int = 0,
) -> Any:
    """Build the ``MLATransformerConfig`` for a DeepSeek-V3-class model.

    Mirrors the generic path in ``megatron_native_engine`` field for field --
    dtypes, variable_seq_lengths, the token dispatcher and recompute -- so a DSv3
    run keeps everything a Qwen3-MoE run gets, and only the MLA and routing
    differences come from here.
    """
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    del deterministic  # accepted for hook-signature parity; DSv3 has no such mode
    del max_tokens_per_gpu  # only the flex/MORI dispatcher sizes a heap from it
    etp = etp or tp

    recompute: dict[str, Any] = {}
    if ec.get("recompute_granularity"):
        recompute["recompute_granularity"] = ec["recompute_granularity"]
        recompute["recompute_method"] = ec.get("recompute_method") or "uniform"
        recompute["recompute_num_layers"] = int(ec.get("recompute_num_layers") or 1)

    num_experts = int(ec.get("num_experts") or hf.get("n_routed_experts") or 0)
    moe_ffn = int(ec.get("moe_ffn_hidden_size") or hf.get("moe_intermediate_size") or 0)
    n_shared = int(hf.get("n_shared_experts") or 0)

    # --- RoPE ---------------------------------------------------------------
    # MLATransformerConfig's defaults happen to BE DeepSeek-V3's YaRN values
    # (yarn / factor 40 / beta 32,1 / theta 10000), so relying on them looks
    # correct on V3 and silently mis-positions every token on anything else:
    # Kimi K2 uses theta 50000, factor 32, beta_fast 1.
    #
    # ``mscale_all_dim`` is wrong even for V3. Megatron defaults it to 0.0 while
    # every shipped DeepSeek config says 1.0, and MLASelfAttention derives
    # ``softmax_scale = _yarn_get_mscale(factor, mscale_all_dim)**2 / sqrt(qk_dim)``
    # from it -- unconditionally, whatever rope_type is. At factor 40 that is
    # 1.3689**2 vs 1.0, so the attention temperature is off by ~1.87x.
    rope_scaling = dict(hf.get("rope_scaling") or {})
    rope: dict[str, Any] = {"rotary_base": float(hf.get("rope_theta") or 10000.0)}
    if rope_scaling:
        kind = str(rope_scaling.get("type") or rope_scaling.get("rope_type") or "yarn")
        if kind != "yarn":
            raise ValueError(
                f"DSv3 rope_scaling type {kind!r} is not supported; MLA offers "
                "'rope' and 'yarn' only"
            )
        rope.update(
            rope_type="yarn",
            rotary_scaling_factor=float(rope_scaling["factor"]),
            original_max_position_embeddings=int(
                rope_scaling.get("original_max_position_embeddings", 4096)
            ),
            beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
            beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
            mscale=float(rope_scaling.get("mscale", 1.0)),
            mscale_all_dim=float(rope_scaling.get("mscale_all_dim", 0.0)),
        )
    else:
        # No scaling: plain RoPE, and a unit scaling factor so the softmax_scale
        # above collapses to the standard 1/sqrt(qk_dim).
        rope.update(rope_type="rope", rotary_scaling_factor=1.0, mscale_all_dim=0.0)

    moe: dict[str, Any] = {}
    if num_experts > 1:
        moe = dict(
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=moe_ffn,
            moe_router_topk=int(ec.get("moe_router_topk") or hf.get("num_experts_per_tok") or 8),
            moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", True)),
            moe_router_load_balancing_type=str(
                ec.get("moe_router_load_balancing_type", "aux_loss")
            ),
            moe_aux_loss_coeff=float(ec.get("moe_aux_loss_coeff", 0.0) or 0.0),
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            moe_permute_fusion=bool(ec.get("moe_permute_fusion", False)),
            moe_layer_freq=_moe_layer_freq(hf),
            # DeepSeek scores experts with sigmoid and selects top-k with the
            # ``noaux_tc`` bias correction, then rescales. Qwen3-MoE's softmax
            # default would change which experts are selected.
            moe_router_score_function=str(
                ec.get("moe_router_score_function") or hf.get("scoring_func") or "sigmoid"
            ),
            moe_router_enable_expert_bias=(
                str(hf.get("topk_method") or "noaux_tc") == "noaux_tc"
            ),
            moe_router_topk_scaling_factor=float(
                ec.get("moe_router_topk_scaling_factor")
                or hf.get("routed_scaling_factor")
                or 1.0
            ),
            # top-k over logits then softmax over the top-k, matching
            # ``norm_topk_prob`` -- see the Qwen3-MoE note in ``model_specs``.
            moe_router_pre_softmax=bool(ec.get("moe_router_pre_softmax") or False),
        )
        # Two knobs the generic MoE path in ``megatron_native_engine`` honours.
        # Building the config here means anything not forwarded is silently
        # dropped, and both of these matter more for DeepSeek than for Qwen3-MoE:
        #
        #   moe_router_dtype -- an fp32 router keeps top-k selection stable, which
        #     is the whole point of the knob (the Qwen3-MoE configs set fp32 for
        #     "lower train/rollout mismatch"). DeepSeek picks experts through a
        #     sigmoid plus the noaux_tc bias, so a bf16 router flipping a marginal
        #     expert changes which experts run and widens the rollout gap.
        #   moe_router_bias_update_rate -- the update rate for that noaux_tc bias.
        #     It only does anything when ``moe_router_enable_expert_bias`` is set,
        #     which this path sets and the Qwen3 path does not.
        if ec.get("moe_router_dtype"):
            moe["moe_router_dtype"] = str(ec["moe_router_dtype"])
        if ec.get("moe_router_bias_update_rate") is not None:
            moe["moe_router_bias_update_rate"] = float(ec["moe_router_bias_update_rate"])
        # DeepSeek restricts top-k to the best ``topk_group`` of ``n_group``
        # expert groups (node-limited routing). V3/R1 ship 8/4; without this the
        # router is free to pick across all groups and selects experts the
        # reference never would. K2's 1/1 is a no-op, and Megatron rejects
        # num_groups=1, so only wire it up when it actually constrains.
        n_group = int(ec.get("moe_router_num_groups") or hf.get("n_group") or 0)
        if n_group > 1:
            moe["moe_router_num_groups"] = n_group
            moe["moe_router_group_topk"] = int(
                ec.get("moe_router_group_topk") or hf.get("topk_group") or n_group
            )
        # Megatron's sigmoid router always renormalises the top-k probabilities
        # (moe_utils: ``probs = scores / scores.sum()`` when topk > 1), which is
        # norm_topk_prob=True. There is no switch for the other case, so refuse
        # it rather than train with silently rescaled routing weights.
        if not bool(hf.get("norm_topk_prob", True)):
            raise ValueError(
                "norm_topk_prob=False is not supported: Megatron's sigmoid router "
                "renormalises the top-k probabilities unconditionally"
            )
        if n_shared > 0:
            moe["moe_shared_expert_intermediate_size"] = moe_ffn * n_shared

    return MLATransformerConfig(
        num_layers=int(hf["num_hidden_layers"]),
        hidden_size=int(hf["hidden_size"]),
        num_attention_heads=int(hf["num_attention_heads"]),
        num_query_groups=int(hf.get("num_key_value_heads") or hf["num_attention_heads"]),
        ffn_hidden_size=int(hf["intermediate_size"]),
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        normalization="RMSNorm",
        layernorm_epsilon=float(hf.get("rms_norm_eps", 1e-6)),
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # DeepSeek normalises both LoRA latents (HF q_a_layernorm /
        # kv_a_layernorm). Megatron creates them only under qk_layernorm, and
        # fuses them into the up-projections as ``layer_norm_weight``. Without
        # this the model simply has no such parameters and the checkpoint's
        # norms would have nowhere to load.
        qk_layernorm=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        sequence_parallel=sp,
        use_cpu_initialization=True,
        variable_seq_lengths=(pp > 1),
        # --- MLA ---
        multi_latent_attention=True,
        q_lora_rank=int(hf.get("q_lora_rank") or 0),
        kv_lora_rank=int(hf["kv_lora_rank"]),
        qk_head_dim=int(hf["qk_nope_head_dim"]),
        qk_pos_emb_head_dim=int(hf["qk_rope_head_dim"]),
        v_head_dim=int(hf["v_head_dim"]),
        **rope,
        # Matches the generic path: alltoall is also the only dispatcher that
        # passes config validation under variable_seq_lengths.
        moe_token_dispatcher_type="alltoall",
        **moe,
        **recompute,
    )



# =============================== weight bridge ===============================
#
# Name mapping, verified against a constructed model rather than assumed. The
# non-obvious parts:
#
#   * The LoRA layernorms are FUSED into the up-projections by TE and surface as
#     ``linear_q_up_proj.layer_norm_weight`` / ``linear_kv_up_proj.layer_norm_weight``,
#     not as separate ``q_layernorm`` / ``kv_layernorm`` modules.
#   * A dense layer's post-attention norm is fused into ``mlp.linear_fc1`` as
#     ``layer_norm_weight``, while a MoE layer keeps a standalone
#     ``pre_mlp_layernorm``. Same HF key, two Megatron spellings.
#   * gate_proj and up_proj are concatenated into one ``linear_fc1`` along dim 0.
#   * The router bias is a BUFFER (``mlp.router.expert_bias``), so it never
#     appears in ``named_parameters()`` and has to be gathered separately.

import re  # noqa: E402

_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")
_GROUPED_EXPERT_RE = re.compile(r"^mlp\.experts\.linear_fc([12])\.weight(\d+)$")
_SEQUENTIAL_EXPERT_RE = re.compile(
    r"^mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight$"
)

# Megatron attention name -> HF attention name. Straight renames only.
_ATTN_MAP = {
    "self_attention.linear_q_down_proj.weight": "self_attn.q_a_proj.weight",
    "self_attention.linear_q_up_proj.weight": "self_attn.q_b_proj.weight",
    "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
    "self_attention.linear_kv_down_proj.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "self_attention.linear_kv_up_proj.weight": "self_attn.kv_b_proj.weight",
    "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "input_layernorm.weight": "input_layernorm.weight",
    # Both spellings of the post-attention norm collapse to the same HF key.
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
}

_TOP_LEVEL_MAP = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.final_layernorm.weight": "model.norm.weight",
    "output_layer.weight": "lm_head.weight",
}


def _strip_module_prefix(name: str) -> str:
    for pre in ("module.module.", "module."):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def _split_gate_up(t: torch.Tensor):
    """``linear_fc1`` is [gate; up] concatenated on dim 0."""
    gate, up = t.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


def megatron_to_hf_dsv3(named_params):
    """Yield ``(hf_name, tensor)`` from GLOBAL-indexed Megatron named params.

    Mirrors ``megatron_to_hf_moe``'s contract: expert params carry GLOBAL expert
    indices and layers GLOBAL layer numbers, so EP/PP relabelling happens in the
    caller. Router-bias buffers may be chained in by the caller; they are handled
    here like any other named tensor.
    """
    for raw, t in named_params:
        name = _strip_module_prefix(raw)

        top = _TOP_LEVEL_MAP.get(name)
        if top is not None:
            yield top, t
            continue

        m = _LAYER_RE.match(name)
        if not m:
            continue
        layer, rest = int(m.group(1)), m.group(2)
        hp = f"model.layers.{layer}."

        direct = _ATTN_MAP.get(rest)
        if direct is not None:
            yield hp + direct, t
            continue

        # dense MLP
        if rest == "mlp.linear_fc1.weight":
            gate, up = _split_gate_up(t)
            yield hp + "mlp.gate_proj.weight", gate
            yield hp + "mlp.up_proj.weight", up
            continue
        if rest == "mlp.linear_fc2.weight":
            yield hp + "mlp.down_proj.weight", t
            continue

        # shared expert
        if rest == "mlp.shared_experts.linear_fc1.weight":
            gate, up = _split_gate_up(t)
            yield hp + "mlp.shared_experts.gate_proj.weight", gate
            yield hp + "mlp.shared_experts.up_proj.weight", up
            continue
        if rest == "mlp.shared_experts.linear_fc2.weight":
            yield hp + "mlp.shared_experts.down_proj.weight", t
            continue

        # routed experts, grouped (``weight{E}``) or sequential (``local_experts.{E}``)
        gm = _GROUPED_EXPERT_RE.match(rest)
        if gm:
            fc, e = gm.group(1), int(gm.group(2))
        else:
            sm = _SEQUENTIAL_EXPERT_RE.match(rest)
            if not sm:
                continue
            e, fc = int(sm.group(1)), sm.group(2)
        ep_ = f"{hp}mlp.experts.{e}."
        if fc == "1":
            gate, up = _split_gate_up(t)
            yield ep_ + "gate_proj.weight", gate
            yield ep_ + "up_proj.weight", up
        else:
            yield ep_ + "down_proj.weight", t


def hf_to_dsv3_megatron(hf_state, d: DSV3Dims, use_grouped_mlp: bool = True):
    """Inverse of :func:`megatron_to_hf_dsv3` at TP=PP=EP=1.

    Returns ``{megatron_name: tensor}``. Sharded topologies slice this the same
    way the Qwen3 path does; keeping the whole-model conversion separate from the
    sharding keeps both testable.
    """
    hf = {_strip_module_prefix(k): v for k, v in hf_state.items()}
    meg: dict[str, torch.Tensor] = {}

    for meg_name, hf_name in _TOP_LEVEL_MAP.items():
        if hf_name in hf:
            meg[meg_name] = hf[hf_name]

    dense_layers = d.first_k_dense_replace
    for layer in range(d.num_layers):
        hp = f"model.layers.{layer}."
        mp = f"decoder.layers.{layer}."
        is_moe = layer >= dense_layers

        for meg_suffix, hf_suffix in _ATTN_MAP.items():
            # post-attention norm has two Megatron spellings; pick by layer type
            if meg_suffix == "pre_mlp_layernorm.weight" and not is_moe:
                continue
            if meg_suffix == "mlp.linear_fc1.layer_norm_weight" and is_moe:
                continue
            if meg_suffix.startswith("mlp.router") and not is_moe:
                continue
            src = hp + hf_suffix
            if src in hf:
                meg[mp + meg_suffix] = hf[src]

        if not is_moe:
            g, u = hp + "mlp.gate_proj.weight", hp + "mlp.up_proj.weight"
            if g in hf and u in hf:
                meg[mp + "mlp.linear_fc1.weight"] = torch.cat([hf[g], hf[u]], dim=0)
            if hp + "mlp.down_proj.weight" in hf:
                meg[mp + "mlp.linear_fc2.weight"] = hf[hp + "mlp.down_proj.weight"]
            continue

        if d.n_shared_experts > 0:
            g = hp + "mlp.shared_experts.gate_proj.weight"
            u = hp + "mlp.shared_experts.up_proj.weight"
            if g in hf and u in hf:
                meg[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                    [hf[g], hf[u]], dim=0
                )
            dn = hp + "mlp.shared_experts.down_proj.weight"
            if dn in hf:
                meg[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[dn]

        for e in range(d.num_experts):
            ep_ = f"{hp}mlp.experts.{e}."
            g, u, dn = ep_ + "gate_proj.weight", ep_ + "up_proj.weight", ep_ + "down_proj.weight"
            if g not in hf or u not in hf:
                continue
            fc1 = torch.cat([hf[g], hf[u]], dim=0)
            if use_grouped_mlp:
                meg[f"{mp}mlp.experts.linear_fc1.weight{e}"] = fc1
                meg[f"{mp}mlp.experts.linear_fc2.weight{e}"] = hf[dn]
            else:
                meg[f"{mp}mlp.experts.local_experts.{e}.linear_fc1.weight"] = fc1
                meg[f"{mp}mlp.experts.local_experts.{e}.linear_fc2.weight"] = hf[dn]

    return meg


def dsv3_router_bias_buffers(module):
    """Router bias lives in a buffer, so ``named_parameters()`` never yields it.

    DeepSeek's noaux_tc top-k reads this bias, so a rollout that never receives
    it selects different experts than the trainer did.
    """
    for name, buf in module.named_buffers():
        if name.endswith("mlp.router.expert_bias"):
            yield name, buf
