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

Scope note: the **weight bridge is not implemented here yet**. The config and
layer spec are enough to construct the model, which is what the registry entry
needs to exist and be tested; HF<->Megatron weight conversion for MLA (q/kv
LoRA projections against ``linear_q_down_proj`` / ``linear_q_up_proj`` /
``linear_kv_down_proj`` / ``linear_kv_up_proj``) plus 256-384 routed experts and
the shared expert is a separate, larger piece of work. ``export_weights``
raises until it lands, rather than silently shipping wrong tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from lumenrl.engine.training.qwen3_megatron_bridge import Qwen3Dims

__all__ = ["DSV3Dims", "is_dsv3", "build_dsv3_dims", "build_dsv3_config"]

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
        # Matches the generic path: alltoall is also the only dispatcher that
        # passes config validation under variable_seq_lengths.
        moe_token_dispatcher_type="alltoall",
        **moe,
        **recompute,
    )


def export_weights_not_implemented(engine):  # noqa: ARG001
    """Placeholder until the MLA weight bridge lands.

    Raising here is deliberate. The alternative -- falling back to the Qwen3
    exporter -- would stream tensors whose names do not match MLA's parameter
    set, and the rollout would silently run on a partly-uninitialised model.
    """
    raise NotImplementedError(
        "DeepSeek-V3 weight export is not implemented yet. The model can be "
        "constructed (config + layer spec), but HF<->Megatron conversion for the "
        "MLA projections (linear_q_down_proj / linear_q_up_proj / "
        "linear_kv_down_proj / linear_kv_up_proj) and the routed + shared experts "
        "still has to be written. See dsv3_megatron_bridge."
    )
