"""V4.1's MoE as Megatron's MoELayer, the one part of a `DSV41Stage` that has to be parallel:
routed experts expert-parallel (alltoall dispatch, grouped GEMMs), with the router and the
shared expert. The rest of a block stays the reference code, replicated over data-parallel
ranks -- V4.1's non-expert weights are ~16 GB in BF16, so on 288 GB MI355X cards there is no
tensor parallelism to do.

Needs the DSv4-patched Megatron (examples/DSv4_DAPO/megatron: sqrtsoftplus routing, the SwiGLU
clamp in routed and shared experts). The router is configured as for DSv4, whose gate V4.1
shares: sqrtsoftplus scores, an expert bias that only selects, normalized top-k weights scaled
by `route_scale`, no group limit.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lumenrl.models.deepseek_v41.config import DSV41Config, LayerSpec


def moe_config(cfg: DSV41Config, *, ep: int, pp: int = 1, dp_dtype: torch.dtype = torch.bfloat16, **overrides):
    """The TransformerConfig a V4.1 MoELayer is built from (only the MoE-relevant fields mean
    anything; the attention fields are placeholders Megatron's validation wants)."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = {
        "num_layers": cfg.n_layers,
        "hidden_size": cfg.dim,
        "num_attention_heads": cfg.n_heads,
        "ffn_hidden_size": cfg.moe_inter_dim,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "activation_func_clamp_value": float(cfg.swiglu_limit),
        "activation_func_clamp_shared_expert": True,
        "add_bias_linear": False,
        "bias_activation_fusion": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "num_moe_experts": cfg.n_routed_experts,
        "moe_ffn_hidden_size": cfg.moe_inter_dim,
        "moe_router_topk": cfg.n_activated_experts,
        "moe_shared_expert_intermediate_size": cfg.moe_inter_dim * cfg.n_shared_experts or None,
        "moe_router_pre_softmax": True,
        "moe_router_score_function": cfg.score_func,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
        "moe_router_topk_scaling_factor": float(cfg.route_scale),
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_aux_loss_coeff": 0.0,
        "moe_router_dtype": "fp32",
        "moe_token_dispatcher_type": "alltoall",
        "moe_grouped_gemm": True,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "sequence_parallel": False,
        "bf16": dp_dtype == torch.bfloat16,
        "params_dtype": dp_dtype,
        "pipeline_dtype": dp_dtype,
        "variable_seq_lengths": pp > 1,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


_SAME_ON_EVERY_RANK = {
    "router.weight": ("gate.weight",),
    "router.expert_bias": ("gate.bias",),
    "shared_experts.linear_fc1.weight": ("shared_experts.w1.weight", "shared_experts.w3.weight"),
    "shared_experts.linear_fc2.weight": ("shared_experts.w2.weight",),
}


def local_experts(n_experts: int, ep_size: int) -> int:
    if n_experts % ep_size:
        raise ValueError(f"{n_experts} experts do not split over EP {ep_size}")
    return n_experts // ep_size


def reference_names(key: str, *, ep_rank: int, n_local: int) -> list[str]:
    """The reference `MoE` state (`gate.*`, `experts.{e}.w1/w2/w3`, `shared_experts.*`) that the
    `MegatronMoE.layer` state `key` of EP rank `ep_rank` is made of, in concatenation order: gate
    before up in `linear_fc1`, local expert k is expert `ep_rank * n_local + k`. Empty for state
    the reference has no counterpart of (TransformerEngine's `_extra_state`)."""
    if key.endswith("_extra_state"):
        return []
    if key in _SAME_ON_EVERY_RANK:
        return list(_SAME_ON_EVERY_RANK[key])
    m = re.fullmatch(r"experts\.linear_fc([12])\.weight(\d+)", key)
    if m is None or int(m.group(2)) >= n_local:
        raise KeyError(f"MoE state {key!r} has no reference counterpart")
    e = ep_rank * n_local + int(m.group(2))
    return [f"experts.{e}.w1.weight", f"experts.{e}.w3.weight"] if m.group(1) == "1" else [f"experts.{e}.w2.weight"]


class MegatronMoE(nn.Module):
    """Drop-in for the reference `MoE` of a block: [b, s, d] in and out."""

    def __init__(self, tfcfg, layer_number: int):
        super().__init__()
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.spec_utils import build_module

        spec = get_moe_module_spec(use_te=True, num_experts=tfcfg.num_moe_experts, moe_grouped_gemm=True)
        pgs = ProcessGroupCollection.use_mpu_process_groups()
        self.layer = build_module(spec, config=tfcfg, layer_number=layer_number, pg_collection=pgs)
        self.ep_rank = mpu.get_expert_model_parallel_rank()
        self.n_local = local_experts(tfcfg.num_moe_experts, mpu.get_expert_model_parallel_world_size())

    def forward(self, x: Tensor) -> Tensor:
        out, bias = self.layer(x.transpose(0, 1).contiguous())
        if bias is not None:
            out = out + bias
        return out.transpose(0, 1)

    def reference_names(self, key: str) -> list[str]:
        """What `convert.load_released` reads for this module's state `key`, relative to where
        the reference `MoE` sits."""
        return reference_names(key.removeprefix("layer."), ep_rank=self.ep_rank, n_local=self.n_local)


def megatron_ffn(tfcfg) -> Callable[[LayerSpec], MegatronMoE]:
    """`DSV41Stage(ffn=...)`: every layer's MoE a `MegatronMoE`, built on the current device."""
    return lambda spec: MegatronMoE(tfcfg, layer_number=spec.layer + 1)


def moe_keys(n_local: int) -> list[str]:
    """The `MegatronMoE.layer` state a rank holding `n_local` experts loads."""
    experts = [f"experts.linear_fc{i}.weight{k}" for k in range(n_local) for i in (1, 2)]
    return [*_SAME_ON_EVERY_RANK, *experts]


def moe_state_from_reference(ffn: dict[str, Tensor], *, n_experts: int, ep_rank: int, ep_size: int) -> dict[str, Tensor]:
    """A reference `MoE`'s state dict as this EP rank's `MegatronMoE.layer` state: its own
    experts, gate and up fused in `linear_fc1`."""
    n_local = local_experts(n_experts, ep_size)
    return {
        key: torch.cat([ffn[name] for name in reference_names(key, ep_rank=ep_rank, n_local=n_local)])
        for key in moe_keys(n_local)
    }


def use_megatron_moe(stage, tfcfg) -> None:
    """Replace every block's reference MoE in `stage` with a `MegatronMoE`, carrying the
    reference weights over (this rank's experts only)."""
    from megatron.core import parallel_state as mpu

    ep_rank, ep_size = mpu.get_expert_model_parallel_rank(), mpu.get_expert_model_parallel_world_size()
    for name, block in stage.layers.items():
        ref = {k: v for k, v in block.ffn.state_dict().items()}
        moe = MegatronMoE(tfcfg, layer_number=int(name) + 1).to(device=torch.cuda.current_device())
        state = moe_state_from_reference(ref, n_experts=stage.cfg.n_routed_experts, ep_rank=ep_rank, ep_size=ep_size)
        missing, unexpected = moe.layer.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.endswith("_extra_state")]
        if missing or unexpected:
            raise KeyError(f"layer {name}: MoE state mismatch, missing {missing[:4]}, unexpected {unexpected[:4]}")
        block.ffn = moe
