"""torchrun worker for the DSv3 sharding test. Not collected by pytest.

Builds the tiny DSv3 model at TP=2 / EP=2 and asserts that
``sharded_state_dict()`` describes MLA correctly. The engine's weight gather is
driven entirely by that metadata, so if it is right the generic gather
reconstructs MLA with no MLA-specific code -- and if it is wrong, the rollout
receives silently mis-assembled attention weights.

Prints ``DSV3_SHARD_OK`` on success; any assertion failure exits non-zero.
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ["LUMENRL_ROOT"])

from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.dist_checkpointing.mapping import ShardedTensor  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec  # noqa: E402
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402

from lumenrl.engine.training.dsv3_megatron_bridge import build_dsv3_config  # noqa: E402

TINY = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": 3, "hidden_size": 128, "num_attention_heads": 4,
    "num_key_value_heads": 4, "intermediate_size": 256, "vocab_size": 512,
    "rms_norm_eps": 1e-6, "q_lora_rank": 32, "kv_lora_rank": 16,
    "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
    "moe_intermediate_size": 64, "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
}

TP, EP = 2, 2

# global shape and per-axis fragmentation the engine's gather relies on.
#   hidden=128  heads=4  qk_nope=16  qk_rope=8  v=16  q_lora=32  kv_lora=16
EXPECTED = {
    # LoRA down-projections compress to the latent and are replicated.
    "self_attention.linear_q_down_proj.weight": ((32, 128), (1, 1)),
    "self_attention.linear_kv_down_proj.weight": ((24, 128), (1, 1)),
    # Up-projections fan out to heads -> column-parallel on dim 0.
    "self_attention.linear_q_up_proj.weight": ((96, 32), (2, 1)),
    "self_attention.linear_kv_up_proj.weight": ((128, 16), (2, 1)),
    # Output projection consumes the head dim -> row-parallel on dim 1.
    "self_attention.linear_proj.weight": ((128, 64), (1, 2)),
    # Fused LoRA norms live on the latent and are replicated.
    "self_attention.linear_q_up_proj.layer_norm_weight": ((32,), (1,)),
    "self_attention.linear_kv_up_proj.layer_norm_weight": ((16,), (1,)),
}


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

    cfg = build_dsv3_config(TINY, {"moe_grouped_gemm": True}, tp=TP, ep=EP, etp=1, sp=True)
    spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True)
    model = GPTModel(
        config=cfg, transformer_layer_spec=spec,
        vocab_size=TINY["vocab_size"], max_sequence_length=64,
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope", parallel_output=False,
    )

    ssd = model.sharded_state_dict()
    for suffix, (want_global, want_frag) in EXPECTED.items():
        key = f"decoder.layers.1.{suffix}"
        assert key in ssd, f"[rank{rank}] {key} absent from sharded_state_dict"
        st = ssd[key]
        assert isinstance(st, ShardedTensor), f"[rank{rank}] {key} is {type(st).__name__}"
        assert tuple(st.global_shape) == want_global, (
            f"[rank{rank}] {key} global {tuple(st.global_shape)} != {want_global}"
        )
        assert tuple(st.axis_fragmentations) == want_frag, (
            f"[rank{rank}] {key} fragmentation {tuple(st.axis_fragmentations)} != {want_frag}"
        )
        # The local shard must be the global shape divided along the sharded axis.
        expect_local = tuple(
            g // f for g, f in zip(want_global, want_frag)
        )
        assert tuple(st.local_shape) == expect_local, (
            f"[rank{rank}] {key} local {tuple(st.local_shape)} != {expect_local}"
        )

    # EP: experts are split across the expert-parallel group, not replicated.
    local_experts = sum(
        1 for n, _ in model.named_parameters()
        if "layers.1.mlp.experts.linear_fc1.weight" in n
    )
    assert local_experts == TINY["n_routed_experts"] // EP, (
        f"[rank{rank}] {local_experts} local experts, expected "
        f"{TINY['n_routed_experts'] // EP} at EP={EP}"
    )

    # The router and its bias are replicated: every rank must be able to emit them.
    assert "decoder.layers.1.mlp.router.weight" in dict(model.named_parameters())
    assert any(
        n.endswith("layers.1.mlp.router.expert_bias") for n, _ in model.named_buffers()
    ), f"[rank{rank}] router bias buffer missing"

    dist.barrier()
    if rank == 0:
        print("DSV3_SHARD_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
