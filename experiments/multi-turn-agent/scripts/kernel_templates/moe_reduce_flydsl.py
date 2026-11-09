import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 128
TOP_K = 2
N_EXPERTS = 8
HIDDEN_DIM_ITERS = 16


@flyc.jit
def _moe_reduce_jit(
    expert_output: fx.Tensor,
    topk_weights: fx.Tensor,
    topk_ids: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(output.shape[0]))
    D = fx.Int32(fx.get_scalar(output.shape[1]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        EXPERT_OUT: fx.Tensor,
        WEIGHTS: fx.Tensor,
        IDS: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        col_block = fx.block_idx.y

        col = col_block * fx.Int32(BLOCK) + tid

        acc = fx.Float32(0.0)
        for k in fx.range_constexpr(TOP_K):
            w = fx.Float32(WEIGHTS[row, fx.Int32(k)])
            slot = row * fx.Int32(TOP_K) + fx.Int32(k)
            val = fx.Float32(EXPERT_OUT[slot, col])
            acc = acc + w * val

        OUT[row, col] = fx.BFloat16(acc)

    n_col_blocks = fx.ceildiv(D, fx.Int32(BLOCK))
    _kernel(expert_output, topk_weights, topk_ids, output).launch(
        grid=(M, n_col_blocks, 1), block=(BLOCK, 1, 1)
    )


def moe_reduce_kernel(expert_output, topk_weights, topk_ids=None,
                      hidden_states=None, num_tokens=None, num_experts=None, top_k=TOP_K):
    if hidden_states is not None:
        M, D = hidden_states.shape
    else:
        M = topk_weights.shape[0]
        D = expert_output.shape[1]
    if topk_ids is None:
        topk_ids = torch.randint(0, N_EXPERTS, (M, top_k), dtype=torch.int32, device=expert_output.device)
    output = torch.empty(M, D, dtype=torch.bfloat16, device=expert_output.device)
    _moe_reduce_jit(expert_output, topk_weights, topk_ids, output)
    return output
