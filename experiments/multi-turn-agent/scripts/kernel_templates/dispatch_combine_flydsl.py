import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 128
TOP_K = 2


@flyc.jit
def _dispatch_jit(
    hidden_states: fx.Tensor,
    topk_ids: fx.Tensor,
    dispatched: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(hidden_states.shape[0]))
    D = fx.Int32(fx.get_scalar(hidden_states.shape[1]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        HS: fx.Tensor,
        IDS: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        col_block = fx.block_idx.y
        col = col_block * fx.Int32(BLOCK) + tid

        val = fx.Float32(HS[row, col])

        for k in fx.range_constexpr(TOP_K):
            slot = row * fx.Int32(TOP_K) + fx.Int32(k)
            OUT[slot, col] = fx.BFloat16(val)

    n_col_blocks = fx.ceildiv(D, fx.Int32(BLOCK))
    _kernel(hidden_states, topk_ids, dispatched).launch(
        grid=(M, n_col_blocks, 1), block=(BLOCK, 1, 1)
    )


@flyc.jit
def _combine_jit(
    expert_output: fx.Tensor,
    topk_weights: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(output.shape[0]))
    D = fx.Int32(fx.get_scalar(output.shape[1]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        EXPERT_OUT: fx.Tensor,
        WEIGHTS: fx.Tensor,
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
    _kernel(expert_output, topk_weights, output).launch(
        grid=(M, n_col_blocks, 1), block=(BLOCK, 1, 1)
    )


def dispatch_combine_kernel(hidden_states, topk_weights, topk_ids=None,
                            expert_output=None, num_tokens=None, num_experts=None, top_k=TOP_K):
    M, D = hidden_states.shape
    if topk_ids is None:
        topk_ids = torch.randint(0, 8, (M, top_k), dtype=torch.int32, device=hidden_states.device)
    total_slots = M * top_k
    dispatched = torch.empty(total_slots, D, dtype=torch.bfloat16, device=hidden_states.device)
    _dispatch_jit(hidden_states, topk_ids, dispatched)
    if expert_output is None:
        expert_output = dispatched
    output = torch.empty(M, D, dtype=torch.bfloat16, device=hidden_states.device)
    _combine_jit(expert_output, topk_weights, output)
    return output
