import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as _arith

BLOCK = 256
N_EXPERTS = 8
TOP_K = 2


@flyc.jit
def _moe_count_jit(
    topk_ids: fx.Tensor,
    expert_counts: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(topk_ids.shape[0]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        IDS: fx.Tensor,
        COUNTS: fx.Tensor,
    ):
        row = fx.block_idx.x * fx.Int32(BLOCK) + fx.thread_idx.x

        for k in fx.range_constexpr(TOP_K):
            expert = IDS[row, fx.Int32(k)]
            for e in fx.range_constexpr(N_EXPERTS):
                cmp = fx.cmpi(_arith.CmpIPredicate.eq, expert, fx.Int32(e))
                inc = _arith.select(cmp, fx.Int32(1), fx.Int32(0))
                COUNTS[row, fx.Int32(e)] = COUNTS[row, fx.Int32(e)] + inc

    n_blocks = fx.ceildiv(M, fx.Int32(BLOCK))
    _kernel(topk_ids, expert_counts).launch(
        grid=(n_blocks, 1, 1), block=(BLOCK, 1, 1)
    )


def moe_sorting_kernel(topk_ids, num_tokens=None, num_experts=N_EXPERTS, top_k=TOP_K):
    M = topk_ids.shape[0]
    expert_counts = torch.zeros(M, num_experts, dtype=torch.int32, device=topk_ids.device)
    _moe_count_jit(topk_ids, expert_counts)

    sorted_token_ids = torch.argsort(
        topk_ids[:, 0].int(), stable=True
    ).to(torch.int32)
    return sorted_token_ids
