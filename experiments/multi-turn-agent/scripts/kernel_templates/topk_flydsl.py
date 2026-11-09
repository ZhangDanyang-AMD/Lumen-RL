import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, arith as _arith
from flydsl.expr.typing import T

BLOCK = 64
TOP_K = 2
N_EXPERTS = 8


@flyc.jit
def _topk_select_jit(
    scores: fx.Tensor,
    topk_vals: fx.Tensor,
    topk_ids: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(scores.shape[0]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        SCORES: fx.Tensor,
        VALS: fx.Tensor,
        IDS: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        best_val_0 = fx.Float32(-1e30)
        best_idx_0 = fx.Int32(-1)
        best_val_1 = fx.Float32(-1e30)
        best_idx_1 = fx.Int32(-1)

        for e in fx.range_constexpr(N_EXPERTS):
            s = fx.Float32(SCORES[row, fx.Int32(e)])
            cmp_0 = fx.cmpf(_arith.CmpFPredicate.OGT, s, best_val_0)
            cmp_1 = fx.cmpf(_arith.CmpFPredicate.OGT, s, best_val_1)

            new_v1 = _arith.select(cmp_0, best_val_0, _arith.select(cmp_1, s, best_val_1))
            new_i1 = _arith.select(cmp_0, best_idx_0, _arith.select(cmp_1, fx.Int32(e), best_idx_1))
            best_val_0 = _arith.select(cmp_0, s, best_val_0)
            best_idx_0 = _arith.select(cmp_0, fx.Int32(e), best_idx_0)
            best_val_1 = new_v1
            best_idx_1 = new_i1

        if tid == fx.Int32(0):
            VALS[row, fx.Int32(0)] = fx.BFloat16(best_val_0)
            VALS[row, fx.Int32(1)] = fx.BFloat16(best_val_1)
            IDS[row, fx.Int32(0)] = best_idx_0
            IDS[row, fx.Int32(1)] = best_idx_1

    _kernel(scores, topk_vals, topk_ids).launch(
        grid=(M, 1, 1), block=(BLOCK, 1, 1)
    )


def topk_kernel(scores, k=TOP_K):
    M, E = scores.shape
    topk_vals = torch.empty(M, k, dtype=torch.bfloat16, device=scores.device)
    topk_ids = torch.empty(M, k, dtype=torch.int32, device=scores.device)
    _topk_select_jit(scores, topk_vals, topk_ids)
    return topk_vals, topk_ids
