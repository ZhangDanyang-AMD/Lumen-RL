import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as _arith

BLOCK = 64
N_EXPERTS = 8
TOP_K = 2


@flyc.jit
def _moe_routing_jit(
    logits: fx.Tensor,
    topk_weights: fx.Tensor,
    topk_ids: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(logits.shape[0]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        LOGITS: fx.Tensor,
        WEIGHTS: fx.Tensor,
        IDS: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        max_val = fx.Float32(-1e30)
        for e in fx.range_constexpr(N_EXPERTS):
            v = fx.Float32(LOGITS[row, fx.Int32(e)])
            max_val = fx.max(max_val, v)

        sum_exp = fx.Float32(0.0)
        for e in fx.range_constexpr(N_EXPERTS):
            v = fx.Float32(LOGITS[row, fx.Int32(e)])
            exp_v = fx.exp(v - max_val)
            sum_exp = sum_exp + exp_v

        inv_sum = fx.Float32(1.0) / sum_exp

        best_val_0 = fx.Float32(-1e30)
        best_idx_0 = fx.Int32(0)
        best_val_1 = fx.Float32(-1e30)
        best_idx_1 = fx.Int32(0)

        for e in fx.range_constexpr(N_EXPERTS):
            v = fx.Float32(LOGITS[row, fx.Int32(e)])
            prob = fx.exp(v - max_val) * inv_sum

            cmp_0 = fx.cmpf(_arith.CmpFPredicate.OGT, prob, best_val_0)
            cmp_1 = fx.cmpf(_arith.CmpFPredicate.OGT, prob, best_val_1)

            new_v1 = _arith.select(cmp_0, best_val_0, _arith.select(cmp_1, prob, best_val_1))
            new_i1 = _arith.select(cmp_0, best_idx_0, _arith.select(cmp_1, fx.Int32(e), best_idx_1))

            best_val_0 = _arith.select(cmp_0, prob, best_val_0)
            best_idx_0 = _arith.select(cmp_0, fx.Int32(e), best_idx_0)
            best_val_1 = new_v1
            best_idx_1 = new_i1

        renorm = fx.Float32(1.0) / (best_val_0 + best_val_1)

        if tid == fx.Int32(0):
            WEIGHTS[row, fx.Int32(0)] = fx.BFloat16(best_val_0 * renorm)
            WEIGHTS[row, fx.Int32(1)] = fx.BFloat16(best_val_1 * renorm)
            IDS[row, fx.Int32(0)] = best_idx_0
            IDS[row, fx.Int32(1)] = best_idx_1

    _kernel(logits, topk_weights, topk_ids).launch(
        grid=(M, 1, 1), block=(BLOCK, 1, 1)
    )


def moe_routing_kernel(logits, num_experts=N_EXPERTS, top_k=TOP_K):
    M = logits.shape[0]
    topk_weights = torch.empty(M, top_k, dtype=torch.bfloat16, device=logits.device)
    topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=logits.device)
    _moe_routing_jit(logits, topk_weights, topk_ids)
    return topk_weights, topk_ids
