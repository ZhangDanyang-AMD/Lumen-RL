import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as _arith

BLOCK = 128
N_EXPERTS = 8
TOP_K = 2


@flyc.jit
def _moe_softmax_route_jit(
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

        max_logit = fx.Float32(-1e30)
        for e in fx.range_constexpr(N_EXPERTS):
            l = fx.Float32(LOGITS[row, fx.Int32(e)])
            max_logit = fx.max(max_logit, l)

        total_exp = fx.Float32(0.0)
        for e in fx.range_constexpr(N_EXPERTS):
            l = fx.Float32(LOGITS[row, fx.Int32(e)])
            total_exp = total_exp + fx.exp(l - max_logit)

        inv_total = fx.Float32(1.0) / total_exp

        top0_val = fx.Float32(-1e30)
        top0_idx = fx.Int32(0)
        top1_val = fx.Float32(-1e30)
        top1_idx = fx.Int32(0)

        for e in fx.range_constexpr(N_EXPERTS):
            l = fx.Float32(LOGITS[row, fx.Int32(e)])
            p = fx.exp(l - max_logit) * inv_total

            gt_0 = fx.cmpf(_arith.CmpFPredicate.OGT, p, top0_val)
            gt_1 = fx.cmpf(_arith.CmpFPredicate.OGT, p, top1_val)

            bump_v1 = _arith.select(gt_0, top0_val, _arith.select(gt_1, p, top1_val))
            bump_i1 = _arith.select(gt_0, top0_idx, _arith.select(gt_1, fx.Int32(e), top1_idx))

            top0_val = _arith.select(gt_0, p, top0_val)
            top0_idx = _arith.select(gt_0, fx.Int32(e), top0_idx)
            top1_val = bump_v1
            top1_idx = bump_i1

        norm_factor = fx.Float32(1.0) / (top0_val + top1_val)

        if tid == fx.Int32(0):
            WEIGHTS[row, fx.Int32(0)] = fx.BFloat16(top0_val * norm_factor)
            WEIGHTS[row, fx.Int32(1)] = fx.BFloat16(top1_val * norm_factor)
            IDS[row, fx.Int32(0)] = top0_idx
            IDS[row, fx.Int32(1)] = top1_idx

    _kernel(logits, topk_weights, topk_ids).launch(
        grid=(M, 1, 1), block=(BLOCK, 1, 1)
    )


def moe_softmax_route_kernel(logits, num_experts=N_EXPERTS, top_k=TOP_K):
    M = logits.shape[0]
    topk_weights = torch.empty(M, top_k, dtype=torch.bfloat16, device=logits.device)
    topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=logits.device)
    _moe_softmax_route_jit(logits, topk_weights, topk_ids)
    return topk_weights, topk_ids
