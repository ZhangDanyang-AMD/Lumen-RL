import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as _arith

BLOCK = 32
N_EXPERTS = 8
TOP_K = 2


@flyc.jit
def _moe_router_select_jit(
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

        peak = fx.Float32(-1e30)
        for e in fx.range_constexpr(N_EXPERTS):
            val = fx.Float32(LOGITS[row, fx.Int32(e)])
            peak = fx.max(peak, val)

        denom = fx.Float32(0.0)
        for e in fx.range_constexpr(N_EXPERTS):
            val = fx.Float32(LOGITS[row, fx.Int32(e)])
            denom = denom + fx.exp(val - peak)

        rcp_denom = fx.Float32(1.0) / denom

        first_val = fx.Float32(-1e30)
        first_idx = fx.Int32(0)
        second_val = fx.Float32(-1e30)
        second_idx = fx.Int32(0)

        for e in fx.range_constexpr(N_EXPERTS):
            val = fx.Float32(LOGITS[row, fx.Int32(e)])
            prob = fx.exp(val - peak) * rcp_denom

            is_top = fx.cmpf(_arith.CmpFPredicate.OGT, prob, first_val)
            is_second = fx.cmpf(_arith.CmpFPredicate.OGT, prob, second_val)

            new_sv = _arith.select(is_top, first_val, _arith.select(is_second, prob, second_val))
            new_si = _arith.select(is_top, first_idx, _arith.select(is_second, fx.Int32(e), second_idx))

            first_val = _arith.select(is_top, prob, first_val)
            first_idx = _arith.select(is_top, fx.Int32(e), first_idx)
            second_val = new_sv
            second_idx = new_si

        scale = fx.Float32(1.0) / (first_val + second_val)

        if tid == fx.Int32(0):
            WEIGHTS[row, fx.Int32(0)] = fx.BFloat16(first_val * scale)
            WEIGHTS[row, fx.Int32(1)] = fx.BFloat16(second_val * scale)
            IDS[row, fx.Int32(0)] = first_idx
            IDS[row, fx.Int32(1)] = second_idx

    _kernel(logits, topk_weights, topk_ids).launch(
        grid=(M, 1, 1), block=(BLOCK, 1, 1)
    )


def moe_router_select_kernel(logits, num_experts=N_EXPERTS, top_k=TOP_K):
    M = logits.shape[0]
    topk_weights = torch.empty(M, top_k, dtype=torch.bfloat16, device=logits.device)
    topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=logits.device)
    _moe_router_select_jit(logits, topk_weights, topk_ids)
    return topk_weights, topk_ids
