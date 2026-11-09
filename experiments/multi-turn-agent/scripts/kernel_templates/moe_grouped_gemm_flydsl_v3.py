import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 128
TOP_K = 2
N_EXPERTS = 8
K_ITERS = 4096


@flyc.jit
def _moe_grouped_topk_jit(
    hidden_states: fx.Tensor,
    w1: fx.Tensor,
    topk_weights: fx.Tensor,
    topk_ids: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(hidden_states.shape[0]))
    N = fx.Int32(fx.get_scalar(w1.shape[1]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        HS: fx.Tensor,
        W: fx.Tensor,
        TW: fx.Tensor,
        TI: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        acc = fx.Float32(0.0)
        for k in fx.range_constexpr(TOP_K):
            eid = TI[row, fx.Int32(k)]
            w_val = fx.Float32(TW[row, fx.Int32(k)])

            dot = fx.Float32(0.0)
            for ki in fx.range_constexpr(K_ITERS):
                k_idx = fx.Int32(ki)
                h_val = fx.Float32(HS[row, k_idx])
                e_val = fx.Float32(W[eid, col, k_idx])
                dot = dot + h_val * e_val

            acc = acc + w_val * dot

        OUT[row, col] = fx.BFloat16(acc)

    n_col_blocks = fx.ceildiv(N, fx.Int32(BLOCK))
    _kernel(hidden_states, w1, topk_weights, topk_ids, output).launch(
        grid=(M, n_col_blocks, 1), block=(BLOCK, 1, 1)
    )


def moe_grouped_topk_kernel(hidden_states, w1, w2=None, w3=None,
                            topk_weights=None, topk_ids=None,
                            gate_up_weight=None, down_weight=None,
                            expert_weights=None, num_tokens=None,
                            num_experts=N_EXPERTS, top_k=TOP_K,
                            hidden_size=None, intermediate_size=None):
    M, K = hidden_states.shape
    if topk_weights is None:
        topk_weights = torch.ones(M, top_k, dtype=torch.float32, device=hidden_states.device) / top_k
    if topk_ids is None:
        topk_ids = torch.randint(0, num_experts, (M, top_k), dtype=torch.int32, device=hidden_states.device)
    N = w1.shape[1]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
    _moe_grouped_topk_jit(hidden_states, w1, topk_weights, topk_ids, output)
    return output
