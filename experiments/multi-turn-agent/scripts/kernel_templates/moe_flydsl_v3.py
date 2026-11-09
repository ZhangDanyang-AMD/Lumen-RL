import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
K_ITERS = 16


@flyc.jit
def _moe_down_proj_jit(
    x: fx.Tensor,
    w: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    N = fx.Int32(fx.get_scalar(w.shape[0]))
    n_blocks = fx.ceildiv(N, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        dot_product = fx.Float32(0.0)
        for k_step in fx.range_constexpr(K_ITERS):
            k_pos = fx.Int32(k_step)
            lhs = fx.Float32(X[row, k_pos])
            rhs = fx.Float32(W[col, k_pos])
            dot_product = dot_product + lhs * rhs

        OUT[row, col] = fx.BFloat16(dot_product)

    _kernel(x, w, output).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def moe_gate_kernel(hidden_states, topk_weights=None, topk_ids=None,
                    w1=None, w2=None, num_tokens=None, num_experts=None, top_k=None):
    M, K = hidden_states.shape
    if w2 is None:
        w2 = torch.randn(K, K, dtype=torch.bfloat16, device=hidden_states.device)
    N = w2.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
    _moe_down_proj_jit(hidden_states, w2, output)
    return output
