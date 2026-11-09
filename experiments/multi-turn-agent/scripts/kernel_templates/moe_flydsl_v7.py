import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 32
K_ITERS = 4


@flyc.jit
def _moe_small_gemm_jit(
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
        row_idx = fx.block_idx.x
        col_blk = fx.block_idx.y
        lane = fx.thread_idx.x
        col_idx = col_blk * fx.Int32(BLOCK) + lane

        acc = fx.Float32(0.0)
        for k_step in fx.range_constexpr(K_ITERS):
            k_pos = fx.Int32(k_step)
            a_elem = fx.Float32(X[row_idx, k_pos])
            b_elem = fx.Float32(W[col_idx, k_pos])
            acc = acc + a_elem * b_elem

        OUT[row_idx, col_idx] = fx.BFloat16(acc)

    _kernel(x, w, output).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def moe_hard_sigmoid_kernel(hidden_states, topk_weights=None, topk_ids=None,
                            w1=None, w2=None, num_tokens=None, num_experts=None, top_k=None):
    M, K = hidden_states.shape
    if w1 is None:
        w1 = torch.randn(K, K, dtype=torch.bfloat16, device=hidden_states.device)
    N = w1.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
    _moe_small_gemm_jit(hidden_states, w1, output)
    return output
