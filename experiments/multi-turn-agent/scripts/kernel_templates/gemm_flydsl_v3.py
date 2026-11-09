import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

THREADS = 64
INNER_K = 16


@flyc.jit
def _gemm_vectorized_jit(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(a.shape[0]))
    N = fx.Int32(fx.get_scalar(b.shape[0]))
    num_col_blocks = fx.ceildiv(N, fx.Int32(THREADS))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def _kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        row_idx = fx.block_idx.x
        col_blk = fx.block_idx.y
        lane = fx.thread_idx.x
        col_idx = col_blk * fx.Int32(THREADS) + lane

        dot_product = fx.Float32(0.0)
        for k_step in fx.range_constexpr(INNER_K):
            k_pos = fx.Int32(k_step)
            lhs = fx.Float32(A[row_idx, k_pos])
            rhs = fx.Float32(B[col_idx, k_pos])
            dot_product = dot_product + lhs * rhs

        C[row_idx, col_idx] = fx.BFloat16(dot_product)

    _kernel(a, b, c).launch(grid=(M, num_col_blocks, 1), block=(THREADS, 1, 1))


def linear_kernel(x, weight, output=None):
    M, K = x.shape
    N = weight.shape[0]
    if output is None:
        output = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _gemm_vectorized_jit(x, weight, output)
    return output
