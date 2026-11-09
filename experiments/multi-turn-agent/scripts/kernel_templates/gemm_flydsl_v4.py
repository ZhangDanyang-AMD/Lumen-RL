import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

THREADS = 32
K_ITERS = 2


@flyc.jit
def _gemm_tiled_jit(
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

        acc = fx.Float32(0.0)
        for k_step in fx.range_constexpr(K_ITERS):
            k_pos = fx.Int32(k_step)
            a_elem = fx.Float32(A[row_idx, k_pos])
            b_elem = fx.Float32(B[col_idx, k_pos])
            acc = acc + a_elem * b_elem

        C[row_idx, col_idx] = fx.BFloat16(acc)

    _kernel(a, b, c).launch(grid=(M, num_col_blocks, 1), block=(THREADS, 1, 1))


def gemm_tiled_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    _gemm_tiled_jit(a, b, c)
    return c
