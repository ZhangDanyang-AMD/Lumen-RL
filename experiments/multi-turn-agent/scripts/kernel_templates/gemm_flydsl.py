import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
K_ITERS = 4


@flyc.jit
def _gemm_jit(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(a.shape[0]))
    N = fx.Int32(fx.get_scalar(b.shape[0]))
    n_blocks = fx.ceildiv(N, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        acc = fx.Float32(0.0)
        for ki in fx.range_constexpr(K_ITERS):
            k_idx = fx.Int32(ki)
            a_val = fx.Float32(A[row, k_idx])
            b_val = fx.Float32(B[col, k_idx])
            acc = acc + a_val * b_val

        C[row, col] = fx.BFloat16(acc)

    _kernel(a, b, c).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def gemm_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    _gemm_jit(a, b, c)
    return c
