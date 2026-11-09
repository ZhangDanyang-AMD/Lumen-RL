import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK_N = 64
K_PER_SPLIT = 1024
N_SPLITS = 4


@flyc.jit
def _splitk_2pass_jit(
    a: fx.Tensor,
    b: fx.Tensor,
    partial: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(a.shape[0]))
    N = fx.Int32(fx.get_scalar(b.shape[0]))
    col_blocks = fx.ceildiv(N, fx.Int32(BLOCK_N))

    @flyc.kernel(known_block_size=[BLOCK_N, 1, 1])
    def _compute(
        A: fx.Tensor,
        B: fx.Tensor,
        P: fx.Tensor,
    ):
        row = fx.block_idx.x
        cb = fx.block_idx.y
        sid = fx.block_idx.z
        tid = fx.thread_idx.x
        col = cb * fx.Int32(BLOCK_N) + tid

        k_start = sid * fx.Int32(K_PER_SPLIT)

        dot = fx.Float32(0.0)
        for ki in fx.range_constexpr(K_PER_SPLIT):
            k_idx = k_start + fx.Int32(ki)
            dot = dot + fx.Float32(A[row, k_idx]) * fx.Float32(B[col, k_idx])

        P[sid, row, col] = dot

    @flyc.kernel(known_block_size=[BLOCK_N, 1, 1])
    def _reduce(
        P: fx.Tensor,
        C: fx.Tensor,
    ):
        row = fx.block_idx.x
        cb = fx.block_idx.y
        tid = fx.thread_idx.x
        col = cb * fx.Int32(BLOCK_N) + tid

        total = fx.Float32(0.0)
        for si in fx.range_constexpr(N_SPLITS):
            total = total + fx.Float32(P[fx.Int32(si), row, col])

        C[row, col] = fx.BFloat16(total)

    _compute(a, b, partial).launch(
        grid=(M, col_blocks, N_SPLITS), block=(BLOCK_N, 1, 1)
    )
    _reduce(partial, output).launch(
        grid=(M, col_blocks, 1), block=(BLOCK_N, 1, 1)
    )


def splitk_gemm_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    partial = torch.zeros(N_SPLITS, M, N, dtype=torch.float32, device=a.device)
    _splitk_2pass_jit(a, b, partial, c)
    return c


_KERNEL_OPERATOR = 'splitk_gemm'
_KERNEL_CFG_ID = 7007
