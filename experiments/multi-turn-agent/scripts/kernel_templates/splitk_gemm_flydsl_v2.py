import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK_N = 128
K_PER_SPLIT = 512
N_SPLITS = 8


@flyc.jit
def _splitk_compute_jit(
    a: fx.Tensor,
    b: fx.Tensor,
    partial: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(a.shape[0]))
    N = fx.Int32(fx.get_scalar(b.shape[0]))
    n_col_blocks = fx.ceildiv(N, fx.Int32(BLOCK_N))

    @flyc.kernel(known_block_size=[BLOCK_N, 1, 1])
    def _kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        PARTIAL: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        split_id = fx.block_idx.z
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK_N) + tid

        k_start = split_id * fx.Int32(K_PER_SPLIT)

        acc = fx.Float32(0.0)
        for ki in fx.range_constexpr(K_PER_SPLIT):
            k_idx = k_start + fx.Int32(ki)
            a_val = fx.Float32(A[row, k_idx])
            b_val = fx.Float32(B[col, k_idx])
            acc = acc + a_val * b_val

        PARTIAL[split_id, row, col] = acc

    _kernel(a, b, partial).launch(
        grid=(M, n_col_blocks, N_SPLITS), block=(BLOCK_N, 1, 1)
    )


def splitk_gemm_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    partial = torch.zeros(N_SPLITS, M, N, dtype=torch.float32, device=a.device)
    _splitk_compute_jit(a, b, partial)
    c = partial.sum(dim=0).to(torch.bfloat16)
    return c


_KERNEL_OPERATOR = 'splitk_gemm'
_KERNEL_CFG_ID = 7003
