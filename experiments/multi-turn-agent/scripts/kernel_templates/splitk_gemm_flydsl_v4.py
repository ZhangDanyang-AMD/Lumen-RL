import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK_N = 64
BLOCK_M = 4
K_PER_SPLIT = 1024
N_SPLITS = 4


@flyc.jit
def _splitk_tiled_jit(
    a: fx.Tensor,
    b: fx.Tensor,
    partial: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(a.shape[0]))
    N = fx.Int32(fx.get_scalar(b.shape[0]))
    m_tiles = fx.ceildiv(M, fx.Int32(BLOCK_M))
    n_tiles = fx.ceildiv(N, fx.Int32(BLOCK_N))

    @flyc.kernel(known_block_size=[BLOCK_N, 1, 1])
    def _kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        PARTIAL: fx.Tensor,
    ):
        m_tile_id = fx.block_idx.x
        n_tile_id = fx.block_idx.y
        split_idx = fx.block_idx.z
        lane = fx.thread_idx.x
        col = n_tile_id * fx.Int32(BLOCK_N) + lane

        k_base = split_idx * fx.Int32(K_PER_SPLIT)

        for mi in fx.range_constexpr(BLOCK_M):
            row = m_tile_id * fx.Int32(BLOCK_M) + fx.Int32(mi)
            acc = fx.Float32(0.0)
            for ki in fx.range_constexpr(K_PER_SPLIT):
                k_pos = k_base + fx.Int32(ki)
                a_val = fx.Float32(A[row, k_pos])
                b_val = fx.Float32(B[col, k_pos])
                acc = acc + a_val * b_val
            PARTIAL[split_idx, row, col] = acc

    _kernel(a, b, partial).launch(
        grid=(m_tiles, n_tiles, N_SPLITS), block=(BLOCK_N, 1, 1)
    )


def splitk_gemm_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    partial = torch.zeros(N_SPLITS, M, N, dtype=torch.float32, device=a.device)
    _splitk_tiled_jit(a, b, partial)
    c = partial.sum(dim=0).to(torch.bfloat16)
    return c


_KERNEL_OPERATOR = 'splitk_gemm'
_KERNEL_CFG_ID = 7006
