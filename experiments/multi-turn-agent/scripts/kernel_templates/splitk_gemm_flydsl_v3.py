import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK_N = 64
K_PER_SPLIT = 1024
N_SPLITS = 4


@flyc.jit
def _splitk_matmul_jit(
    a: fx.Tensor,
    bt: fx.Tensor,
    partial: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(a.shape[0]))
    N = fx.Int32(fx.get_scalar(bt.shape[1]))
    n_col_tiles = fx.ceildiv(N, fx.Int32(BLOCK_N))

    @flyc.kernel(known_block_size=[BLOCK_N, 1, 1])
    def _kernel(
        A: fx.Tensor,
        BT: fx.Tensor,
        OUT: fx.Tensor,
    ):
        m_idx = fx.block_idx.x
        n_tile = fx.block_idx.y
        split_id = fx.block_idx.z
        tid = fx.thread_idx.x
        n_idx = n_tile * fx.Int32(BLOCK_N) + tid

        k_offset = split_id * fx.Int32(K_PER_SPLIT)

        result = fx.Float32(0.0)
        for ki in fx.range_constexpr(K_PER_SPLIT):
            k_pos = k_offset + fx.Int32(ki)
            a_elem = fx.Float32(A[m_idx, k_pos])
            bt_elem = fx.Float32(BT[k_pos, n_idx])
            result = result + a_elem * bt_elem

        OUT[split_id, m_idx, n_idx] = result

    _kernel(a, bt, partial).launch(
        grid=(M, n_col_tiles, N_SPLITS), block=(BLOCK_N, 1, 1)
    )


def splitk_gemm_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    bt = b.t().contiguous()
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    partial = torch.zeros(N_SPLITS, M, N, dtype=torch.float32, device=a.device)
    _splitk_matmul_jit(a, bt, partial)
    c = partial.sum(dim=0).to(torch.bfloat16)
    return c


_KERNEL_OPERATOR = 'splitk_gemm'
_KERNEL_CFG_ID = 7005
