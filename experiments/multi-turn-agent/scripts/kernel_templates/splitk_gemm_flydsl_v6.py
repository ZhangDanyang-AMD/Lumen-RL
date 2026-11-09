import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK_N = 128
K_PER_SPLIT = 512
N_SPLITS = 8


@flyc.jit
def _splitk_fused_jit(
    a: fx.Tensor,
    b: fx.Tensor,
    partial: fx.Tensor,
):
    rows = fx.Int32(fx.get_scalar(a.shape[0]))
    cols = fx.Int32(fx.get_scalar(b.shape[0]))
    col_groups = fx.ceildiv(cols, fx.Int32(BLOCK_N))

    @flyc.kernel(known_block_size=[BLOCK_N, 1, 1])
    def _gemm_split(
        X: fx.Tensor,
        W: fx.Tensor,
        P: fx.Tensor,
    ):
        r = fx.block_idx.x
        cg = fx.block_idx.y
        s = fx.block_idx.z
        t = fx.thread_idx.x
        c = cg * fx.Int32(BLOCK_N) + t

        k0 = s * fx.Int32(K_PER_SPLIT)

        sum_val = fx.Float32(0.0)
        for kk in fx.range_constexpr(K_PER_SPLIT):
            ki = k0 + fx.Int32(kk)
            xv = fx.Float32(X[r, ki])
            wv = fx.Float32(W[c, ki])
            prod = xv * wv
            sum_val = sum_val + prod

        P[s, r, c] = sum_val

    _gemm_split(a, b, partial).launch(
        grid=(rows, col_groups, N_SPLITS), block=(BLOCK_N, 1, 1)
    )


def splitk_gemm_kernel(a, b, c=None):
    M, K = a.shape
    N = b.shape[0]
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    partial = torch.zeros(N_SPLITS, M, N, dtype=torch.float32, device=a.device)
    _splitk_fused_jit(a, b, partial)
    c = partial.sum(dim=0).to(torch.bfloat16)
    return c


_KERNEL_OPERATOR = 'splitk_gemm'
_KERNEL_CFG_ID = 7008
