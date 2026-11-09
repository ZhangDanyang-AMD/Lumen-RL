import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
K_ITERS = 4096


@flyc.jit
def _gemm_a8w8_fused_scale_jit(
    x: fx.Tensor,
    w: fx.Tensor,
    x_scale: fx.Tensor,
    w_scale: fx.Tensor,
    y: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    N = fx.Int32(fx.get_scalar(w.shape[0]))
    n_blocks = fx.ceildiv(N, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        XS: fx.Tensor,
        WS: fx.Tensor,
        Y: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        sx = fx.Float32(XS[row])
        sw = fx.Float32(WS[col])
        combined_scale = sx * sw

        acc = fx.Float32(0.0)
        for ki in fx.range_constexpr(K_ITERS):
            k_idx = fx.Int32(ki)
            x_val = fx.Float32(X[row, k_idx])
            w_val = fx.Float32(W[col, k_idx])
            acc = acc + x_val * w_val

        Y[row, col] = fx.BFloat16(acc * combined_scale)

    _kernel(x, w, x_scale, w_scale, y).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def gemm_a8w8_fused_kernel(x, w, x_scale, w_scale, y=None):
    M, K = x.shape
    N = w.shape[0]
    if y is None:
        y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _gemm_a8w8_fused_scale_jit(x, w, x_scale, w_scale, y)
    return y
