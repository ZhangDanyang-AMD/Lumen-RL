import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64


@flyc.jit
def _quant_jit(
    x: fx.Tensor,
    output: fx.Tensor,
    scale: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    N = fx.Int32(fx.get_scalar(x.shape[1]))
    n_col_blocks = fx.ceildiv(N, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        X: fx.Tensor,
        OUT: fx.Tensor,
        SCALE: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        x_val = fx.Float32(X[row, col])
        abs_val = x_val * x_val
        sq = abs_val + fx.Float32(1e-12)
        inv_sq = fx.Float32(rocdl.rsq(T.f32, sq.value))
        abs_x = x_val * x_val * inv_sq

        for offset in [32, 16, 8, 4, 2, 1]:
            other = fx.Float32(fx.shuffle_down(abs_x, fx.Int32(offset), fx.Int32(BLOCK)))
            if abs_x < other:
                abs_x = other
        amax = fx.Float32(fx.shuffle_idx(abs_x, fx.Int32(0), fx.Int32(BLOCK)))

        s = amax / fx.Float32(240.0) + fx.Float32(1e-12)
        inv_s = fx.Float32(1.0) / s

        OUT[row, col] = fx.BFloat16(x_val * inv_s)

    _kernel(x, output, scale).launch(grid=(M, n_col_blocks, 1), block=(BLOCK, 1, 1))


def quantize_kernel(x, output=None, scale=None, n_cols=None):
    M, N = x.shape
    if output is None:
        output = torch.zeros_like(x)
    if scale is None:
        scale = torch.empty(M, dtype=torch.float32, device=x.device)
    _quant_jit(x, output, scale)
    return output
