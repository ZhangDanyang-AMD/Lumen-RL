import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

THREADS = 32


@flyc.jit
def _absmax_quant_jit(
    x: fx.Tensor,
    output: fx.Tensor,
    scale: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    N = fx.Int32(fx.get_scalar(x.shape[1]))
    n_col_tiles = fx.ceildiv(N, fx.Int32(THREADS))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def _kernel(
        X: fx.Tensor,
        OUT: fx.Tensor,
        SCALE: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_tile = fx.block_idx.y
        lane = fx.thread_idx.x
        col = col_tile * fx.Int32(THREADS) + lane

        val = fx.Float32(X[row, col])

        # absmax: find max absolute value across warp
        abs_val = val
        if val < fx.Float32(0.0):
            abs_val = fx.Float32(0.0) - val

        local_max = abs_val
        for s in [16, 8, 4, 2, 1]:
            neighbor = fx.Float32(fx.shuffle_down(local_max, fx.Int32(s), fx.Int32(THREADS)))
            if neighbor > local_max:
                local_max = neighbor
        row_max = fx.Float32(fx.shuffle_idx(local_max, fx.Int32(0), fx.Int32(THREADS)))

        tile_scale = row_max / fx.Float32(127.0) + fx.Float32(1e-10)
        inv_scale = fx.Float32(1.0) / tile_scale

        OUT[row, col] = fx.BFloat16(val * inv_scale)

    _kernel(x, output, scale).launch(grid=(M, n_col_tiles, 1), block=(THREADS, 1, 1))


def absmax_quantize(x, output=None, scale=None, n_cols=None):
    M, N = x.shape
    if output is None:
        output = torch.zeros_like(x)
    if scale is None:
        scale = torch.empty(M, dtype=torch.float32, device=x.device)
    _absmax_quant_jit(x, output, scale)
    return output
