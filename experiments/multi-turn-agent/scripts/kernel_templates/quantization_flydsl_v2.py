import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

THREADS = 64


@flyc.jit
def _dynamic_quant_jit(
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
        val_sq = val * val

        # cross-lane max reduction
        local_max = val_sq
        for s in [32, 16, 8, 4, 2, 1]:
            neighbor = fx.Float32(fx.shuffle_down(local_max, fx.Int32(s), fx.Int32(THREADS)))
            if local_max < neighbor:
                local_max = neighbor
        row_max = fx.Float32(fx.shuffle_idx(local_max, fx.Int32(0), fx.Int32(THREADS)))

        # compute per-tile scale
        rms = fx.Float32(rocdl.rsq(T.f32, (row_max + fx.Float32(1e-8)).value))
        inv_max = rms * row_max
        tile_scale = inv_max / fx.Float32(127.0) + fx.Float32(1e-10)
        inv_scale = fx.Float32(1.0) / tile_scale

        OUT[row, col] = fx.BFloat16(val * inv_scale)

    _kernel(x, output, scale).launch(grid=(M, n_col_tiles, 1), block=(THREADS, 1, 1))


def per_tensor_quantize(x, output=None, scale=None, n_cols=None):
    M, N = x.shape
    if output is None:
        output = torch.zeros_like(x)
    if scale is None:
        scale = torch.empty(M, dtype=torch.float32, device=x.device)
    _dynamic_quant_jit(x, output, scale)
    return output
