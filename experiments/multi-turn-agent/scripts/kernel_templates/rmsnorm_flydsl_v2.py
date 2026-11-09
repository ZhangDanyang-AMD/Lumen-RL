import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

WARP_SIZE = 64
COL_STRIDE = 8


@flyc.jit
def _layernorm_jit(
    x: fx.Tensor,
    weight: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def _kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        # compute sum and sum of squares for mean and variance
        total = fx.Float32(0.0)
        total_sq = fx.Float32(0.0)
        for i in fx.range_constexpr(COL_STRIDE):
            col = tid + fx.Int32(WARP_SIZE * i)
            val = fx.Float32(X[row, col])
            total = total + val
            total_sq = total_sq + val * val

        # warp reduce both
        for offset in [32, 16, 8, 4, 2, 1]:
            total = total + fx.Float32(fx.shuffle_down(total, fx.Int32(offset), fx.Int32(WARP_SIZE)))
            total_sq = total_sq + fx.Float32(fx.shuffle_down(total_sq, fx.Int32(offset), fx.Int32(WARP_SIZE)))

        total = fx.Float32(fx.shuffle_idx(total, fx.Int32(0), fx.Int32(WARP_SIZE)))
        total_sq = fx.Float32(fx.shuffle_idx(total_sq, fx.Int32(0), fx.Int32(WARP_SIZE)))

        n_elem = fx.Float32(WARP_SIZE * COL_STRIDE)
        mean = total / n_elem
        var = total_sq / n_elem - mean * mean
        inv_std = fx.Float32(rocdl.rsq(T.f32, (var + fx.Float32(1e-5)).value))

        for i in fx.range_constexpr(COL_STRIDE):
            col = tid + fx.Int32(WARP_SIZE * i)
            val = fx.Float32(X[row, col])
            w_val = fx.Float32(W[col])
            normed = (val - mean) * inv_std * w_val
            OUT[row, col] = fx.BFloat16(normed)

    _kernel(x, weight, output).launch(grid=(M, 1, 1), block=(WARP_SIZE, 1, 1))


def fused_layernorm(x, weight, eps=1e-5):
    M, N = x.shape
    output = torch.empty_like(x)
    _layernorm_jit(x, weight, output)
    return output
