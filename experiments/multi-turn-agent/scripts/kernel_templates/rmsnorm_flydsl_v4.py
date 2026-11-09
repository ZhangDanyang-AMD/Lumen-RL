import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
N_COLS = 2048
ITERS = N_COLS // BLOCK


@flyc.jit
def rmsnorm_2k_launch(x: fx.Tensor, weight: fx.Tensor, output: fx.Tensor):
    M = fx.Int32(fx.get_scalar(x.shape[0]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(X: fx.Tensor, W: fx.Tensor, OUT: fx.Tensor):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        sum_sq = fx.Float32(0.0)
        for i in fx.range_constexpr(ITERS):
            col = tid + fx.Int32(BLOCK * i)
            x_val = fx.Float32(X[row, col])
            sum_sq = sum_sq + x_val * x_val

        for offset in [32, 16, 8, 4, 2, 1]:
            sum_sq = sum_sq + fx.Float32(fx.shuffle_down(sum_sq, fx.Int32(offset), fx.Int32(BLOCK)))
        rrms = fx.Float32(rocdl.rsq(T.f32, (sum_sq / fx.Float32(N_COLS) + fx.Float32(1e-6)).value))
        rrms = fx.Float32(fx.shuffle_idx(rrms, fx.Int32(0), fx.Int32(BLOCK)))

        for i in fx.range_constexpr(ITERS):
            col = tid + fx.Int32(BLOCK * i)
            x_val = fx.Float32(X[row, col])
            w_val = fx.Float32(W[col])
            OUT[row, col] = fx.BFloat16(x_val * rrms * w_val)

    _kernel(x, weight, output).launch(grid=(M, 1, 1), block=(BLOCK, 1, 1))


def fused_rmsnorm_2k(x, weight, eps=1e-6):
    M, N = x.shape
    output = torch.empty_like(x)
    rmsnorm_2k_launch(x, weight, output)
    return output
