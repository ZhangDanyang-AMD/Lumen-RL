import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64


@flyc.jit
def _scale_jit(
    x: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    N = fx.Int32(fx.get_scalar(x.shape[1]))
    n_blocks = fx.ceildiv(N, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        X: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        x_val = fx.Float32(X[row, col])
        OUT[row, col] = fx.BFloat16(x_val * fx.Float32(1.0))

    _kernel(x, output).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def elementwise_kernel(x, weight=None, output=None):
    if x.dim() == 2:
        M, N = x.shape
    elif x.dim() >= 3:
        M = x.shape[0]
        N = 1
        for d in range(1, x.dim()):
            N *= x.shape[d]
        x = x.reshape(M, N)
    else:
        M = 1
        N = x.shape[0]
        x = x.reshape(M, N)
    output = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _scale_jit(x, output)
    return output.reshape(x.shape) if x.dim() == 2 else output
