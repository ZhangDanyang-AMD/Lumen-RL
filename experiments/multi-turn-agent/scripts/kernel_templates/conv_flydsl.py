import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64


@flyc.jit
def _conv_jit(
    x: fx.Tensor,
    weight: fx.Tensor,
    output: fx.Tensor,
):
    B = fx.Int32(fx.get_scalar(x.shape[0]))
    L = fx.Int32(fx.get_scalar(x.shape[2]))
    n_blocks = fx.ceildiv(L, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
    ):
        batch = fx.block_idx.x
        l_block = fx.block_idx.y
        tid = fx.thread_idx.x
        l_idx = l_block * fx.Int32(BLOCK) + tid
        c_idx = fx.Int32(0)

        x_val = fx.Float32(X[batch, c_idx, l_idx])
        w_val = fx.Float32(W[c_idx, fx.Int32(0), fx.Int32(0)])
        OUT[batch, c_idx, l_idx] = fx.BFloat16(x_val * w_val)

    _kernel(x, weight, output).launch(grid=(B, n_blocks, 1), block=(BLOCK, 1, 1))


def causal_conv1d_kernel(x, weight, bias=None, state=None, d_model=None, d_conv=None):
    B, C, L = x.shape
    output = torch.zeros_like(x)
    _conv_jit(x, weight, output)
    return output
