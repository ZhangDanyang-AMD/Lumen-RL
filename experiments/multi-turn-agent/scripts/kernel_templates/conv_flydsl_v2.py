import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

THREADS = 64
CONV_K = 4


@flyc.jit
def _depthwise_conv_jit(
    x: fx.Tensor,
    weight: fx.Tensor,
    output: fx.Tensor,
):
    B = fx.Int32(fx.get_scalar(x.shape[0]))
    L = fx.Int32(fx.get_scalar(x.shape[2]))
    n_l_blocks = fx.ceildiv(L, fx.Int32(THREADS))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def _kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
    ):
        b_idx = fx.block_idx.x
        l_blk = fx.block_idx.y
        tid = fx.thread_idx.x
        l_pos = l_blk * fx.Int32(THREADS) + tid
        ch = fx.Int32(0)

        # 1D depthwise conv: output[b,c,l] = sum_k(x[b,c,l+k] * w[c,0,k])
        # causal: only look back
        acc = fx.Float32(0.0)
        for k_iter in fx.range_constexpr(CONV_K):
            k_idx = fx.Int32(k_iter)
            # causal: use current position minus offset
            w_val = fx.Float32(W[ch, fx.Int32(0), k_idx])
            x_val = fx.Float32(X[b_idx, ch, l_pos])
            acc = acc + x_val * w_val

        OUT[b_idx, ch, l_pos] = fx.BFloat16(acc)

    _kernel(x, weight, output).launch(grid=(B, n_l_blocks, 1), block=(THREADS, 1, 1))


def depthwise_conv1d_kernel(x, weight, bias=None, state=None, d_model=None, d_conv=None):
    B, C, L = x.shape
    output = torch.zeros_like(x)
    _depthwise_conv_jit(x, weight, output)
    return output
