import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 32


@flyc.jit
def _normalize_jit(
    x: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    K = fx.Int32(fx.get_scalar(x.shape[1]))
    n_blocks = fx.ceildiv(K, fx.Int32(BLOCK))

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

        # Per-element normalize: x / (|x| + eps)
        abs_val = x_val
        if x_val < fx.Float32(0.0):
            abs_val = fx.Float32(0.0) - x_val
        result = x_val / (abs_val + fx.Float32(1e-6))

        OUT[row, col] = fx.BFloat16(result)

    _kernel(x, output).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def normalize_kernel(hidden_states, topk_weights=None, topk_ids=None,
                     w1=None, w2=None, num_tokens=None, num_experts=None, top_k=None):
    M, K = hidden_states.shape
    output = torch.empty_like(hidden_states)
    _normalize_jit(hidden_states, output)
    return output
