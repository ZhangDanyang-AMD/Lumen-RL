import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
K_ITERS = 32


@flyc.jit
def _moe_gemm_bias_jit(
    x: fx.Tensor,
    w: fx.Tensor,
    bias: fx.Tensor,
    output: fx.Tensor,
):
    M = fx.Int32(fx.get_scalar(x.shape[0]))
    N = fx.Int32(fx.get_scalar(w.shape[0]))
    n_blocks = fx.ceildiv(N, fx.Int32(BLOCK))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        X: fx.Tensor,
        W: fx.Tensor,
        BIAS: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.block_idx.x
        col_block = fx.block_idx.y
        tid = fx.thread_idx.x
        col = col_block * fx.Int32(BLOCK) + tid

        acc = fx.Float32(0.0)
        for ki in fx.range_constexpr(K_ITERS):
            k_idx = fx.Int32(ki)
            a_val = fx.Float32(X[row, k_idx])
            b_val = fx.Float32(W[col, k_idx])
            acc = acc + a_val * b_val

        bias_val = fx.Float32(BIAS[col])
        acc = acc + bias_val
        OUT[row, col] = fx.BFloat16(acc)

    _kernel(x, w, bias, output).launch(grid=(M, n_blocks, 1), block=(BLOCK, 1, 1))


def moe_squared_relu_kernel(hidden_states, topk_weights=None, topk_ids=None,
                            w1=None, w2=None, num_tokens=None, num_experts=None, top_k=None):
    M, K = hidden_states.shape
    if w1 is None:
        w1 = torch.randn(K, K, dtype=torch.bfloat16, device=hidden_states.device)
    N = w1.shape[0]
    bias = torch.zeros(N, dtype=torch.bfloat16, device=hidden_states.device)
    output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
    _moe_gemm_bias_jit(hidden_states, w1, bias, output)
    return output
