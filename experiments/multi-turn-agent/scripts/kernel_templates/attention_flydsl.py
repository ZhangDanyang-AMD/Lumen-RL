import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
D_ITERS = 2


@flyc.jit
def _attn_jit(
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    o: fx.Tensor,
):
    S = fx.Int32(fx.get_scalar(q.shape[2]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
    ):
        s_idx = fx.block_idx.x
        tid = fx.thread_idx.x
        b_idx = fx.Int32(0)
        h_idx = fx.Int32(0)

        dot = fx.Float32(0.0)
        for di in fx.range_constexpr(D_ITERS):
            d_idx = tid + fx.Int32(BLOCK * di)
            q_val = fx.Float32(Q[b_idx, h_idx, s_idx, d_idx])
            k_val = fx.Float32(K[b_idx, h_idx, s_idx, d_idx])
            dot = dot + q_val * k_val

        for offset in [32, 16, 8, 4, 2, 1]:
            dot = dot + fx.Float32(fx.shuffle_down(dot, fx.Int32(offset), fx.Int32(BLOCK)))
        attn_score = fx.Float32(fx.shuffle_idx(dot, fx.Int32(0), fx.Int32(BLOCK)))

        scale = fx.Float32(1.0 / (BLOCK * D_ITERS) ** 0.5)

        for di in fx.range_constexpr(D_ITERS):
            d_idx = tid + fx.Int32(BLOCK * di)
            v_val = fx.Float32(V[b_idx, h_idx, s_idx, d_idx])
            O[b_idx, h_idx, s_idx, d_idx] = fx.BFloat16(v_val * attn_score * scale)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _attn_jit(q, k, v, o)
    return o
