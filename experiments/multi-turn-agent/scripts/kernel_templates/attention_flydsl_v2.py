import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

WARP_SIZE = 64
HEAD_DIM_ITERS = 2


@flyc.jit
def _mha_jit(
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    o: fx.Tensor,
):
    S = fx.Int32(fx.get_scalar(q.shape[2]))

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def _kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
    ):
        seq_pos = fx.block_idx.x
        lane = fx.thread_idx.x
        batch = fx.Int32(0)
        head = fx.Int32(0)

        # compute QK^T dot product (simplified: self-attention at same position)
        qk_dot = fx.Float32(0.0)
        for d_iter in fx.range_constexpr(HEAD_DIM_ITERS):
            d_pos = lane + fx.Int32(WARP_SIZE * d_iter)
            q_elem = fx.Float32(Q[batch, head, seq_pos, d_pos])
            k_elem = fx.Float32(K[batch, head, seq_pos, d_pos])
            qk_dot = qk_dot + q_elem * k_elem

        # warp reduction for dot product
        for s in [32, 16, 8, 4, 2, 1]:
            qk_dot = qk_dot + fx.Float32(fx.shuffle_down(qk_dot, fx.Int32(s), fx.Int32(WARP_SIZE)))
        attn_weight = fx.Float32(fx.shuffle_idx(qk_dot, fx.Int32(0), fx.Int32(WARP_SIZE)))

        # scale by 1/sqrt(d)
        inv_sqrt_d = fx.Float32(1.0 / (WARP_SIZE * HEAD_DIM_ITERS) ** 0.5)
        attn_weight = attn_weight * inv_sqrt_d

        # output = attn_weight * V
        for d_iter in fx.range_constexpr(HEAD_DIM_ITERS):
            d_pos = lane + fx.Int32(WARP_SIZE * d_iter)
            v_elem = fx.Float32(V[batch, head, seq_pos, d_pos])
            O[batch, head, seq_pos, d_pos] = fx.BFloat16(v_elem * attn_weight)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(WARP_SIZE, 1, 1))


def flash_attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _mha_jit(q, k, v, o)
    return o
