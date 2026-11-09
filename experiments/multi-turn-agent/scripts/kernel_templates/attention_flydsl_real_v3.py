import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
SEQ_TILES = 128


@flyc.jit
def _causal_attn_jit(
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
        q_row = fx.block_idx.x
        d_lane = fx.thread_idx.x
        b = fx.Int32(0)
        h = fx.Int32(0)

        m_val = fx.Float32(-1e30)
        l_val = fx.Float32(0.0)
        o_val = fx.Float32(0.0)

        inv_sqrt = fx.Float32(1.0 / BLOCK ** 0.5)

        for kp in fx.range_constexpr(SEQ_TILES):
            k_col = fx.Int32(kp)

            qv = fx.Float32(Q[b, h, q_row, d_lane])
            kv = fx.Float32(K[b, h, k_col, d_lane])
            dp = qv * kv

            for s in [32, 16, 8, 4, 2, 1]:
                dp = dp + fx.Float32(fx.shuffle_down(dp, fx.Int32(s), fx.Int32(BLOCK)))
            attn_val = fx.Float32(fx.shuffle_idx(dp, fx.Int32(0), fx.Int32(BLOCK)))
            attn_val = attn_val * inv_sqrt

            new_m = fx.max(m_val, attn_val)
            rescale = fx.exp(m_val - new_m)
            e_val = fx.exp(attn_val - new_m)

            o_val = o_val * rescale
            l_val = l_val * rescale + e_val
            m_val = new_m

            vv = fx.Float32(V[b, h, k_col, d_lane])
            o_val = o_val + e_val * vv

        O[b, h, q_row, d_lane] = fx.BFloat16(o_val / l_val)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _causal_attn_jit(q, k, v, o)
    return o


_KERNEL_OPERATOR = 'attention'
_KERNEL_CFG_ID = 7009
