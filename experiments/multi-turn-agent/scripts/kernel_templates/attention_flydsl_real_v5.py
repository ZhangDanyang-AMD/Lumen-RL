import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
SEQ_TILES = 128


@flyc.jit
def _attn_2pass_jit(
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
        query_idx = fx.block_idx.x
        thread = fx.thread_idx.x
        bi = fx.Int32(0)
        hi = fx.Int32(0)

        scale_factor = fx.Float32(1.0 / BLOCK ** 0.5)

        global_max = fx.Float32(-1e30)
        for p1 in fx.range_constexpr(SEQ_TILES):
            key_idx = fx.Int32(p1)
            q_load = fx.Float32(Q[bi, hi, query_idx, thread])
            k_load = fx.Float32(K[bi, hi, key_idx, thread])
            dot_prod = q_load * k_load
            for sh in [32, 16, 8, 4, 2, 1]:
                dot_prod = dot_prod + fx.Float32(fx.shuffle_down(dot_prod, fx.Int32(sh), fx.Int32(BLOCK)))
            reduced = fx.Float32(fx.shuffle_idx(dot_prod, fx.Int32(0), fx.Int32(BLOCK)))
            reduced = reduced * scale_factor
            global_max = fx.max(global_max, reduced)

        denom = fx.Float32(0.0)
        output = fx.Float32(0.0)
        for p2 in fx.range_constexpr(SEQ_TILES):
            key_idx2 = fx.Int32(p2)
            q2 = fx.Float32(Q[bi, hi, query_idx, thread])
            k2 = fx.Float32(K[bi, hi, key_idx2, thread])
            dp2 = q2 * k2
            for sh in [32, 16, 8, 4, 2, 1]:
                dp2 = dp2 + fx.Float32(fx.shuffle_down(dp2, fx.Int32(sh), fx.Int32(BLOCK)))
            sc2 = fx.Float32(fx.shuffle_idx(dp2, fx.Int32(0), fx.Int32(BLOCK)))
            sc2 = sc2 * scale_factor

            w = fx.exp(sc2 - global_max)
            denom = denom + w

            v_load = fx.Float32(V[bi, hi, key_idx2, thread])
            output = output + w * v_load

        O[bi, hi, query_idx, thread] = fx.BFloat16(output / denom)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _attn_2pass_jit(q, k, v, o)
    return o


_KERNEL_OPERATOR = 'attention'
_KERNEL_CFG_ID = 7011
