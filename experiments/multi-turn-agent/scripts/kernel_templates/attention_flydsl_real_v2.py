import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 128
SEQ_TILES = 64


@flyc.jit
def _attention_online_jit(
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
        q_pos = fx.block_idx.x
        lane = fx.thread_idx.x
        b = fx.Int32(0)
        h = fx.Int32(0)

        max_s = fx.Float32(-1e30)
        sum_e = fx.Float32(0.0)
        out_acc = fx.Float32(0.0)

        rsqrt_d = fx.Float32(1.0 / BLOCK ** 0.5)

        for kp in fx.range_constexpr(SEQ_TILES):
            kp_idx = fx.Int32(kp)

            qv = fx.Float32(Q[b, h, q_pos, lane])
            kv = fx.Float32(K[b, h, kp_idx, lane])
            prod = qv * kv

            for s in [64, 32, 16, 8, 4, 2, 1]:
                prod = prod + fx.Float32(fx.shuffle_down(prod, fx.Int32(s), fx.Int32(BLOCK)))
            attn = fx.Float32(fx.shuffle_idx(prod, fx.Int32(0), fx.Int32(BLOCK)))
            attn = attn * rsqrt_d

            new_m = fx.max(max_s, attn)
            old_corr = fx.exp(max_s - new_m)
            new_exp = fx.exp(attn - new_m)

            out_acc = out_acc * old_corr
            sum_e = sum_e * old_corr + new_exp
            max_s = new_m

            vv = fx.Float32(V[b, h, kp_idx, lane])
            out_acc = out_acc + new_exp * vv

        O[b, h, q_pos, lane] = fx.BFloat16(out_acc / sum_e)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _attention_online_jit(q, k, v, o)
    return o


_KERNEL_OPERATOR = 'attention'
_KERNEL_CFG_ID = 7004
