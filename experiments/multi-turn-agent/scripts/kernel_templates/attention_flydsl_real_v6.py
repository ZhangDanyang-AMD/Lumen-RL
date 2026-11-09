import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
SEQ_TILES = 128
D_TILES = 2


@flyc.jit
def _attn_dtile_jit(
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
        seq_pos = fx.block_idx.x
        lane = fx.thread_idx.x
        b0 = fx.Int32(0)
        h0 = fx.Int32(0)

        max_score = fx.Float32(-1e30)
        sum_exp = fx.Float32(0.0)

        for dt in fx.range_constexpr(D_TILES):
            d_base = fx.Int32(dt) * fx.Int32(BLOCK)
            d_idx = d_base + lane

            v_acc = fx.Float32(0.0)
            local_max = fx.Float32(-1e30)
            local_sum = fx.Float32(0.0)

            rsqrt_hd = fx.Float32(1.0 / (BLOCK * D_TILES) ** 0.5)

            for kp in fx.range_constexpr(SEQ_TILES):
                kv_idx = fx.Int32(kp)

                q_d = fx.Float32(Q[b0, h0, seq_pos, d_idx])
                k_d = fx.Float32(K[b0, h0, kv_idx, d_idx])
                partial = q_d * k_d

                for s in [32, 16, 8, 4, 2, 1]:
                    partial = partial + fx.Float32(fx.shuffle_down(partial, fx.Int32(s), fx.Int32(BLOCK)))
                sc = fx.Float32(fx.shuffle_idx(partial, fx.Int32(0), fx.Int32(BLOCK)))
                sc = sc * rsqrt_hd

                nm = fx.max(local_max, sc)
                corr = fx.exp(local_max - nm)
                ev = fx.exp(sc - nm)

                v_acc = v_acc * corr
                local_sum = local_sum * corr + ev
                local_max = nm

                v_d = fx.Float32(V[b0, h0, kv_idx, d_idx])
                v_acc = v_acc + ev * v_d

            O[b0, h0, seq_pos, d_idx] = fx.BFloat16(v_acc / local_sum)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _attn_dtile_jit(q, k, v, o)
    return o


_KERNEL_OPERATOR = 'attention'
_KERNEL_CFG_ID = 7012
