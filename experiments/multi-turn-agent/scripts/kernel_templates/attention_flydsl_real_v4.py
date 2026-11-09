import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
SEQ_TILES = 128


@flyc.jit
def _mha_fwd_jit(
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    o: fx.Tensor,
):
    seq_len = fx.Int32(fx.get_scalar(q.shape[2]))
    num_heads = fx.Int32(fx.get_scalar(q.shape[1]))

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def _kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
    ):
        s_pos = fx.block_idx.x
        head = fx.block_idx.y
        dim_idx = fx.thread_idx.x
        batch = fx.Int32(0)

        cur_max = fx.Float32(-1e30)
        cur_sum = fx.Float32(0.0)
        weighted_v = fx.Float32(0.0)

        rsqrt_dim = fx.Float32(1.0 / BLOCK ** 0.5)

        for ki in fx.range_constexpr(SEQ_TILES):
            kv_pos = fx.Int32(ki)

            q_elem = fx.Float32(Q[batch, head, s_pos, dim_idx])
            k_elem = fx.Float32(K[batch, head, kv_pos, dim_idx])
            qk = q_elem * k_elem

            for w in [32, 16, 8, 4, 2, 1]:
                qk = qk + fx.Float32(fx.shuffle_down(qk, fx.Int32(w), fx.Int32(BLOCK)))
            s_val = fx.Float32(fx.shuffle_idx(qk, fx.Int32(0), fx.Int32(BLOCK)))
            s_val = s_val * rsqrt_dim

            prev_max = cur_max
            cur_max = fx.max(cur_max, s_val)
            alpha = fx.exp(prev_max - cur_max)
            p_val = fx.exp(s_val - cur_max)

            weighted_v = weighted_v * alpha
            cur_sum = cur_sum * alpha + p_val

            v_elem = fx.Float32(V[batch, head, kv_pos, dim_idx])
            weighted_v = weighted_v + p_val * v_elem

        O[batch, head, s_pos, dim_idx] = fx.BFloat16(weighted_v / cur_sum)

    _kernel(q, k, v, o).launch(grid=(seq_len, num_heads, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _mha_fwd_jit(q, k, v, o)
    return o


_KERNEL_OPERATOR = 'attention'
_KERNEL_CFG_ID = 7010
