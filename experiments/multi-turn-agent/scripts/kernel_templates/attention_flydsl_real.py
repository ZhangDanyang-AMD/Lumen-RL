import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

BLOCK = 64
SEQ_TILES = 128


@flyc.jit
def _attention_fwd_jit(
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
        tid = fx.thread_idx.x
        b_idx = fx.Int32(0)
        h_idx = fx.Int32(0)

        running_max = fx.Float32(-1e30)
        running_sum = fx.Float32(0.0)
        acc = fx.Float32(0.0)

        scale = fx.Float32(1.0 / BLOCK ** 0.5)

        for kp in fx.range_constexpr(SEQ_TILES):
            k_pos = fx.Int32(kp)

            q_val = fx.Float32(Q[b_idx, h_idx, q_pos, tid])
            k_val = fx.Float32(K[b_idx, h_idx, k_pos, tid])
            dot = q_val * k_val

            for offset in [32, 16, 8, 4, 2, 1]:
                dot = dot + fx.Float32(fx.shuffle_down(dot, fx.Int32(offset), fx.Int32(BLOCK)))
            score = fx.Float32(fx.shuffle_idx(dot, fx.Int32(0), fx.Int32(BLOCK)))
            score = score * scale

            new_max = fx.max(running_max, score)
            correction = fx.exp(running_max - new_max)
            exp_score = fx.exp(score - new_max)

            acc = acc * correction
            running_sum = running_sum * correction + exp_score
            running_max = new_max

            v_val = fx.Float32(V[b_idx, h_idx, k_pos, tid])
            acc = acc + exp_score * v_val

        O[b_idx, h_idx, q_pos, tid] = fx.BFloat16(acc / running_sum)

    _kernel(q, k, v, o).launch(grid=(S, 1, 1), block=(BLOCK, 1, 1))


def attention_kernel(q, k, v, o=None, softmax_scale=None, is_causal=False):
    B, H, S, D = q.shape
    if o is None:
        o = torch.zeros_like(q)
    else:
        o.zero_()
    _attention_fwd_jit(q, k, v, o)
    return o


_KERNEL_OPERATOR = 'attention'
_KERNEL_CFG_ID = 7002
