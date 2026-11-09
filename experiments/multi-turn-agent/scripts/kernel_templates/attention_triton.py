import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["S", "D"],
)
@triton.jit
def _fmha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    B, H, S, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < S

    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(m_mask[:, None] & n_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m_i = m_new

    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]

    o_ptrs = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])


def flash_attention(q, k, v, sm_scale=None):
    B, H, S, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)
    _fmha_kernel[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S, D,
    )
    return o
