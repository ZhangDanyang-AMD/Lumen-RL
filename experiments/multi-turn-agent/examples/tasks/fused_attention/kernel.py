"""Editable fused causal-attention Triton kernel for MI300X."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // N_HEADS
    off_h = off_bh % N_HEADS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    )

    # Online softmax keeps the complete QK^T / softmax / V pipeline in one launch.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    qk_scale = SM_SCALE * 1.4426950408889634

    for start_n in range(0, N_CTX, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(
            k_base + cols[:, None] * stride_ks + offs_d[None, :] * stride_kd
        )
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(offs_m[:, None] >= cols[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)

        v = tl.load(
            v_base + cols[:, None] * stride_vs + offs_d[None, :] * stride_vd
        )
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = acc / l_i[:, None]
    out_base = out_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        out.to(tl.float16),
    )


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal scaled-dot-product attention for contiguous [B,H,S,D] FP16 tensors."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must have the same [B,H,S,D] shape")
    if q.dtype != torch.float16 or q.shape[-1] != 64:
        raise TypeError("attention task expects FP16 with head_dim=64")
    batch, heads, sequence, head_dim = q.shape
    if sequence % 64:
        raise ValueError("sequence length must be a multiple of 64")
    out = torch.empty_like(q)
    grid = (triton.cdiv(sequence, 64), batch * heads)
    _fused_attention_kernel[grid](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        N_HEADS=heads,
        N_CTX=sequence,
        HEAD_DIM=head_dim,
        SM_SCALE=head_dim**-0.5,
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=4,
        waves_per_eu=2,
    )
    return out
