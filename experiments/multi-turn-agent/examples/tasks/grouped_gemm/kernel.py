"""Editable single-launch grouped GEMM Triton kernel for MI300X."""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    group_m_ptr,
    c_ptr,
    stride_ae: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M_MAX: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M_MAX, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_expert = grid_m * grid_n
    expert = pid // tiles_per_expert
    tile = pid % tiles_per_expert
    pid_m = tile // grid_n
    pid_n = tile % grid_n
    valid_m = tl.load(group_m_ptr + expert)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr
        + expert * stride_ae
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < valid_m) & (k_start + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_start + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = (
        c_ptr
        + expert * stride_ce
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < valid_m) & (offs_n[None, :] < N),
    )


def grouped_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    group_m: torch.Tensor,
) -> torch.Tensor:
    """Run variable-token-count GEMMs for all experts in one GPU launch.

    ``activations`` is padded [E,M_max,K], ``weights`` is [E,K,N], and
    ``group_m[e]`` is the routed token count for expert ``e``.
    """

    if activations.ndim != 3 or weights.ndim != 3:
        raise ValueError("activations and weights must be rank-3")
    experts, m_max, k = activations.shape
    if weights.shape[0] != experts or weights.shape[1] != k:
        raise ValueError("expert and K dimensions must match")
    if group_m.shape != (experts,) or group_m.dtype != torch.int32:
        raise TypeError("group_m must be int32[E]")
    if activations.dtype != torch.float16 or weights.dtype != torch.float16:
        raise TypeError("grouped GEMM expects FP16 operands")
    n = weights.shape[2]
    out = torch.empty(
        (experts, m_max, n), device=activations.device, dtype=torch.float16
    )
    block_m, block_n, block_k = 32, 64, 32
    grid = (
        experts * triton.cdiv(m_max, block_m) * triton.cdiv(n, block_n),
    )
    _grouped_gemm_kernel[grid](
        activations,
        weights,
        group_m,
        out,
        *activations.stride(),
        *weights.stride(),
        *out.stride(),
        M_MAX=m_max,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        waves_per_eu=2,
    )
    return out
