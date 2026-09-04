"""Editable Triton FP16 dense GEMM kernel for MI300X."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)

    # Group row tiles so neighbouring programs reuse the same B panel in L2.
    group_m = 8
    width = group_m * grid_n
    group_id = pid // width
    first_m = group_id * group_m
    group_size = tl.minimum(grid_m - first_m, group_m)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m) & (k_start + offs_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_start + offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b`` for contiguous FP16 matrices."""

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm expects A[M,K] and B[K,N]")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("GEMM task expects FP16 operands")
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _gemm_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        waves_per_eu=2,
    )
    return out
