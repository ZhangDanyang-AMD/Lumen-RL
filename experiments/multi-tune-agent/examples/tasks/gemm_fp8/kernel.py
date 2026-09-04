"""Editable Triton FP8 A8W8 dense GEMM kernel for AMD gfx942."""

import torch
import triton
import triton.language as tl


# Deliberately explicit launch knobs: GEAK may tune these values in this file.
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_M = 8
NUM_WARPS = 4
NUM_STAGES = 2
WAVES_PER_EU = 2

FP8_DTYPE = getattr(torch, "float8_e4m3fnuz", None)


@triton.jit
def _fp8_a8w8_gemm_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
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
    GROUP_M: tl.constexpr,
):
    """Multiply FP8 matrices, then apply row/column scales in FP32."""

    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)

    # Group M tiles so neighboring programs reuse the same B panel in cache.
    programs_per_group = GROUP_M * grid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    row_scale = tl.load(
        a_scale_ptr + offs_m, mask=offs_m < m, other=0.0
    ).to(tl.float32)
    column_scale = tl.load(
        b_scale_ptr + offs_n, mask=offs_n < n, other=0.0
    ).to(tl.float32)
    output = accumulator * row_scale[:, None] * column_scale[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        output.to(tl.float16),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def _gfx_arch(device: torch.device) -> str:
    properties = torch.cuda.get_device_properties(device)
    gcn_arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if gcn_arch:
        return gcn_arch
    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    if major is not None and minor is not None:
        return "sm_%s%s" % (major, minor)
    return str(getattr(properties, "name", "unknown"))


def require_gfx942_fp8(device: torch.device | str | None = None) -> torch.device:
    """Reject unsupported hosts before allocation or Triton compilation."""

    if FP8_DTYPE is None:
        raise RuntimeError(
            "This task requires torch.float8_e4m3fnuz; the installed PyTorch "
            "build does not expose that dtype."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This task requires a ROCm CUDA device with gfx942 architecture; "
            "no CUDA/ROCm device is available."
        )
    resolved = torch.device("cuda" if device is None else device)
    if resolved.type != "cuda":
        raise RuntimeError("FP8 A8W8 GEMM requires a CUDA/ROCm device.")
    architecture = _gfx_arch(resolved)
    if architecture != "gfx942":
        shown = architecture or "unknown"
        raise RuntimeError(
            "FP8 A8W8 GEMM supports gfx942 only; detected %s." % shown
        )
    return resolved


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int, int]:
    require_gfx942_fp8(a.device)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype != FP8_DTYPE or b.dtype != FP8_DTYPE:
        raise TypeError("A and B must both use torch.float8_e4m3fnuz")
    if a.device != b.device:
        raise ValueError("A and B must be on the same gfx942 device")

    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0 or k == 0:
        raise ValueError("M, N, and K must all be positive")
    expected_scales = ((m,), (n,))
    if a_scale.shape != expected_scales[0] or b_scale.shape != expected_scales[1]:
        raise ValueError("expected activation scales [M] and weight scales [N]")
    for name, scale in (("activation", a_scale), ("weight", b_scale)):
        if scale.dtype != torch.float32:
            raise TypeError("%s scales must be float32" % name)
        if scale.device != a.device or not scale.is_contiguous():
            raise ValueError(
                "%s scales must be contiguous and on the operand device" % name
            )
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError("%s scales must be finite and positive" % name)
    if out is not None:
        if out.shape != (m, n) or out.dtype != torch.float16:
            raise ValueError("out must have shape [M,N] and dtype float16")
        if out.device != a.device or not out.is_contiguous():
            raise ValueError("out must be contiguous and on the operand device")
    return m, n, k


def fp8_a8w8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Compute ``(A_fp8 * a_scale) @ (B_fp8 * b_scale)`` into FP16.

    ``a_scale`` has one FP32 value per activation row. ``b_scale`` has one
    FP32 value per output column. B may be a transposed row-quantized weight
    tensor; arbitrary positive strides are passed through to Triton.
    """

    if validate:
        m, n, k = _validate_inputs(a, b, a_scale, b_scale, out)
    else:
        m, k = a.shape
        n = b.shape[1]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.float16)

    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    _fp8_a8w8_gemm_kernel[grid](
        a,
        b,
        a_scale,
        b_scale,
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        waves_per_eu=WAVES_PER_EU,
    )
    return out
