"""Editable gfx950-only native MXFP4 dense GEMM wrapper.

The packed operands and E8M0 scales are consumed by AITER's production
``gemm_afp4wfp4`` implementation.  On gfx950 that implementation lowers its
``tl.dot_scaled(..., "e2m1", ...)`` operation to the architecture's native
scaled FP4 matrix instructions.  There is deliberately no fallback.
"""

from __future__ import annotations

from typing import Any

import torch


SUPPORTED_ARCH = "gfx950"
QUANT_BLOCK_SIZE = 32
OUTPUT_DTYPE = torch.bfloat16

# Editable tuning surface.  ``None`` asks AITER to select its production
# config.  An explicit config must contain the fields documented below.
GEMM_CONFIG: dict[str, Any] | None = None
# Example shape of an explicit AITER config (values are intentionally omitted):
# {
#     "BLOCK_SIZE_M": ...,
#     "BLOCK_SIZE_N": ...,
#     "BLOCK_SIZE_K": ...,
#     "GROUP_SIZE_M": ...,
#     "NUM_KSPLIT": ...,
#     "SPLITK_BLOCK_SIZE": ...,
#     "num_warps": ...,
#     "num_stages": ...,
#     "waves_per_eu": ...,
#     "matrix_instr_nonkdim": ...,
#     "cache_modifier": ...,
# }


def runtime_arch(device: torch.device | str | int | None = None) -> str:
    """Return the live HIP architecture without consulting build overrides."""

    if not torch.cuda.is_available():
        return "no HIP device"
    if isinstance(device, int):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device("cuda" if device is None else device)
    if resolved.type != "cuda":
        return resolved.type
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
    return (arch or "unknown").split(":", 1)[0].lower()


def require_gfx950(device: torch.device | str | int | None = None) -> None:
    """Reject unsupported hardware before importing or launching AITER."""

    detected = runtime_arch(device)
    if detected != SUPPORTED_ARCH:
        raise RuntimeError(
            f"native MXFP4 dense GEMM requires gfx950; detected {detected}"
        )


def _is_packed_e2m1_dtype(dtype: torch.dtype) -> bool:
    native = getattr(torch, "float4_e2m1fn_x2", None)
    return dtype == torch.uint8 or (native is not None and dtype == native)


def _is_e8m0_dtype(dtype: torch.dtype) -> bool:
    native = getattr(torch, "float8_e8m0fnu", None)
    return dtype == torch.uint8 or (native is not None and dtype == native)


def mxfp4_gemm(
    a_packed: torch.Tensor,
    w_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute ``dequant(A) @ dequant(W).T`` with native gfx950 MXFP4 GEMM.

    ``a_packed`` is ``[M, K/2]`` and ``w_packed`` is ``[N, K/2]``; each byte
    stores two E2M1 values.  Scales are row-major E8M0 tensors with shapes
    ``[M, K/32]`` and ``[N, K/32]``.  The result is BF16 ``[M, N]``.
    """

    require_gfx950(a_packed.device)

    tensors = (a_packed, w_packed, a_scales, w_scales)
    if any(t.device != a_packed.device for t in tensors):
        raise ValueError("packed operands and scales must be on the same device")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("packed operands and scales must all be rank-2")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("packed operands and scales must be contiguous")
    if not _is_packed_e2m1_dtype(a_packed.dtype) or not _is_packed_e2m1_dtype(
        w_packed.dtype
    ):
        raise TypeError("A and W must be packed E2M1 (float4_e2m1fn_x2 or uint8)")
    if not _is_e8m0_dtype(a_scales.dtype) or not _is_e8m0_dtype(w_scales.dtype):
        raise TypeError("A and W scales must be E8M0 (float8_e8m0fnu or uint8)")

    m, k_bytes = a_packed.shape
    n, w_k_bytes = w_packed.shape
    logical_k = 2 * k_bytes
    if k_bytes != w_k_bytes:
        raise ValueError("A and W packed K dimensions must match")
    if logical_k % QUANT_BLOCK_SIZE:
        raise ValueError(f"logical K must be divisible by {QUANT_BLOCK_SIZE}")
    expected_groups = logical_k // QUANT_BLOCK_SIZE
    if a_scales.shape != (m, expected_groups):
        raise ValueError(
            f"A scales must have shape {(m, expected_groups)}, got {tuple(a_scales.shape)}"
        )
    if w_scales.shape != (n, expected_groups):
        raise ValueError(
            f"W scales must have shape {(n, expected_groups)}, got {tuple(w_scales.shape)}"
        )
    if out is not None:
        if out.shape != (m, n) or out.dtype != OUTPUT_DTYPE:
            raise ValueError(f"out must be BF16 with shape {(m, n)}")
        if out.device != a_packed.device or not out.is_contiguous():
            raise ValueError("out must be contiguous and on the operands' device")

    # Import only after the hard architecture gate.  This prevents unsupported
    # hosts from triggering JIT setup or accidentally reaching a GPU launch.
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    selected = GEMM_CONFIG if config is None else config
    selected = None if selected is None else dict(selected)
    return gemm_afp4wfp4(
        a_packed,
        w_packed,
        a_scales,
        w_scales,
        dtype=OUTPUT_DTYPE,
        y=out,
        config=selected,
        skip_reduce=False,
    )
