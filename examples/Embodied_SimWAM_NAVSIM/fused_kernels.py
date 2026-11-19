"""Triton kernels used by plugin.py when the matching switches are on (imported on first use)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, g_ptr, w_ptr, dx_ptr, stride_x, stride_g, stride_dx, n_cols, eps,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    gw = g * w
    dot = tl.sum(gw * x, axis=0) / n_cols
    dx = r * gw - x * (r * r * r) * dot
    tl.store(dx_ptr + row * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def rmsnorm_backward(x: torch.Tensor, grad: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Input gradient of ``x * rsqrt(mean(x^2) + eps) * weight`` over the last dim, in fp32, one launch."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    g2 = grad.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if g2.stride(-1) != 1:
        g2 = g2.contiguous()
    dx = torch.empty((x2.shape[0], n), dtype=x.dtype, device=x.device)
    _rmsnorm_bwd_kernel[(x2.shape[0],)](x2, g2, weight, dx, x2.stride(0), g2.stride(0), dx.stride(0), n, eps,
                                        BLOCK=triton.next_power_of_2(n))
    return dx.view(x.shape)


@triton.jit
def _rope_kernel(x_ptr, f_ptr, out_ptr, seq_len, stride_xb, stride_xs, stride_fs, stride_fj, stride_ob, stride_os,
                 half_dim, n_pairs, sign, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // seq_len
    s = pid % seq_len
    p = tl.arange(0, BLOCK)
    mask = p < n_pairs
    j = p % half_dim
    xa = tl.load(x_ptr + b * stride_xb + s * stride_xs + 2 * p, mask=mask, other=0.0).to(tl.float32)
    xb = tl.load(x_ptr + b * stride_xb + s * stride_xs + 2 * p + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(f_ptr + s * stride_fs + j * stride_fj, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(f_ptr + s * stride_fs + j * stride_fj + 1, mask=mask, other=0.0).to(tl.float32) * sign
    oa = xa * c - xb * sn
    ob = xa * sn + xb * c
    o = out_ptr + b * stride_ob + s * stride_os + 2 * p
    tl.store(o, oa.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(o + 1, ob.to(out_ptr.dtype.element_ty), mask=mask)


def rope(x: torch.Tensor, freqs_real: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate consecutive pairs of ``x`` [B, S, H*D] by ``freqs_real`` [S, 1, D/2, 2] (cos, sin), in fp32.

    ``inverse`` rotates by the conjugate (the backward of the forward rotation).
    """
    bsz, seq, width = x.shape
    half = freqs_real.shape[-2]
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty((bsz, seq, width), dtype=x.dtype, device=x.device)
    n_pairs = width // 2
    _rope_kernel[(bsz * seq,)](x, freqs_real, out, seq, x.stride(0), x.stride(1), freqs_real.stride(0),
                               freqs_real.stride(-2), out.stride(0), out.stride(1), half, n_pairs,
                               -1.0 if inverse else 1.0, BLOCK=triton.next_power_of_2(n_pairs))
    return out
