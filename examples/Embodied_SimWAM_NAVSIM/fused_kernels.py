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


@triton.jit
def _ln_modulate_fwd_kernel(x_ptr, sh_ptr, sc_ptr, y_ptr, mean_ptr, rstd_ptr, seq_len, n_cols, eps,
                            sxb, sxs, shb, shs, scb, scs, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + b * sxb + s * sxs + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / n_cols + eps)
    sc = tl.load(sc_ptr + b * scb + s * scs + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(sh_ptr + b * shb + s * shs + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * (1.0 + sc) + sh
    tl.store(y_ptr + row * n_cols + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _ln_modulate_bwd_kernel(dy_ptr, x_ptr, sc_ptr, mean_ptr, rstd_ptr, dx_ptr, seq_len, n_cols,
                            sxb, sxs, scb, scs, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + b * sxb + s * sxs + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(sc_ptr + b * scb + s * scs + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    g = dy * (1.0 + sc)
    c1 = tl.sum(g * xhat, axis=0) / n_cols
    c2 = tl.sum(g, axis=0) / n_cols
    dx = (g - c2 - xhat * c1) * rstd
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _bcast_strides(t: torch.Tensor, shape) -> tuple:
    """Batch and sequence strides of ``t`` broadcast to ``shape`` [B, S, D] (0 on broadcast dims); last dim dense."""
    t = t.expand(shape)
    return t, t.stride(0), t.stride(1)


class _LayerNormModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, eps):
        bsz, seq, n = x.shape
        sh, shb, shs = _bcast_strides(shift, x.shape)
        sc, scb, scs = _bcast_strides(scale, x.shape)
        y = torch.empty((bsz, seq, n), dtype=x.dtype, device=x.device)
        mean = torch.empty(bsz * seq, dtype=torch.float32, device=x.device)
        rstd = torch.empty_like(mean)
        _ln_modulate_fwd_kernel[(bsz * seq,)](x, sh, sc, y, mean, rstd, seq, n, eps, x.stride(0), x.stride(1),
                                              shb, shs, scb, scs, BLOCK=triton.next_power_of_2(n))
        ctx.save_for_backward(x, shift, scale, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, shift, scale, mean, rstd = ctx.saved_tensors
        bsz, seq, n = x.shape
        dy = dy.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            sc, scb, scs = _bcast_strides(scale, x.shape)
            dx = torch.empty((bsz, seq, n), dtype=x.dtype, device=x.device)
            _ln_modulate_bwd_kernel[(bsz * seq,)](dy, x, sc, mean, rstd, dx, seq, n, x.stride(0), x.stride(1),
                                                  scb, scs, BLOCK=triton.next_power_of_2(n))
        dshift = dscale = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            xhat = (x.float() - mean.view(bsz, seq, 1)) * rstd.view(bsz, seq, 1)
            red = [i for i in range(3) if shift.shape[i] == 1 and x.shape[i] != 1] if shift.dim() == 3 else None
            if ctx.needs_input_grad[1]:
                dshift = dy.float().sum(dim=red, keepdim=True).to(shift.dtype) if red else dy.to(shift.dtype)
            if ctx.needs_input_grad[2]:
                red_s = [i for i in range(3) if scale.shape[i] == 1 and x.shape[i] != 1] if scale.dim() == 3 else None
                p = dy.float() * xhat
                dscale = p.sum(dim=red_s, keepdim=True).to(scale.dtype) if red_s else p.to(scale.dtype)
        return dx, dshift, dscale, None


def layernorm_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """``layer_norm(x, eps) * (1 + scale) + shift`` without affine weights, over the last dim of ``x`` [B, S, D],
    in fp32 and one launch each way; ``shift`` / ``scale`` broadcast to ``x`` (3-D, last dim dense)."""
    return _LayerNormModulate.apply(x, shift, scale, eps)
