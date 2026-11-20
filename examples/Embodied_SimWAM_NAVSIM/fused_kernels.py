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


@triton.jit
def _gate_residual_kernel(x_ptr, g_ptr, r_ptr, o_ptr, seq_len, n_cols, sxb, sxs, sgb, sgs, srb, srs,
                          ROUND_T: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + b * sxb + s * sxs + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + b * sgb + s * sgs + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + b * srb + s * srs + cols, mask=mask, other=0.0).to(tl.float32)
    t = g * r
    if ROUND_T:
        # torch rounds gate * residual to bf16 before the add
        t = t.to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + row * n_cols + cols, (x + t).to(o_ptr.dtype.element_ty), mask=mask)


class _GateResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, residual, save):
        bsz, seq, n = x.shape
        t_dtype = torch.promote_types(gate.dtype, residual.dtype)
        out = torch.empty((bsz, seq, n), dtype=torch.promote_types(x.dtype, t_dtype), device=x.device)
        g, gb, gs = _bcast_strides(gate, x.shape)
        r, rb, rs = _bcast_strides(residual, x.shape)
        block = min(1024, triton.next_power_of_2(n))
        # enable_fp_fusion=False: no fused multiply-add where torch rounds the product first
        _gate_residual_kernel[(bsz * seq, triton.cdiv(n, block))](
            x, g, r, out, seq, n, x.stride(0), x.stride(1), gb, gs, rb, rs,
            ROUND_T=t_dtype == torch.bfloat16, BLOCK=block, enable_fp_fusion=False)
        if save:
            ctx.save_for_backward(gate, residual)
        ctx.t_dtype = t_dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        # the ops autograd runs for x + gate * residual; returning full-size gradients leaves the reduction over
        # broadcast dimensions and the casts to the inputs' dtypes to the engine, as for the original ops
        gate, residual = ctx.saved_tensors
        grad_t = grad.to(ctx.t_dtype)
        dgate = (grad_t * residual).to(gate.dtype) if ctx.needs_input_grad[1] else None
        dres = (grad_t * gate).to(residual.dtype) if ctx.needs_input_grad[2] else None
        return grad, dgate, dres, None


def gate_residual(x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """``x + gate * residual`` in one launch with torch's roundings (the product rounded to its dtype before the
    add), so the output is bit-identical to the two ops; the backward runs the same ops autograd would. ``x`` is
    ``[B, S, D]``; ``gate`` / ``residual`` broadcast to it (3-D, last dim dense)."""
    save = torch.is_grad_enabled() and (x.requires_grad or gate.requires_grad or residual.requires_grad)
    return _GateResidual.apply(x, gate, residual, save)


@triton.jit
def _lora_up_add_kernel(y_ptr, b_ptr, o_ptr, n_rows, n_cols, stride_y, scale,
                        R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, R)
    rmask = rows < n_rows
    cmask = cols < n_cols
    mid = tl.load(y_ptr + rows[:, None] * stride_y + n_cols + rr[None, :], mask=rmask[:, None], other=0.0)
    bt = tl.load(b_ptr + cols[None, :] * R + rr[:, None], mask=cmask[None, :], other=0.0)
    up = tl.dot(mid, bt)
    mask = rmask[:, None] & cmask[None, :]
    base = tl.load(y_ptr + rows[:, None] * stride_y + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + rows[:, None] * n_cols + cols[None, :], (base + scale * up).to(o_ptr.dtype.element_ty), mask=mask)


class _LoRALinearCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_a, lora_b, wa, bias_a, scale, gemm, save):
        r = lora_a.shape[0]
        n = wa.shape[0] - r
        x2 = x.reshape(-1, x.shape[-1])
        y = gemm(x2, wa, bias_a)
        m = y.shape[0]
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        block_m = 16 if m <= 16 else 32 if m <= 64 else 64
        _lora_up_add_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, 128))](
            y, lora_b, out, m, n, y.stride(0), scale, R=r, BLOCK_M=block_m, BLOCK_N=128)
        if save:
            # the scaled copy of x @ A.T: the B gradient then needs no separate scaling kernel
            ctx.save_for_backward(x2, y[:, n:] * scale, lora_b, wa)
        ctx.scale, ctx.x_shape = scale, x.shape
        return out.view(*x.shape[:-1], n)

    @staticmethod
    def backward(ctx, grad):
        x2, mid_s, lora_b, wa = ctx.saved_tensors
        n = lora_b.shape[0]
        g2 = grad.reshape(-1, n)
        # scaled after or before the rank-r GEMMs instead of on the full gradient: the same values for a
        # power-of-two scale
        dmid = (g2 @ lora_b) * ctx.scale
        dx = da = db = None
        if ctx.needs_input_grad[0]:
            dx = (g2 @ wa[:n] + dmid @ wa[n:]).view(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            da = dmid.t() @ x2
        if ctx.needs_input_grad[2]:
            db = g2.t() @ mid_s
        return dx, da, db, None, None, None, None, None


def lora_linear(x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, wa: torch.Tensor,
                bias_a, scale: float, gemm, save=None) -> torch.Tensor:
    """``x @ W.T + bias + scale * (x @ A.T) @ B.T`` where ``wa`` is ``[W; A]`` in one tensor (``lora_a`` is its
    last ``r`` rows) and ``bias_a`` the bias padded with ``r`` zeros: one GEMM ``gemm(x2d, wa, bias_a)`` gives the
    base output and ``x @ A.T`` side by side, then one kernel adds ``scale * mid @ B.T`` in fp32 and rounds once.
    The backward runs torch GEMMs; ``lora_a`` / ``lora_b`` receive the adapter gradients. ``save`` says whether
    to keep what the backward needs; by default whenever an input requires grad, as autograd's
    ``needs_input_grad`` (which follows ``requires_grad`` even under ``no_grad``)."""
    if save is None:
        save = x.requires_grad or lora_a.requires_grad or lora_b.requires_grad
    return _LoRALinearCat.apply(x, lora_a, lora_b, wa, bias_a, scale, gemm, save)
