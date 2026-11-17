"""PyTorch versions of the DeepSeek-V4.1 inference kernels the backbone depends on.

The released `inference/kernel.py` is tilelang. These follow its arithmetic step by step (fp32
compute, the same scale rounding, the same clamps) so that a model built on them can be checked
against the released reference, and so that the trainer can reproduce the rollout's activation
quantization as fake quantization with a straight-through gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pow2_ceil(x: Tensor) -> Tensor:
    """2 ** ceil(log2(x)) for positive normal fp32 x, as the kernels' exponent-bit trick computes
    it: an exact power of two stays put, anything else rounds up to the next one."""
    mantissa, exponent = torch.frexp(x)  # x = mantissa * 2**exponent, mantissa in [0.5, 1)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return torch.ldexp(torch.ones_like(x), exponent)


def _groups(x: Tensor, block_size: int) -> Tensor:
    if x.size(-1) % block_size:
        raise ValueError(f"last dim {x.size(-1)} is not a multiple of the block size {block_size}")
    return x.float().unflatten(-1, (-1, block_size))


def fp8_fake_quant(x: Tensor, block_size: int = 32, pow2_scale: bool = True) -> Tensor:
    """Block-wise E4M3 quantize-dequantize along the last dim (`act_quant(..., inplace=True)`).
    `pow2_scale` is the UE8M0 scale format: the scale rounds up to a power of two."""
    g = _groups(x, block_size)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = pow2_ceil(amax / FP8_MAX) if pow2_scale else amax / FP8_MAX
    q = (g / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).flatten(-2).to(x.dtype)


def round_e2m1(x: Tensor) -> Tensor:
    """Round fp32 values already inside [-6, 6] to the nearest E2M1 value, ties to even."""
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    a = x.abs()
    idx = torch.bucketize(a, mids)  # a == mids[k] lands on k, i.e. rounds down
    at_tie = a == mids[idx.clamp(max=mids.numel() - 1)]
    # grid points with an even mantissa are 0, 1, 2, 4 (indices 0, 2, 4, 6): ties at odd k go up
    idx = idx + (at_tie & (idx % 2 == 1)).long()
    return torch.copysign(grid[idx], x)


def fp4_fake_quant(x: Tensor, block_size: int = 32, scale_format: str = "e8m0") -> Tensor:
    """Block-wise E2M1 quantize-dequantize along the last dim (`fp4_act_quant(..., inplace=True)`).
    `e8m0`: power-of-two scales (MXFP4, the indexer). `e4m3`: E4M3 scales, used for the compressed
    KV with 16-element blocks."""
    g = _groups(x, block_size)
    amax = g.abs().amax(dim=-1, keepdim=True)
    if scale_format == "e4m3":
        scale = (amax.clamp_min(FP4_MAX * 2.0**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    elif scale_format == "e8m0":
        scale = pow2_ceil(amax.clamp_min(FP4_MAX * 2.0**-126) / FP4_MAX)
    else:
        raise ValueError(f"unknown FP4 scale format {scale_format!r}")
    q = round_e2m1((g / scale).clamp(-FP4_MAX, FP4_MAX))
    return (q * scale).flatten(-2).to(x.dtype)


def straight_through(x: Tensor, quantized: Tensor) -> Tensor:
    """Forward the quantized value, backward the identity (quantization-aware training)."""
    return x + (quantized - x).detach()


def sparse_attention(q: Tensor, kv: Tensor, attn_sink: Tensor, topk_idxs: Tensor, softmax_scale: float) -> Tensor:
    """Shared-KV sparse attention with a per-head sink logit (`sparse_attn`).

    q [B, S, H, D], kv [B, N, D], attn_sink [H] (fp32, a raw logit, not scaled), topk_idxs
    [B, S, T] int with -1 for an empty slot. Each query attends to the gathered kv rows plus the
    sink; a row with no valid slot outputs 0. Computed in fp32 by gathering, so a repeated index
    counts twice, as in the kernel.
    """
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp_min(0).long()
    rows = torch.gather(kv.float(), 1, idx.reshape(b, -1, 1).expand(-1, -1, d)).reshape(b, s, -1, d)
    scores = torch.einsum("bshd,bstd->bsht", q.float(), rows) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bsht,bstd->bshd", probs, rows).to(q.dtype)


def hc_split_sinkhorn(
    mixes: Tensor, hc_scale: Tensor, hc_base: Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6
) -> tuple[Tensor, Tensor, Tensor]:
    """Split one hyper-connection projection into pre [.., hc], post [.., hc] and a doubly
    stochastic comb [.., hc, hc] (`hc_split_sinkhorn`), in fp32."""
    m = mixes.float()
    hc = hc_mult
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
