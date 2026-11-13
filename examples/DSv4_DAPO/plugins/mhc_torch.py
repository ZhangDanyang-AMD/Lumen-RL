"""Pure-PyTorch mHC (DeepSeek-V4 hyper-connection), autograd only.

Reimplements ``tile_kernels.modeling.mhc.ops`` so that the radixark/Megatron-LM
PR #28 call sites can run without tile_kernels / tilelang. Semantics were read
off the tilelang kernels in tile_kernels 1.0.0:

  mhc/norm_fn_kernel.py            _mhc_pre_norm_fn_fwd_{mul,norm}
  mhc/pre_split_mixes_kernel.py    _mhc_pre_split_mixes_fwd
  mhc/sinkhorn_kernel.py           _mhc_sinkhorn_fwd
  mhc/pre_apply_mix_kernel.py      _mhc_pre_apply_mix_fwd
  mhc/post_kernel.py               _mhc_post_fwd
  mhc/head_compute_mix_kernel.py   _mhc_head_compute_mix_fwd

Backward is left to autograd; the kernels' hand-written backwards are exact
derivatives of these forwards, so the two agree up to accumulation order.

Note ``mhc_pre_norm_fn`` RMS-normalizes over the *whole* flattened
``mhc_mult * hidden`` vector (the kernel is instantiated with n_rms_group=1,
rms_group_size=mhc_mult*hidden), not per hyper-connection stream.
"""

import torch
from torch import Tensor

_HC_POST_MULT_VALUE = 2.0


def round_to_tf32(x: Tensor) -> Tensor:
    """tile_kernels rounds the fp32 GEMM operands to tf32 before the matmul.

    The bit-twiddle is a dtype-reinterpret view and therefore not differentiable,
    so it is applied straight-through: rounding has derivative 1 almost everywhere.
    """
    with torch.no_grad():
        rounded = (x.detach().view(torch.int32) + 0x1000).view(torch.float32)
    return x + (rounded - x.detach())


def mhc_pre_norm_fn(
    residual: Tensor,
    fn: Tensor,
    norm_weight: Tensor | None = None,
    norm_eps: float = 1e-6,
    emulate_tf32: bool = True,
) -> Tensor:
    """[..., mhc, hidden] bf16 x [mult3, mhc*hidden] fp32 -> [..., mult3] fp32."""
    assert fn.dtype == torch.float32
    if norm_weight is not None:
        fn = fn * norm_weight
    if emulate_tf32:
        fn = round_to_tf32(fn)

    mult3, mhc_hidden = fn.shape
    x = residual.reshape(*residual.shape[:-2], mhc_hidden).float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
    return (x @ fn.t()) * rms


def mhc_pre_split_mixes(
    mixes: Tensor,
    scale: Tensor,
    base: Tensor,
    mhc_mult: int,
    post_mult_value: float,
    pre_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """[..., mult3] -> pre [..., m, 1], post [..., m, 1], comb [..., m, m]."""
    m = mhc_mult
    outer = mixes.shape[:-1]

    pre = torch.sigmoid(mixes[..., :m] * scale[0] + base[:m]) + pre_eps
    post = torch.sigmoid(mixes[..., m : 2 * m] * scale[1] + base[m : 2 * m]) * post_mult_value
    comb = mixes[..., 2 * m :] * scale[2] + base[2 * m :]

    return pre.reshape(*outer, m, 1), post.reshape(*outer, m, 1), comb.reshape(*outer, m, m)


def sinkhorn_normalize(comb: Tensor, repeat: int = 20, eps: float = 1e-6) -> Tensor:
    """Doubly-stochastic normalization; first row pass is a softmax, then plain sums."""
    x = torch.softmax(comb, dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


def mhc_pre_apply_mix(residual: Tensor, mix: Tensor) -> Tensor:
    """[..., m, h] bf16 x [..., m, 1] fp32 -> [..., h] bf16."""
    return (residual.float() * mix).sum(dim=-2).to(residual.dtype)


def mhc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """out[..., o, h] = post[..., o] * x[..., h] + sum_i comb[..., i, o] * residual[..., i, h]."""
    mixed = torch.einsum('...io,...ih->...oh', comb, residual.float())
    return (mixed + post * x.float().unsqueeze(-2)).to(residual.dtype)


def mhc_head_compute_mix(mix: Tensor, scale: Tensor, base: Tensor, pre_eps: float) -> Tensor:
    return torch.sigmoid(mix * scale[0] + base) + pre_eps
