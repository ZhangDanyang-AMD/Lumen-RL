"""Drop-in replacement for ``miles_plugins.models.deepseek_v4.ops.hyper_connection``.

Same public API (``HCHeadParams``, ``DeepSeekV4HyperConnectionUtil``) as the miles
plugin, so the radixark/Megatron-LM PR #28 call sites in ``transformer_layer.py``
and ``transformer_block.py`` work unmodified, but backed by ``mhc_torch`` instead
of tile_kernels. No tile_kernels / tilelang / einops dependency.
"""

import os
import sys

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhc_torch as M  # noqa: E402

try:
    from megatron.core.transformer.module import MegatronModule as _Base
except Exception:  # megatron may be absent, or present but unimportable (TE)
    _Base = None

_HC_POST_MULT_VALUE = 2.0

# Probe knob: force the block output contiguous before it can reach a pipeline
# send. Set MHC_CONTIGUOUS=0 to reproduce the fork's view-returning behaviour.
_CONTIGUOUS_BLOCK_OUTPUT = os.environ.get('MHC_CONTIGUOUS', '1') != '0'


class HCHeadParams(torch.nn.Module if _Base is None else _Base):
    def __init__(self, config):
        if _Base is None:
            super().__init__()
        else:
            super().__init__(config)
        hc_mult = config.dsv4_hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))
        for p in [self.hc_head_fn, self.hc_head_base, self.hc_head_scale]:
            p._keep_fp32 = True

    def forward(self):
        raise NotImplementedError


class TorchHyperConnectionUtil:
    """Hyper-Connection helper backed by ``mhc_torch``."""

    def __init__(self, config):
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = config.dsv4_hc_mult
        self.hc_sinkhorn_iters = config.dsv4_hc_sinkhorn_iters
        self.hc_eps = config.dsv4_hc_eps

    def hc_pre_raw(
        self, x: Tensor, hc_fn: Tensor, hc_scale: Tensor, hc_base: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype = x.dtype
        x_bf16 = (x if x.dtype == torch.bfloat16 else x.bfloat16()).contiguous()

        mixes = M.mhc_pre_norm_fn(x_bf16, hc_fn, None, self.norm_eps)
        pre_mix, post, comb = M.mhc_pre_split_mixes(
            mixes, hc_scale, hc_base, self.hc_mult, _HC_POST_MULT_VALUE, self.hc_eps
        )
        comb = M.sinkhorn_normalize(comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps)
        layer_input = M.mhc_pre_apply_mix(x_bf16, pre_mix)
        return layer_input.to(dtype), post, comb

    def hc_post_raw(self, x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        dtype = x.dtype
        x_bf16 = (x if x.dtype == torch.bfloat16 else x.bfloat16()).contiguous()
        res_bf16 = (
            residual if residual.dtype == torch.bfloat16 else residual.bfloat16()
        ).contiguous()
        return M.mhc_post(x_bf16, res_bf16, post, comb).to(dtype)

    def hc_head_raw(
        self, x: Tensor, hc_fn: Tensor, hc_scale: Tensor, hc_base: Tensor
    ) -> Tensor:
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32
        dtype = x.dtype
        x_bf16 = (x if x.dtype == torch.bfloat16 else x.bfloat16()).contiguous()

        # The kernel path pads fn out to mhc_mult*(2+mhc_mult) rows and then slices
        # the first mhc_mult columns back off; the padding rows are zero, so using
        # the unpadded fn directly is equivalent.
        mixes = M.mhc_pre_norm_fn(x_bf16, hc_fn, None, self.norm_eps)
        scale = hc_scale.reshape(1) if hc_scale.numel() == 1 else hc_scale
        out_mix = M.mhc_head_compute_mix(mixes, scale, hc_base, self.hc_eps)
        layer_input = M.mhc_pre_apply_mix(x_bf16, out_mix.unsqueeze(-1))
        return layer_input.to(dtype)

    # --- Megatron call sites: [s, b, hc, d] <-> [b, s, hc, d] ---------------

    def layer_pre(
        self, hidden_states: Tensor, hc_fn: Tensor, hc_scale: Tensor, hc_base: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32
        x = hidden_states.transpose(0, 1)
        x, post, comb = self.hc_pre_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return x.transpose(0, 1), post, comb

    def layer_post(self, output_with_bias, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        assert isinstance(out, torch.Tensor)
        out = out.transpose(0, 1)
        residual_bshd = residual.transpose(0, 1)
        hidden_states = self.hc_post_raw(x=out, residual=residual_bshd, post=post, comb=comb)
        hidden_states = hidden_states.transpose(0, 1)
        # The [b,s,hc,d] -> [s,b,hc,d] rearrange is a view, so the block output that
        # crosses a pipeline boundary is non-contiguous and NCCL warns about it in
        # p2p. miles' einops.rearrange has the same property.
        if _CONTIGUOUS_BLOCK_OUTPUT:
            hidden_states = hidden_states.contiguous()
        return hidden_states

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        s, b, d = hidden_states.shape
        return hidden_states.unsqueeze(2).expand(s, b, self.hc_mult, d)

    def block_head(
        self, hidden_states: Tensor, hc_fn: Tensor, hc_scale: Tensor, hc_base: Tensor
    ) -> Tensor:
        x = hidden_states.transpose(0, 1)
        x = self.hc_head_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return x.transpose(0, 1)


DeepSeekV4HyperConnectionUtil = TorchHyperConnectionUtil
