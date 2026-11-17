"""Stand-in for the released tilelang `kernel.py`, so the reference `model.py` next to this file
runs on CPU in plain PyTorch. Only the paths a BF16-weight model takes are provided; the FP8/FP4
GEMMs are for quantized weights and raise."""

from lumenrl.models.deepseek_v41 import ops


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    if not inplace:
        raise NotImplementedError("only the in-place (quantize-dequantize) form is needed for BF16 weights")
    x.copy_(ops.fp8_fake_quant(x, block_size, pow2_scale=scale_fmt is not None))
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    if not inplace:
        raise NotImplementedError("only the in-place (quantize-dequantize) form is needed for BF16 weights")
    import torch

    fmt = "e4m3" if scale_dtype == torch.float8_e4m3fn else "e8m0"
    x.copy_(ops.fp4_fake_quant(x, block_size, fmt))
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    return ops.sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    return ops.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("FP8 weights are not used by the reference tests")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("FP4 weights are not used by the reference tests")
