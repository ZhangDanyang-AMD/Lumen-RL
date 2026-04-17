from __future__ import annotations

import torch

from lumenrl.quantization.fp8_config import FP8Config
from lumenrl.quantization.weight_quantizer import WeightQuantizer


def test_quantize_tensor_roundtrip() -> None:
    cfg = FP8Config(use_weight_pow2_scale=False)
    wq = WeightQuantizer(cfg)
    x = torch.randn(2, 256, dtype=torch.float32)
    q, scales = wq.quantize_tensor(x, block_size=128)
    dq = wq.dequantize_tensor(q, scales, block_size=128)
    err = (x - dq).abs().max().item()
    assert err < 0.25


def test_quantize_state_dict() -> None:
    cfg = FP8Config()
    wq = WeightQuantizer(cfg)
    sd = {"w": torch.randn(64, 128, dtype=torch.float32)}
    out = wq.quantize_state_dict(sd)
    assert "w" in out
    assert "w_fp8_scales" in out
    assert "w_fp8_meta" in out
    back = wq.dequantize_state_dict(out)
    assert back["w"].shape == (64, 128)
    assert back["w"].dtype == torch.bfloat16


def test_blockwise_shape() -> None:
    cfg = FP8Config()
    wq = WeightQuantizer(cfg)
    x = torch.randn(3, 4, 256, dtype=torch.float32)
    q, scales = wq.quantize_tensor(x, block_size=128)
    assert q.shape == x.shape
    assert scales.shape == (3, 4, 2)
