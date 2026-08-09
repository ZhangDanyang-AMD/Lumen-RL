from __future__ import annotations

import pytest
import torch

from lumenrl.engine.inference import fp8_weight_quantizer as quantizer


def _expand_block_scales(
    scales: torch.Tensor,
    shape: tuple[int, int],
    block_size: tuple[int, int],
) -> torch.Tensor:
    return scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(
        block_size[1],
        dim=1,
    )[: shape[0], : shape[1]]


def test_per_block_quantizer_returns_vllm_dequant_scale() -> None:
    weight = torch.tensor(
        [[-4.0, -2.0], [1.0, 3.0]],
        dtype=torch.float32,
    )

    fp8_weight, weight_scale_inv = quantizer._per_block_cast_to_fp8(
        weight,
        block_size=(2, 2),
    )

    expected_scale = weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
    assert weight_scale_inv.item() == pytest.approx(expected_scale.item())
    reconstructed = fp8_weight.float() * _expand_block_scales(
        weight_scale_inv,
        tuple(weight.shape),
        (2, 2),
    )
    torch.testing.assert_close(reconstructed, weight, rtol=0.06, atol=0.06)


def test_rocm_per_block_quantizer_casts_directly_to_fnuz(monkeypatch) -> None:
    monkeypatch.setattr(quantizer, "_is_rocm", lambda: True)
    weight = torch.tensor(
        [[-4.0, -2.0], [1.0, 3.0]],
        dtype=torch.float32,
    )

    fp8_weight, weight_scale_inv = quantizer._per_block_cast_to_fp8(
        weight,
        block_size=(2, 2),
    )

    assert fp8_weight.dtype == torch.float8_e4m3fnuz
    # vLLM intentionally uses 224 instead of torch.finfo(...).max (240)
    # for ROCm fnuz accuracy.
    expected_scale = weight.abs().max() / 224.0
    assert weight_scale_inv.item() == pytest.approx(expected_scale.item())
    reconstructed = fp8_weight.float() * _expand_block_scales(
        weight_scale_inv,
        tuple(weight.shape),
        (2, 2),
    )
    torch.testing.assert_close(reconstructed, weight, rtol=0.06, atol=0.06)


def test_quantizer_replaces_only_terminal_weight_suffix() -> None:
    assert hasattr(quantizer, "_scale_name_for_weight")
    assert quantizer._scale_name_for_weight(
        "model.layers.0.self_attn.indexer.weights_proj.weight"
    ) == "model.layers.0.self_attn.indexer.weights_proj.weight_scale_inv"


@pytest.mark.parametrize(
    "name",
    [
        "embed.weight",
        "head.weight",
        "layers.0.ffn.gate.weight",
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wgate.weight",
        "layers.2.attn.indexer.compressor.wkv.weight",
        "layers.2.attn.indexer.compressor.wgate.weight",
        "layers.2.attn.indexer.weights_proj.weight",
    ],
)
def test_quantizer_skips_redhat_non_fp8_weights(name: str) -> None:
    assert quantizer._should_quantize(name) is False
