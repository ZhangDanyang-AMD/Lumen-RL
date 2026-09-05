import json

import pytest

from multi_tune_agent.request_parser import (
    recognize_gemm_request,
    recognize_kernel_request,
)
from multi_tune_agent.runtime import ModelTurn


class JsonBackend:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, tools=()):
        text = json.dumps(self.payload)
        return ModelTurn(
            text,
            {"role": "assistant", "content": text},
            [],
            [],
            {},
        )


class FailedBackend:
    def generate(self, messages, tools=()):
        raise RuntimeError("model unavailable")


def test_model_recognizes_noisy_gemm_request():
    request = "MI308 做矩阵乘法，参数大概 3/128/128，用 FlyDSL"
    result = recognize_gemm_request(
        request,
        JsonBackend(
            {
                "operator": "gemm",
                "target_gpu": "MI308",
                "dtype": "fp16",
                "m": 3,
                "n": 128,
                "k": 128,
                "language": "flydsl",
                "confidence": 0.92,
            }
        ),
    )
    assert result["target_gpu"] == "MI308X"
    assert result["language"] == "flydsl"
    assert (result["m"], result["n"], result["k"]) == (3, 128, 128)
    assert result["recognition"] == "model"


def test_model_cannot_invent_unmentioned_dimensions():
    with pytest.raises(ValueError, match="could not recognize"):
        recognize_gemm_request(
            "MI308 GEMM but dimensions are missing",
            JsonBackend(
                {
                    "operator": "gemm",
                    "target_gpu": "MI308X",
                    "dtype": "fp16",
                    "m": 1,
                    "n": 128,
                    "k": 128,
                    "language": "triton",
                    "confidence": 0.9,
                }
            ),
        )


def test_deterministic_parser_is_available_when_model_is_down():
    result = recognize_gemm_request(
        "MI308 GEMM MNK分别 shi 3/128/128", FailedBackend()
    )
    assert (result["m"], result["n"], result["k"]) == (3, 128, 128)
    assert result["recognition"] == "deterministic_fallback"


def test_complete_fp8_kernel_with_independent_scale_granularities():
    result = recognize_kernel_request(
        "MI355 GEMM m=16 n=32 k=64 FP8, input per-token, weight per-channel, HIP",
        JsonBackend(
            {
                "operator": "gemm",
                "target_gpu": "MI355",
                "format": "float8",
                "input_dtype": "fp8",
                "weight_dtype": "fp8",
                "output_dtype": "fp16",
                "input_scale_granularity": "per-token",
                "weight_scale_granularity": "per-channel",
                "block_size": None,
                "dimensions": {"m": 16, "n": 32, "k": 64},
                "shapes": [[16, 32, 64]],
                "language": "hip",
                "confidence": 0.97,
            }
        ),
    )
    assert result["target_gpu"] == "gfx950"
    assert result["format"] == "fp8"
    assert result["input_scale_granularity"] == "per_token"
    assert result["weight_scale_granularity"] == "per_channel"
    assert result["shapes"] == [[16, 32, 64]]
    assert result["missing_fields"] == []


def test_quantized_kernel_reports_missing_scale_fields():
    result = recognize_kernel_request(
        "MI300 softmax FP8 shape 8x2048",
        JsonBackend(
            {
                "operator": "softmax",
                "target_gpu": "MI300",
                "format": "fp8",
                "dimensions": {},
                "shapes": [[8, 2048]],
                "confidence": 0.9,
            }
        ),
    )
    assert result["missing_fields"] == [
        "input_scale_granularity",
        "weight_scale_granularity",
    ]


def test_gfx942_fp8_gemm_implies_standard_scale_contract():
    result = recognize_kernel_request(
        "MI308 FP8 GEMM M=1 N=128 K=128 in Triton",
        JsonBackend(
            {
                "operator": "gemm",
                "target_gpu": "MI308",
                "format": "fp8",
                "dimensions": {"m": 1, "n": 128, "k": 128},
                "shapes": [[1, 128, 128]],
                "language": "triton",
                "confidence": 0.95,
            }
        ),
    )

    assert result["target_gpu"] == "gfx942"
    assert result["input_scale_granularity"] == "per_token"
    assert result["weight_scale_granularity"] == "per_channel"
    assert set(result["format_implied_fields"]) == {
        "input_scale_granularity",
        "weight_scale_granularity",
    }
    assert result["missing_fields"] == []


def test_native_mxfp4_implies_scales_and_block_size():
    result = recognize_kernel_request(
        "MI325 GEMM m=4 n=128 k=256 native MXFP4",
        JsonBackend(
            {
                "operator": "gemm",
                "target_gpu": "MI325",
                "format": "mxfp4",
                "dimensions": {"m": 4, "n": 128, "k": 256},
                "shapes": [[4, 128, 256]],
                "confidence": 0.95,
            }
        ),
    )
    assert result["input_scale_granularity"] == "per_block"
    assert result["weight_scale_granularity"] == "per_block"
    assert result["block_size"] == 32
    assert set(result["format_implied_fields"]) == {
        "input_scale_granularity",
        "weight_scale_granularity",
        "block_size",
    }
    assert "block_size" not in result["explicit_fields"]
    assert result["missing_fields"] == []


def test_chinese_suffixes_and_mxfp8_request_are_recognized():
    request = "请在MI308 上写MXFP8 GEMMkernel，MKN分别是1/128/128；flydsl写"
    result = recognize_kernel_request(
        request,
        JsonBackend(
            {
                "operator": "gemm",
                "target_gpu": "gfx942",
                "format": "mxfp8",
                "dimensions": {},
                "shapes": [[1, 128], [128, 128]],
                "language": "flydsl",
                "evidence": {
                    "operator": "GEMMkernel",
                    "target_gpu": "MI308",
                    "format": "MXFP8",
                    "language": "flydsl",
                },
                "confidence": 0.95,
            }
        ),
    )

    assert result["target_gpu"] == "gfx942"
    assert result["format"] == "mxfp8"
    assert result["language"] == "flydsl"
    assert result["shapes"] == [[1, 128, 128]]
    assert result["input_scale_granularity"] == "per_block"
    assert result["weight_scale_granularity"] == "per_block"
    assert result["block_size"] == 32
    assert result["missing_fields"] == []
    assert result["model_evidence"]["target_gpu"] == "MI308"


def test_arbitrary_attention_dimensions_are_preserved():
    request = (
        "MI350 attention batch=2 heads=16 seq_len=4096 head_dim=128, "
        "input BF16 output FP16 in Triton"
    )
    result = recognize_kernel_request(
        request,
        JsonBackend(
            {
                "operator": "attention",
                "target_gpu": "MI350",
                "format": "bf16",
                "input_dtype": "bfloat16",
                "output_dtype": "float16",
                "dimensions": {
                    "batch": 2,
                    "heads": 16,
                    "seq_len": 4096,
                    "head_dim": 128,
                },
                "shapes": [],
                "language": "triton",
                "confidence": 0.93,
            }
        ),
    )
    assert result["operator"] == "attention"
    assert result["dimensions"] == {
        "batch": 2,
        "heads": 16,
        "seq_len": 4096,
        "head_dim": 128,
    }
    assert result["input_dtype"] == "bf16"
    assert result["output_dtype"] == "fp16"
    assert result["missing_fields"] == []


def test_hallucinated_contract_fields_are_rejected_and_language_defaults():
    result = recognize_kernel_request(
        "MI300 softmax shape 8x128",
        JsonBackend(
            {
                "operator": "softmax",
                "target_gpu": "MI355",
                "format": "fp8",
                "input_dtype": "int8",
                "input_scale_granularity": "per-token",
                "block_size": 32,
                "dimensions": {},
                "shapes": [[8, 128]],
                "language": "hip",
                "confidence": 0.9,
            }
        ),
    )
    assert result["target_gpu"] == "gfx942"
    assert result["format"] is None
    assert result["input_dtype"] is None
    assert result["input_scale_granularity"] is None
    assert result["block_size"] is None
    assert result["language"] == "triton"
    assert result["missing_fields"] == []


def test_noisy_operator_can_come_from_model_but_values_need_evidence():
    result = recognize_kernel_request(
        "MI308 atenshun, batch=2 seq=1024, bf16, per tensor",
        JsonBackend(
            {
                "operator": "attention",
                "target_gpu": "MI308",
                "format": "bf16",
                "dimensions": {"batch": 2, "seq": 1024},
                "shapes": [],
                "language": None,
                "confidence": 0.82,
            }
        ),
    )
    assert result["operator"] == "attention"
    assert result["target_gpu"] == "gfx942"
    assert result["format"] == "bf16"
    assert result["language"] == "triton"


def test_failed_model_returns_partial_deterministic_result():
    result = recognize_kernel_request("please optimize softmax", FailedBackend())
    assert result["operator"] == "softmax"
    assert set(result["missing_fields"]) == {"target_gpu", "shape_or_dimensions"}


def test_negated_language_does_not_override_requested_backend():
    result = recognize_kernel_request(
        "gfx942 GEMM M=1 N=128 K=128 FP16. Use HIP, not Triton.",
        FailedBackend(),
    )
    assert result["language"] == "hip"
    assert result["recognition"] == "deterministic_fallback"


def test_only_empty_kernel_request_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        recognize_kernel_request(" \n ", FailedBackend())
