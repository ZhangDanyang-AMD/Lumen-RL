import json

import pytest

from multi_tune_agent.request_parser import recognize_gemm_request
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
