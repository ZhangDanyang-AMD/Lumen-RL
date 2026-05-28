from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from lumenrl.engine.training.megatron_backend import MegatronBackend
from lumenrl.workers.actor_worker import LumenActorWorker


def test_build_model_uses_stub_when_tiny_requested() -> None:
    model = MegatronBackend.build_model(
        model_name="",
        config={"megatron_cfg": {"use_tiny_lm": True, "hidden_size": 32, "vocab_size": 64}},
    )
    assert isinstance(model, nn.Module)
    out = model(torch.randint(0, 64, (2, 4), dtype=torch.long))
    assert out.shape == (2, 4, 64)


def test_build_model_prefers_hf_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyHFModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grad_ckpt_enabled = False

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
            self.grad_ckpt_enabled = bool(gradient_checkpointing_kwargs)

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> DummyHFModel:
            _ = args
            assert kwargs["attn_implementation"] == "sdpa"
            return DummyHFModel()

    monkeypatch.setattr(
        "lumenrl.engine.training.megatron_backend.AutoModelForCausalLM",
        DummyAutoModel,
    )

    model = MegatronBackend.build_model(
        model_name="dummy/model",
        config={"megatron_cfg": {"use_tiny_lm": False, "hf_attn_patch": False}},
    )
    assert isinstance(model, DummyHFModel)
    assert model.grad_ckpt_enabled is True


def test_apply_lumen_spec_returns_same_model_when_no_features() -> None:
    model = nn.Linear(8, 8)
    out = MegatronBackend.apply_lumen_spec(model, megatron_config={"lumen": {}})
    assert out is model


def test_apply_lumen_spec_fp8_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError):
        MegatronBackend.apply_lumen_spec(nn.Linear(4, 4), megatron_config={"lumen": {"fp8": True}})


def test_actor_worker_megatron_path_keeps_api_compatible() -> None:
    worker = LumenActorWorker(
        rank=0,
        world_size=1,
        config={
            "policy": {
                "training_backend": "megatron",
                "training": {"megatron_cfg": {"use_tiny_lm": True, "hidden_size": 16, "vocab_size": 32}},
                "lr": 1e-4,
            }
        },
    )
    worker.init_model()
    assert worker._model is not None
    assert worker._optimizer is not None
