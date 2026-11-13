"""The dtype send_weights_rdma puts on the wire for an actor's weights."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import lumenrl.engine.inference.rdma_weight_transfer as rdma
from lumenrl.workers.actor_worker import LumenActorWorker

_PARAMS = [
    ("model.embed_tokens.weight", torch.zeros(4, 2, dtype=torch.float32)),
    ("model.norm.weight", torch.zeros(2, dtype=torch.bfloat16)),
]


def _sent_dtypes(monkeypatch, **engine_attrs) -> dict[str, torch.dtype]:
    sent: dict[str, torch.dtype] = {}

    def fake_send(group, weights, *, bucket_size_bytes, version):
        sent.update((name, tensor.dtype) for name, tensor in weights)
        return {"buckets": 1.0}

    monkeypatch.setattr(rdma, "send_weight_stream", fake_send)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    worker = SimpleNamespace(
        _engine=SimpleNamespace(
            get_per_tensor_param=lambda: (iter(_PARAMS), None), **engine_attrs,
        ),
        rank=0,
        _rdma_weight_group=object(),
    )
    LumenActorWorker.send_weights_rdma(worker, version=1, bucket_size_mb=1)
    return sent


def test_fp32_weights_go_out_as_bf16(monkeypatch) -> None:
    monkeypatch.delenv("LUMENRL_SYNC_FP32", raising=False)
    assert _sent_dtypes(monkeypatch) == {
        "model.embed_tokens.weight": torch.bfloat16,
        "model.norm.weight": torch.bfloat16,
    }


def test_sync_fp32_keeps_them_fp32(monkeypatch) -> None:
    monkeypatch.setenv("LUMENRL_SYNC_FP32", "1")
    assert _sent_dtypes(monkeypatch)["model.embed_tokens.weight"] == torch.float32


def test_an_engine_whose_fp32_is_deliberate_sends_it_as_fp32(monkeypatch) -> None:
    monkeypatch.delenv("LUMENRL_SYNC_FP32", raising=False)
    assert _sent_dtypes(monkeypatch, exports_fp32_on_purpose=True) == {
        "model.embed_tokens.weight": torch.float32,
        "model.norm.weight": torch.bfloat16,
    }


def test_the_megatron_engine_declares_its_fp32_deliberate() -> None:
    from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine

    assert MegatronNativeEngine.exports_fp32_on_purpose is True
