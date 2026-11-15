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


def _ipc_export_kwargs(monkeypatch, *, expert_tp_slice: bool, backend: str = "vllm", rollout_tp: int = 8) -> dict:
    import sys
    from types import ModuleType

    import lumenrl.engine.inference.bucketed_weight_transfer as bwt

    fake_ray = ModuleType("ray")
    fake_ray.get_runtime_context = lambda: SimpleNamespace(get_job_id=lambda: "job")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    class Sender:
        stats: dict = {}

        def __init__(self, **kwargs):
            pass

        async def async_send_weights(self, weights):
            list(weights)

    monkeypatch.setattr(bwt, "BucketedWeightSender", Sender)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    asked: dict = {}

    def get_per_tensor_param(**kwargs):
        asked.update(kwargs)
        return [], None

    worker = SimpleNamespace(
        _engine=SimpleNamespace(get_per_tensor_param=get_per_tensor_param),
        rank=11,
        config={
            "policy": {"generation_backend": backend,
                       "generation": {"vllm_cfg": {"tensor_parallel_size": rollout_tp}}},
            "weight_sync": {"expert_tp_slice": expert_tp_slice},
        },
        _sent_nbytes=lambda param, keep_fp32: 0,
    )
    LumenActorWorker.update_weights_ipc_send(worker)
    return asked


def test_ipc_send_asks_for_the_rollout_tp_ranks_expert_slice(monkeypatch) -> None:
    assert _ipc_export_kwargs(monkeypatch, expert_tp_slice=True)["expert_tp_slice"] == (3, 8)


def test_ipc_send_asks_for_whole_experts_unless_configured_for_vllm(monkeypatch) -> None:
    assert _ipc_export_kwargs(monkeypatch, expert_tp_slice=False)["expert_tp_slice"] is None
    assert _ipc_export_kwargs(monkeypatch, expert_tp_slice=True, backend="atom")["expert_tp_slice"] is None
    assert _ipc_export_kwargs(monkeypatch, expert_tp_slice=True, rollout_tp=1)["expert_tp_slice"] is None
