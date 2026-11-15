"""ATOM bucket sizing over the colocated CUDA-IPC weight sync.

ATOM needs the bucket to hold the largest tensor before the first one is sent.
The sender used to find it by iterating the engine's weights, which drained the
Megatron-native engine's one-shot stream: nothing was sent, and ATOM generated
from the weights it had discarded for the update.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lumenrl.engine.training.megatron_native_engine import (
    MegatronNativeEngine,
    StageOutput,
    StageRecipe,
)
from lumenrl.workers.actor_worker import LumenActorWorker

PARAMS = [
    ("model.embed_tokens.weight", torch.randn(64, 32).bfloat16()),
    ("model.layers.0.input_layernorm.weight", torch.randn(32).bfloat16()),
]
LARGEST = 64 * 32 * 2


class _Sender:
    """Captures the stream and the bucket floor instead of using a socket."""

    captured: list[str] = []
    kwargs: dict = {}

    def __init__(self, **kwargs):
        _Sender.kwargs = kwargs
        self.stats: dict[str, float] = {}
        _Sender.captured = []

    async def async_send_weights(self, stream):
        for name, _ in stream:
            _Sender.captured.append(name)
        return True


def _worker(monkeypatch, engine) -> LumenActorWorker:
    fake_ray = ModuleType("ray")
    fake_ray.get_runtime_context = lambda: SimpleNamespace(get_job_id=lambda: "job-0")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    fake_transfer = ModuleType("lumenrl.engine.inference.bucketed_weight_transfer")
    fake_transfer.BucketedWeightSender = _Sender
    monkeypatch.setitem(
        sys.modules, "lumenrl.engine.inference.bucketed_weight_transfer", fake_transfer,
    )
    w = object.__new__(LumenActorWorker)
    w.rank = 0
    w.config = {"policy": {"generation_backend": "atom"}}
    w._peak_carry_alloc = 0.0
    w._peak_carry_reserved = 0.0
    w._engine = engine
    return w


class _StreamingEngine:
    """Like the Megatron-native engine: a stream readable once, plus its bound."""

    def get_per_tensor_param(self, **kw):
        return (p for p in PARAMS), None

    def max_export_tensor_nbytes(self, stack_experts: bool = False) -> int:
        return LARGEST


def test_one_shot_stream_is_sent_whole(monkeypatch) -> None:
    _worker(monkeypatch, _StreamingEngine()).update_weights_ipc_send()
    assert _Sender.captured == [n for n, _ in PARAMS]
    assert _Sender.kwargs["min_bucket_bytes"] == LARGEST


def test_reiterable_weights_are_sized_by_reading_them(monkeypatch) -> None:
    engine = SimpleNamespace(get_per_tensor_param=lambda **kw: (PARAMS, None))
    _worker(monkeypatch, engine).update_weights_ipc_send()
    assert _Sender.captured == [n for n, _ in PARAMS]
    assert _Sender.kwargs["min_bucket_bytes"] == LARGEST


def test_one_shot_stream_without_a_bound_is_refused(monkeypatch) -> None:
    engine = SimpleNamespace(get_per_tensor_param=lambda **kw: (iter(PARAMS), None))
    with pytest.raises(TypeError, match="one-shot iterator"):
        _worker(monkeypatch, engine).update_weights_ipc_send()


def _output(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> StageOutput:
    return StageOutput(
        name=name, shape=shape, dtype=dtype, pp_stage=0, ep_owner=None,
        local_e=None, tp_split_dim=None, tp_size=1, etp_size=1, gate_up=False,
    )


def test_engine_bound_is_the_largest_planned_tensor(monkeypatch) -> None:
    plans = {
        False: [
            StageRecipe("embedding", "pass", (_output("e", (151936, 4096), torch.bfloat16),)),
            StageRecipe("qkv", "split", (
                _output("q", (4096, 4096), torch.bfloat16),
                _output("k", (1024, 4096), torch.bfloat16),
            )),
            StageRecipe("bias", "pass", (_output("b", (128,), torch.float32),)),
        ],
        True: [StageRecipe("experts", "pass", (
            _output("w", (8, 1536, 2048), torch.bfloat16),
        ), stacked=True)],
    }
    engine = object.__new__(MegatronNativeEngine)
    monkeypatch.setattr(
        engine, "_stage_plan", lambda stack_experts=False: plans[stack_experts], raising=False,
    )
    assert engine.max_export_tensor_nbytes() == 151936 * 4096 * 2
    assert engine.max_export_tensor_nbytes(stack_experts=True) == 8 * 1536 * 2048 * 2
