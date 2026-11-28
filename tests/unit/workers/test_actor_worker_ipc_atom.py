"""ATOM over the colocated CUDA-IPC weight sync.

ATOM's runner maps the staging buffer on the first bucket and keeps that mapping
until the last, so the buffer cannot grow mid-sync. The sender used to size its
bucket to the largest tensor by iterating the engine's weights first, which
drained the Megatron-native engine's one-shot stream: nothing was sent, and ATOM
generated from the weights it had discarded for the update. The sender now
streams a plain bucket and ATOM's receiver stages oversized tensors in a buffer
sized by ``atom_cfg.ipc_staging_mb``.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lumenrl.workers.actor_worker import LumenActorWorker

PARAMS = [
    ("model.embed_tokens.weight", torch.randn(64, 32).bfloat16()),
    ("model.layers.0.input_layernorm.weight", torch.randn(32).bfloat16()),
]


class _Sender:
    """Captures the stream and the sender's settings instead of using a socket."""

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


def test_one_shot_stream_is_sent_whole(monkeypatch) -> None:
    engine = SimpleNamespace(get_per_tensor_param=lambda **kw: ((p for p in PARAMS), None))
    _worker(monkeypatch, engine).update_weights_ipc_send(bucket_size_mb=512)
    assert _Sender.captured == [n for n, _ in PARAMS]
    assert _Sender.kwargs["bucket_size_mb"] == 512


# --- ATOM receiver -----------------------------------------------------------

KIB = 1 << 10


class _Socket:
    """REP socket that replays the sender's messages."""

    def __init__(self, messages):
        self.messages = list(messages)

    def recv_pyobj(self):
        return self.messages.pop(0)

    def send(self, _data):
        pass

    def setsockopt(self, *_a):
        pass

    def connect(self, _addr):
        pass

    def close(self):
        pass


def _meta(name, t, offset, handle=None):
    return {"name": name, "shape": t.shape, "dtype": t.dtype, "offset": offset, "handle": handle}


def _receive(monkeypatch, messages, staging_mb):
    """Run ATOM's IPC receive loop; return what the runner read for each tensor."""
    from lumenrl.engine.inference import atom_moe_weight_sync, atom_ray_server

    fake_zmq = SimpleNamespace(
        REP=0, LINGER=0,
        Context=lambda: SimpleNamespace(socket=lambda _kind: _Socket(messages), term=lambda: None),
    )
    monkeypatch.setattr(atom_ray_server, "zmq", fake_zmq)
    # Handles stand for the tensors themselves.
    fake_sync = ModuleType("atom.rollout.weight_sync")
    fake_sync.rebuild_ipc_handle = lambda handle, device_id=None: handle
    monkeypatch.setitem(sys.modules, "atom.rollout.weight_sync", fake_sync)
    monkeypatch.setattr("torch.multiprocessing.reductions.reduce_tensor", lambda t: t)
    monkeypatch.setattr(atom_moe_weight_sync, "assert_bucket_fully_applied", lambda *a, **k: None)
    monkeypatch.setattr(atom_moe_weight_sync, "atom_routes_fused_experts", lambda: True)

    loaded: dict[str, torch.Tensor] = {}

    def runner(_cmd, ipc_handles, bucket_meta, **_kw):
        staging = ipc_handles[0]
        for name, m in bucket_meta.items():
            dtype = getattr(torch, m["dtype"].replace("torch.", ""))
            raw = staging[m["offset"]: m["offset"] + m["nbytes"]]
            loaded[name] = raw.view(dtype).view(m["shape"]).clone()
        return []

    server = atom_ray_server.ATOMRayServer(
        model_name="m", engine_kwargs={}, replica_rank=0, ipc_staging_mb=staging_mb,
    )
    server.engine = SimpleNamespace(
        core_mgr=SimpleNamespace(broadcast_utility_command_sync=runner),
    )
    server._update_weights_from_ipc_sync()
    return loaded


def _sync_with_large_tensor_second():
    """Sender with a 64 KiB bucket: one small bucket, then a 512 KiB direct send."""
    small = torch.randn(16, 16, device="cuda").bfloat16()
    large = torch.randn(512, 256, device="cuda").float()
    sender_buf = torch.zeros(64 * KIB, dtype=torch.uint8, device="cuda")
    sender_buf[: small.nbytes].copy_(small.view(-1).view(torch.uint8))
    messages = [
        sender_buf,
        {"bucket_meta": {"small": _meta("small", small, 0)}, "is_last": False},
        {"bucket_meta": {"large": _meta("large", large, 0, handle=large)}, "is_last": True},
    ]
    return messages, small, large


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_staging_holds_a_tensor_larger_than_the_sender_bucket(monkeypatch) -> None:
    messages, small, large = _sync_with_large_tensor_second()
    loaded = _receive(monkeypatch, messages, staging_mb=1)
    assert torch.equal(loaded["small"], small)
    assert torch.equal(loaded["large"], large)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tensor_larger_than_staging_after_first_bucket_is_refused(monkeypatch) -> None:
    messages, _, _ = _sync_with_large_tensor_second()
    with pytest.raises(RuntimeError, match="ipc_staging_mb"):
        _receive(monkeypatch, messages, staging_mb=0)
