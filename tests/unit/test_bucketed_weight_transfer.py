"""Tests for the production bucketed weight-transfer transport."""

from __future__ import annotations

import asyncio
import threading

import torch

from lumenrl.engine.inference.bucketed_weight_transfer import (
    BucketedWeightReceiver,
    BucketedWeightSender,
)


def test_shared_memory_roundtrip_across_multiple_buckets(
    tmp_path, monkeypatch,
) -> None:
    """Preserve tensor names, values, shapes, and dtypes across bucket reuse."""
    monkeypatch.setenv("LUMEN_WEIGHT_TRANSFER_ZMQ_TIMEOUT_MS", "5000")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    weights = [
        ("model.large", torch.arange(200_000, dtype=torch.float32)),
        ("model.small", torch.arange(100_000, dtype=torch.float32).reshape(200, 500)),
        ("model.ids", torch.arange(257, dtype=torch.int64)),
    ]
    socket_path = tmp_path / "weight-transfer.sock"
    handle = f"ipc://{socket_path}"
    sender = BucketedWeightSender(handle, bucket_size_mb=1, use_shm=True)
    receiver = BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)

    sender_errors: list[BaseException] = []

    def _send() -> None:
        try:
            asyncio.run(sender.async_send_weights(weights))
        except BaseException as exc:  # surfaced in the test thread below
            sender_errors.append(exc)

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

    received: dict[str, torch.Tensor] = {}

    def _capture(bucket: list[tuple[str, torch.Tensor]]) -> None:
        # Receiver tensors view a staging buffer that the next bucket overwrites.
        received.update({name: tensor.clone() for name, tensor in bucket})

    receiver.receive_weights(on_bucket_received=_capture)
    thread.join(timeout=10)

    assert not thread.is_alive(), "weight sender did not finish"
    assert not sender_errors
    assert set(received) == {name for name, _ in weights}
    for name, expected in weights:
        actual = received[name]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
