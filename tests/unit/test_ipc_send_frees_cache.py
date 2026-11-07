"""The IPC weight sync returns the actor's cached allocator memory first.

The colocated vLLM worker allocates its online-FP8 reload buffers during the
sync, in another process; it cannot reclaim the trainer's cache, so a trainer
still holding its training-peak cache leaves it too little memory (seen at
202.7 of 206 GB on MI300X). The RDMA path already empties the cache here.
"""

from __future__ import annotations

import pytest
import ray
import torch

from lumenrl.workers.actor_worker import LumenActorWorker


class _Stop(Exception):
    pass


def test_ipc_send_empties_the_cache_before_gathering_weights(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset_peak"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(ray, "get_runtime_context",
                        lambda: type("C", (), {"get_job_id": lambda self: "j"})())

    class _Engine:
        def get_per_tensor_param(self, **kw):
            events.append("gather")
            raise _Stop

    w = object.__new__(LumenActorWorker)
    w._engine = _Engine()
    w.rank = 0
    w.config = {}
    w._peak_carry_alloc = 0.0
    w._peak_carry_reserved = 0.0

    with pytest.raises(_Stop):
        w.update_weights_ipc_send()
    # Training peaks are read (reset_peak) before the cache is returned.
    assert events == ["reset_peak", "empty_cache", "gather"]
