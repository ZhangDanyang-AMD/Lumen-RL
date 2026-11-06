"""policy.training.max_memory_fraction caps the actor's caching allocator.

A colocated rollout engine and RCCL allocate outside the trainer's allocator;
uncapped, its cache grows into every free byte and they OOM.
"""

from __future__ import annotations

import pytest
import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.workers.actor_worker import apply_memory_cap


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda f, device=None: seen.append((f, device)))
    return seen


def test_cap_is_applied_to_the_current_device(calls) -> None:
    apply_memory_cap({"max_memory_fraction": 0.7})
    assert calls == [(0.7, 3)]


def test_no_cap_by_default(calls) -> None:
    apply_memory_cap({})
    apply_memory_cap({"max_memory_fraction": None})
    assert calls == []
    assert LumenRLConfig().policy.training.max_memory_fraction is None


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_out_of_range_cap_is_rejected(calls, bad) -> None:
    with pytest.raises(ValueError, match="max_memory_fraction"):
        apply_memory_cap({"max_memory_fraction": bad})
