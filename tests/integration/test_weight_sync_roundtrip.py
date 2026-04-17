"""Weight sync manager vs actor / sink workers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from lumenrl.core.protocol import DataProto
from lumenrl.engine.training.weight_sync import WeightSyncManager
from lumenrl.workers.actor_worker import LumenActorWorker


pytestmark = pytest.mark.gpu


@pytest.fixture
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class _WeightSink:
    """In-process destination recording the last pushed state dict."""

    def __init__(self) -> None:
        self.sd: dict[str, torch.Tensor] = {}

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.sd = {k: v.clone() for k, v in state_dict.items()}


def test_weight_sync_changes_output(small_dense_model: dict[str, Any], _require_cuda: None) -> None:
    """Policy outputs change after a train step; sync transfers updated weights."""
    actor = LumenActorWorker(rank=0, world_size=1, config=small_dense_model)
    actor.init_model()

    b, t = 2, 12
    batch = DataProto(
        tensors={
            "input_ids": torch.randint(0, 256, (b, t), dtype=torch.long),
            "advantages": torch.ones(b, t, dtype=torch.float32),
        },
    )

    before = actor.compute_log_probs(batch)
    actor.train_step(batch)
    after = actor.compute_log_probs(batch)

    assert not torch.allclose(before["log_probs"], after["log_probs"], atol=1e-6, rtol=0.0)

    sink = _WeightSink()
    mgr = WeightSyncManager({"mode": "ray_object_store"})
    mgr.transfer([actor], [sink])
    assert sink.sd
    keys_actor = set(actor.get_state_dict().keys())
    assert keys_actor == set(sink.sd.keys())


def test_state_dict_integrity(small_dense_model: dict[str, Any], _require_cuda: None) -> None:
    """Keys and shapes are preserved across an in-process WeightSyncManager transfer."""
    actor = LumenActorWorker(rank=0, world_size=1, config=small_dense_model)
    actor.init_model()
    pre = actor.get_state_dict()

    b, t = 2, 12
    batch = DataProto(
        tensors={
            "input_ids": torch.randint(0, 256, (b, t), dtype=torch.long),
            "advantages": torch.ones(b, t, dtype=torch.float32),
        },
    )
    actor.train_step(batch)
    post = actor.get_state_dict()
    assert set(pre.keys()) == set(post.keys())

    sink = _WeightSink()
    WeightSyncManager({"mode": "ray_object_store"}).transfer([actor], [sink])

    assert set(sink.sd.keys()) == set(post.keys())
    for k in post:
        assert sink.sd[k].shape == post[k].shape
        assert sink.sd[k].dtype == post[k].dtype
