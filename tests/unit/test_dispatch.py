from __future__ import annotations

import torch

from lumenrl.controller.dispatch import DispatchMode, collect_proto, dispatch_proto
from lumenrl.controller.worker_group_factory import resolve_worker_class
from lumenrl.core.protocol import DataProto


def test_dispatch_even_split() -> None:
    p = DataProto(tensors={"x": torch.arange(12).view(4, 3).float()})
    chunks = dispatch_proto(p, num_workers=2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    recomposed = torch.cat([c["x"] for c in chunks], dim=0)
    assert torch.allclose(recomposed, p["x"])


def test_collect_merges() -> None:
    parts = [
        DataProto(tensors={"x": torch.zeros(2, 1)}),
        DataProto(tensors={"x": torch.ones(3, 1)}),
    ]
    merged = collect_proto(parts)
    assert merged.batch_size == 5
    assert torch.allclose(merged["x"][:2], torch.zeros(2, 1))
    assert torch.allclose(merged["x"][2:], torch.ones(3, 1))


def test_dispatch_one_to_all_keeps_identical_batch_size() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(3, 2).float()})
    chunks = dispatch_proto(p, num_workers=3, mode=DispatchMode.ONE_TO_ALL)
    assert len(chunks) == 3
    assert all(c.batch_size == p.batch_size for c in chunks)


def test_dispatch_nd_mapping() -> None:
    p = DataProto(tensors={"x": torch.arange(20).view(10, 2).float()})
    # Workers [0,1] share group 0; [2,3] share group 1.
    chunks = dispatch_proto(
        p,
        num_workers=4,
        mode=DispatchMode.DP_COMPUTE_PROTO,
        mesh_mapping=[0, 0, 1, 1],
    )
    assert len(chunks) == 4
    assert chunks[0].batch_size == chunks[1].batch_size
    assert chunks[2].batch_size == chunks[3].batch_size


def test_resolve_worker_class_from_role_key() -> None:
    cls = resolve_worker_class("actor.default")
    assert cls.__name__ == "LumenActorWorker"
