from __future__ import annotations

import torch

from lumenrl.controller.dispatch import collect_proto, dispatch_proto
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
