from __future__ import annotations

import torch

from lumenrl.core.protocol import DataProto


def test_create_empty() -> None:
    p = DataProto()
    assert p.batch_size == 0
    assert p.keys() == []
    assert p.meta == {}


def test_create_with_tensors() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(2, 3)}, meta={"k": 1})
    assert p.batch_size == 2
    assert "x" in p
    assert torch.equal(p["x"], torch.arange(6).view(2, 3))
    assert p.meta["k"] == 1


def test_batch_size() -> None:
    p = DataProto(tensors={"a": torch.zeros(5, 2), "b": torch.ones(5)})
    assert len(p) == 5
    assert p.batch_size == 5


def test_split_and_merge() -> None:
    p = DataProto(
        tensors={"x": torch.arange(12).view(4, 3).float()},
        meta={"m": True},
    )
    chunks = p.split(3)
    assert len(chunks) == 3
    sizes = [c.batch_size for c in chunks]
    assert sum(sizes) == 4
    merged = DataProto.merge(chunks)
    assert merged.batch_size == 4
    assert torch.allclose(merged["x"], p["x"])
    assert merged.meta == p.meta


def test_to_device_cpu_only() -> None:
    p = DataProto(tensors={"x": torch.tensor([1.0], device="cpu")})
    q = p.to("cpu")
    assert q["x"].device.type == "cpu"
    assert torch.equal(q["x"], p["x"])


def test_select_keys() -> None:
    p = DataProto(tensors={"a": torch.zeros(2), "b": torch.ones(2), "c": torch.full((2,), 2.0)})
    s = p.select(["b", "a"])
    assert set(s.keys()) == {"a", "b"}
    assert "c" not in s


def test_update() -> None:
    p = DataProto(tensors={"a": torch.zeros(2)}, meta={"x": 1})
    q = DataProto(tensors={"b": torch.ones(2)}, meta={"y": 2})
    p.update(q)
    assert set(p.keys()) == {"a", "b"}
    assert p.meta["x"] == 1 and p.meta["y"] == 2


def test_mini_batches() -> None:
    p = DataProto(tensors={"t": torch.arange(10).view(10, 1)})
    batches = list(p.mini_batches(3))
    assert len(batches) == 4
    assert [b.batch_size for b in batches] == [3, 3, 3, 1]
    cat = torch.cat([b["t"] for b in batches], dim=0)
    assert torch.equal(cat.squeeze(-1), torch.arange(10))


def test_router_distributions() -> None:
    p = DataProto()
    p.add_router_distributions(0, torch.randn(2, 4))
    p.add_router_distributions(2, torch.randn(2, 4))
    dists = p.get_router_distributions()
    assert set(dists.keys()) == {0, 2}
    assert dists[0].shape == (2, 4)
    assert p.has_router_distributions()


def test_merge_preserves_data() -> None:
    a = DataProto(tensors={"x": torch.tensor([[1.0], [2.0]])}, meta={"tag": "a"})
    b = DataProto(tensors={"x": torch.tensor([[3.0]])}, meta={"tag": "a"})
    m = DataProto.merge([a, b])
    assert torch.allclose(m["x"], torch.tensor([[1.0], [2.0], [3.0]]))
    assert m.meta["tag"] == "a"


def test_merge_key_mismatch_raises() -> None:
    a = DataProto(tensors={"x": torch.tensor([[1.0]]), "y": torch.tensor([[2.0]])})
    b = DataProto(tensors={"x": torch.tensor([[3.0]])})
    import pytest
    with pytest.raises(ValueError, match="keys mismatch"):
        DataProto.merge([a, b])
