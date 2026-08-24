from __future__ import annotations

import pytest

from lumenrl.transfer.eagle_mooncake_store import (
    SegmentedEagleMooncakeStore,
)
from lumenrl.transfer.mooncake_config import MooncakeConfig


class FakeStore:
    def __init__(self, config: MooncakeConfig, *, full: bool = False):
        self.config = config
        self.full = full
        self.setup_device = object()
        self.put_keys: list[str] = []
        self.flushes = 0
        self.removed: list[str] = []
        self.closed = False

    def setup(self, device=None):
        self.setup_device = device

    def put(self, key, *args, **kwargs):
        if self.full:
            raise RuntimeError("batch_put_from failed: store full")
        self.put_keys.append(key)
        return {"key": key}

    def flush(self):
        self.flushes += 1

    def get(self, key, *args, **kwargs):
        return {"key": key}

    def remove_eagle3_tensors(self, key, *args, **kwargs):
        self.removed.append(key)

    def close(self):
        self.closed = True


def _make_pool(monkeypatch, *, size: int = 3, full_indices=()):
    monkeypatch.setenv("LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE", str(size))
    monkeypatch.setenv("LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE", "2GB")
    monkeypatch.setenv("MOONCAKE_LOCAL_BUFFER_SIZE", "1GB")
    monkeypatch.setenv("MOONCAKE_DEVICE_NAME", "ionic_0,ionic_1")
    created: list[FakeStore] = []

    def factory(config):
        store = FakeStore(config, full=len(created) in full_indices)
        created.append(store)
        return store

    config = MooncakeConfig(device_name="unused", host_buffer_size=4096)
    pool = SegmentedEagleMooncakeStore(
        config,
        _store_factory=factory,
        _sleep=lambda _: None,
    )
    return pool, created


def test_round_robin_distribution_and_store_configuration(monkeypatch):
    pool, stores = _make_pool(monkeypatch)

    for index in range(5):
        pool.put(f"key-{index}", object())

    assert [store.put_keys for store in stores] == [
        ["key-0", "key-3"],
        ["key-1", "key-4"],
        ["key-2"],
    ]
    assert [store.config.device_name for store in stores] == [
        "ionic_0",
        "ionic_1",
        "ionic_0",
    ]
    assert all(store.config.global_segment_size == 2 * 1024**3 for store in stores)
    assert all(store.config.local_buffer_size == 1024**3 for store in stores)
    assert all(store.config.async_put_pool_size == 1 for store in stores)


def test_full_store_fails_over_without_publishing_early(monkeypatch):
    pool, stores = _make_pool(monkeypatch, full_indices={0})

    result = pool.put("key", object())

    assert result == {"key": "key"}
    assert stores[0].put_keys == []
    assert stores[1].put_keys == ["key"]
    assert stores[1].flushes == 1


def test_get_remove_flush_and_close_cover_initialized_stores(monkeypatch):
    pool, stores = _make_pool(monkeypatch)
    pool.put("key-0", object())
    pool.put("key-1", object())

    assert pool.get("key-0") == {"key": "key-0"}
    pool.remove_eagle3_tensors("key-0")
    pool.flush()
    pool.close()
    pool.close()

    assert stores[0].removed == ["key-0"]
    assert [store.flushes for store in stores] == [2, 2]
    assert all(store.closed for store in stores)


def test_all_full_segments_use_bounded_backpressure(monkeypatch):
    monkeypatch.setenv("LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS", "0")
    pool, _ = _make_pool(monkeypatch, size=2, full_indices={0, 1})

    with pytest.raises(RuntimeError, match="remained full"):
        pool.put("key", object())
