"""The coverage check on ATOM's own SHM loader, which returns no counts."""

import pytest

from lumenrl.engine.inference.atom_moe_weight_sync import (
    RecordingCoreManager,
    assert_sync_applied,
)


class _CoreManager:
    def __init__(self, applied):
        self.applied, self.sessions = list(applied), 0

    def broadcast_utility_command_sync(self, cmd, **kwargs):
        return [{"cmd": cmd, "result": self.applied.pop(0)}]

    def weight_sync_session(self):
        self.sessions += 1
        return self


def test_it_records_each_bucket_and_forwards_everything_else():
    core = RecordingCoreManager(_CoreManager([3, 1]))

    core.broadcast_utility_command_sync("update_weights_shm", shm_name="x", bucket_meta={"a": 0, "b": 0, "c": 0})
    core.broadcast_utility_command_sync("update_weights_shm", shm_name="x", bucket_meta={"d": 0})
    core.weight_sync_session()

    assert core.buckets == [(3, [3]), (1, [1])]
    assert core._inner.sessions == 1


def test_a_sync_an_engine_applied_none_of_is_an_error():
    with pytest.raises(RuntimeError, match="applied none"):
        assert_sync_applied([(103, [0]), (71, [0])], context="shm reload")


def test_packed_shards_counting_once_is_not_an_error():
    assert_sync_applied([(103, [97]), (71, [62]), (7, [1])], context="shm reload")


def test_the_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("LUMENRL_WEIGHT_SYNC_CHECK", "off")
    assert_sync_applied([(5, [0])], context="shm reload")
