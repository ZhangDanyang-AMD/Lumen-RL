"""The IPC weight sync returns the actors' cached memory before the receivers start.

The colocated vLLM workers begin allocating their online-FP8 reload buffers as
soon as update_weights_from_ipc starts. The actors' own empty_cache() inside
update_weights_ipc_send runs only once each send call begins, so with a
training-peak cache (~137 GB) the device still reached 204 of 206 GB at a sync.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumenrl.trainer.rl_trainer import RLTrainer


class _Stop(Exception):
    pass


def test_actor_cache_is_freed_before_receivers_start(monkeypatch) -> None:
    events: list[str] = []

    class _Remote:
        def remote(self, *a, **k):
            events.append("recv")
            return object()

    class _Server:
        update_weights_from_ipc = _Remote()

    class _ActorWG:
        num_workers = 2

        def execute_all_sync(self, name, *a, **k):
            events.append(name)
            return [{}, {}]

        def execute_all_async(self, name, *a, **k):
            events.append(name)
            raise _Stop

    t = object.__new__(RLTrainer)
    t.config = SimpleNamespace(
        weight_sync=SimpleNamespace(backend="auto"),
        policy=SimpleNamespace(generation=SimpleNamespace(vllm_cfg=SimpleNamespace(
            update_weights_bucket_megabytes=512, use_shm=False))),
    )
    t._ray_rollout_mgr = SimpleNamespace(servers=[_Server(), _Server()], num_replicas=2)
    t._actor_wg = _ActorWG()
    t._rollout_wg = None
    t._ray_vllm_engine = SimpleNamespace(enable_sleep=False)
    t.global_step = 0
    monkeypatch.setattr(RLTrainer, "_ipc_endpoints_match_actors", lambda self, mgr: True)

    with pytest.raises(_Stop):
        t._sync_weights_ipc()
    assert events[0] == "free_cached_memory"
    assert events.index("free_cached_memory") < events.index("recv")
