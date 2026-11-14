"""Order of events when a colocated, sleeping vLLM rollout takes new weights over IPC."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from lumenrl.trainer.rl_trainer import RLTrainer


def test_actors_free_their_cache_before_a_sleeping_rollout_wakes(monkeypatch):
    events = []
    fake_ray = ModuleType("ray")
    fake_ray.wait = lambda pending, num_returns=1: ([pending[0]], pending[1:])
    fake_ray.get = lambda ref: {}
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    trainer = object.__new__(RLTrainer)
    trainer.global_step = 0
    trainer.config = SimpleNamespace(
        weight_sync=SimpleNamespace(backend="auto"),
        policy=SimpleNamespace(generation=SimpleNamespace(
            vllm_cfg=SimpleNamespace(update_weights_bucket_megabytes=512, use_shm=False),
        )),
    )
    trainer._ray_rollout_mgr = SimpleNamespace(
        servers=[SimpleNamespace(update_weights_from_ipc=SimpleNamespace(remote=lambda *a: "recv"))],
    )
    trainer._ipc_endpoints_match_actors = lambda mgr: True
    trainer._actor_wg = SimpleNamespace(
        execute_all_sync=lambda name: events.append(name) or [{}],
        execute_all_async=lambda name, **kw: events.append(name) or ["send"],
    )
    trainer._ray_vllm_engine = SimpleNamespace(
        enable_sleep=True, wake=lambda tags=None: events.append(f"wake {tags[0]}"),
    )

    trainer._sync_weights_ipc()

    assert events == ["free_cached_memory", "wake weights", "update_weights_ipc_send", "wake kv_cache"]
