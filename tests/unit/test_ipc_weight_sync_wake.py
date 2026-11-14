"""Order of events when a colocated, sleeping vLLM rollout takes new weights over IPC."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from lumenrl.trainer.rl_trainer import RLTrainer


def test_actors_free_their_gradients_and_cache_before_a_sleeping_rollout_wakes(monkeypatch):
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

    assert events == [
        "release_grad_buffers", "free_cached_memory", "wake weights", "update_weights_ipc_send", "wake kv_cache",
    ]


def _rollout_setup_trainer(monkeypatch, events, *, separated):
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda name, trust_remote_code: SimpleNamespace(pad_token="<pad>", eos_token="</s>"),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(policy=SimpleNamespace(generation=SimpleNamespace(
        vllm_cfg=SimpleNamespace(transport="ray_http"), atom_cfg=SimpleNamespace(transport="fifo"),
    )))
    trainer._use_vllm = True
    trainer._gen_backend = "vllm"
    trainer._rollout_wg = object() if separated else None
    trainer._actor_wg = SimpleNamespace(execute_all_sync=lambda name: events.append(name) or [{}])
    trainer._setup_ray_vllm_rollout = lambda model_name, vcfg: events.append("start vllm")
    return trainer


def test_actors_free_their_gradients_and_cache_before_a_colocated_rollout_starts(monkeypatch):
    events = []

    _rollout_setup_trainer(monkeypatch, events, separated=False)._setup_ray_rollout_engine("model")

    assert events == ["release_grad_buffers", "free_cached_memory", "start vllm"]


def test_a_separated_rollout_starts_without_touching_the_actors(monkeypatch):
    events = []

    _rollout_setup_trainer(monkeypatch, events, separated=True)._setup_ray_rollout_engine("model")

    assert events == ["start vllm"]
