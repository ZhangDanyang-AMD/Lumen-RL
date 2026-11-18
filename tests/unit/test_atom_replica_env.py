"""What each colocated ATOM replica's actor gets in its environment."""

import types

import pytest
import ray

from lumenrl.engine.inference.atom_ray_server import ATOMReplicaManager


class _Group:
    num_workers = 4

    def execute_all_sync(self, method):
        assert method == "get_colocation_info"
        return [{"node_id": "ab" * 28, "gpu_ids": [i]} for i in range(self.num_workers)]


def _replica_envs(monkeypatch, **kwargs):
    envs = []

    class _Remote:
        def options(self, **opts):
            envs.append(opts["runtime_env"]["env_vars"])
            return self

        def remote(self, **_):
            return types.SimpleNamespace(launch=types.SimpleNamespace(remote=lambda: None))

    monkeypatch.setattr(ray, "remote", lambda cls: _Remote())
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: types.SimpleNamespace(get_job_id=lambda: "01"))
    monkeypatch.delenv("LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS", raising=False)
    mgr = ATOMReplicaManager(_Group(), "model", {"tensor_parallel_size": 2}, **kwargs)
    monkeypatch.setattr(mgr, "_get_true_vocab_size", lambda: None)
    mgr.create()
    return envs


@pytest.mark.parametrize("keep, want", [(True, "1"), (False, None)])
def test_sleep_keeps_weights_reaches_every_replica(monkeypatch, keep, want):
    """The runner reads it in ATOM's own processes, which inherit the replica actor's
    environment; set in the config only, it must not depend on the raylets'."""
    envs = _replica_envs(monkeypatch, sleep_keeps_weights=keep)

    assert len(envs) == 2
    assert [env.get("LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS") for env in envs] == [want, want]
