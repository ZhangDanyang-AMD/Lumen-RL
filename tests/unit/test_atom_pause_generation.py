"""Pausing the ATOM rollout around a weight sync, for partial rollout."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from lumenrl.engine.inference.atom_ray_server import ATOMRayServer, ATOMReplicaManager


class CoreMgr:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._fail = fail

    def broadcast_utility_command_sync(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self._fail is not None:
            raise self._fail
        return [{"cmd": cmd}]


def _server(core_mgr: CoreMgr) -> ATOMRayServer:
    server = object.__new__(ATOMRayServer)
    server.engine = SimpleNamespace(core_mgr=core_mgr)
    return server


def test_pause_and_resume_reach_every_engine_with_the_mode() -> None:
    core_mgr = CoreMgr()
    server = _server(core_mgr)

    asyncio.run(server.pause_generation("keep", True))
    asyncio.run(server.resume_generation())

    assert core_mgr.calls == [
        ("pause_generation", {"mode": "keep", "clear_cache": True}),
        ("resume_generation", {}),
    ]


def test_a_refused_pause_fails_the_sync() -> None:
    server = _server(CoreMgr(fail=RuntimeError("connector holds 2")))

    with pytest.raises(RuntimeError, match="connector holds 2"):
        asyncio.run(server.pause_generation())


@pytest.mark.parametrize(("modes", "expected"), [(("keep",), ["keep"]), (None, [])])
def test_pause_modes_come_from_the_atom_build(monkeypatch, modes, expected) -> None:
    module = types.ModuleType("atom.model_engine.engine_utility")
    if modes is not None:
        module.GENERATION_PAUSE_MODES = modes
    monkeypatch.setitem(sys.modules, "atom.model_engine.engine_utility", module)

    assert asyncio.run(_server(CoreMgr()).generation_pause_modes()) == expected


class Remote:
    def __init__(self, value=None) -> None:
        self.value = value
        self.args: list[tuple] = []

    def remote(self, *args):
        self.args.append(args)
        return self.value


def test_the_pool_pauses_in_the_modes_every_replica_takes(monkeypatch) -> None:
    import ray

    monkeypatch.setattr(ray, "get", lambda refs: refs)
    servers = [
        SimpleNamespace(
            generation_pause_modes=Remote(modes),
            pause_generation=Remote(True),
            resume_generation=Remote(True),
        )
        for modes in (["keep"], ["keep", "wait"])
    ]
    mgr = object.__new__(ATOMReplicaManager)
    mgr.servers = servers

    assert mgr.generation_pause_modes() == {"keep"}
    mgr.pause_all(mode="keep", clear_cache=True)
    mgr.resume_all()

    assert [s.pause_generation.args for s in servers] == [[("keep", True)]] * 2
    assert [s.resume_generation.args for s in servers] == [[()]] * 2
