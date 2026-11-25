"""A clean-main weight sync owns one session across every IPC/SHM bucket."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "lumenrl/engine/inference/atom_ray_server.py"


def _wrap(method):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "ATOMRayServer")
    fn = next(node for node in actor.body if isinstance(node, ast.FunctionDef)
              and node.name == f"_update_weights_from_{method}_sync")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns[fn.name]


@pytest.mark.parametrize("method", ["ipc", "shm"])
def test_all_buckets_share_one_main_weight_sync_session(method):
    events = []
    manager = SimpleNamespace()
    manager.active = False

    @contextmanager
    def session():
        assert not manager.active
        manager.active = True
        events.append("begin")
        try:
            yield
        finally:
            manager.active = False
            events.append("end")

    manager.weight_sync_session = session
    server = SimpleNamespace(engine=SimpleNamespace(core_mgr=manager))

    def cycle(version):
        for bucket in ("first_nonfinal", "middle_nonfinal", "last_final"):
            assert manager.active, "main would reject non-final update outside session"
            events.append((bucket, version))
        return "complete"

    setattr(server, f"_update_weights_from_{method}_cycle", cycle)
    assert _wrap(method)(server, 2) == "complete"
    assert events == ["begin", ("first_nonfinal", 2), ("middle_nonfinal", 2),
                      ("last_final", 2), "end"]
    assert not manager.active


@pytest.mark.parametrize("method", ["ipc", "shm"])
def test_missing_session_refuses_before_any_bucket(method):
    server = SimpleNamespace(engine=SimpleNamespace(core_mgr=SimpleNamespace()))
    setattr(server, f"_update_weights_from_{method}_cycle",
            lambda version: pytest.fail("bucket dispatched without main session"))
    with pytest.raises(RuntimeError, match="weight_sync_session"):
        _wrap(method)(server, 1)
