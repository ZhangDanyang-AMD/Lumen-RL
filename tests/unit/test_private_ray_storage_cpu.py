"""Private plasma must be applied before Ray starts, not inferred from TMPDIR."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "lumenrl/controller/ray_cluster.py"


def _module(monkeypatch, existing=False):
    ray = types.ModuleType("ray")
    calls = []
    ray.is_initialized = lambda: existing
    ray.init = lambda **kwargs: calls.append(kwargs)
    ray.get_runtime_context = lambda: SimpleNamespace(runtime_env={"env_vars": {}})
    cfg = types.ModuleType("lumenrl.core.config")
    cfg.ClusterConfig = type("ClusterConfig", (), {})
    for name in ("lumenrl", "lumenrl.core"):
        stub = types.ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "lumenrl.core.config", cfg)
    monkeypatch.setitem(sys.modules, "ray", ray)
    spec = importlib.util.spec_from_file_location("private_ray_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, calls


def _config(address=None):
    return SimpleNamespace(ray_address=address, num_nodes=1, gpus_per_node=4)


def test_private_plasma_init_uses_bounded_disk_options(monkeypatch, tmp_path):
    module, calls = _module(monkeypatch)
    plasma = tmp_path / "plasma"
    temp = tmp_path / "temp"
    plasma.mkdir()
    temp.mkdir()
    monkeypatch.setenv("LUMENRL_RAY_PLASMA_DIRECTORY", str(plasma))
    monkeypatch.setenv("LUMENRL_RAY_PRIVATE_TEMP_DIR", str(temp))
    monkeypatch.setenv("LUMENRL_RAY_OBJECT_STORE_MIB", "4096")
    module.RayCluster(_config()).init()
    assert calls == [{"num_gpus": 4, "_plasma_directory": str(plasma),
                      "object_store_memory": 4096 * 1024 * 1024,
                      "_temp_dir": str(temp), "ignore_reinit_error": True}]


@pytest.mark.parametrize("missing", ["LUMENRL_RAY_PRIVATE_TEMP_DIR", "LUMENRL_RAY_OBJECT_STORE_MIB"])
def test_missing_budget_or_private_temp_refused_before_ray_init(monkeypatch, tmp_path, missing):
    module, calls = _module(monkeypatch)
    monkeypatch.setenv("LUMENRL_RAY_PLASMA_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("LUMENRL_RAY_PRIVATE_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("LUMENRL_RAY_OBJECT_STORE_MIB", "4096")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match="private Ray plasma|Private Ray plasma"):
        module.RayCluster(_config()).init()
    assert calls == []


def test_remote_cluster_cannot_claim_private_plasma(monkeypatch, tmp_path):
    module, calls = _module(monkeypatch)
    monkeypatch.setenv("LUMENRL_RAY_PLASMA_DIRECTORY", str(tmp_path))
    with pytest.raises(RuntimeError, match="remote Ray cluster"):
        module.RayCluster(_config(address="ray://somewhere")).init()
    assert calls == []
