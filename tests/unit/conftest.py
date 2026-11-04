from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "Kimi_K3_SDDD_MI350_ATOM_MXFP4"
)


@pytest.fixture
def load_k3_example_module():
    def load(relative_path: str):
        path = EXAMPLE_ROOT / relative_path
        module_name = "k3_example_" + "_".join(path.with_suffix("").parts[-3:])
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def fake_lumen(monkeypatch):
    calls: dict[str, object] = {}

    class Config:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

        def enable(self, model):
            calls["model"] = model
            return "manager", model

    lumen = types.ModuleType("lumen")
    lumen.__path__ = []
    config_module = types.ModuleType("lumen.config")
    config_module.LumenConfig = Config
    quantize_module = types.ModuleType("lumen.quantize")
    quantize_module.__path__ = []

    monkeypatch.setitem(sys.modules, "lumen", lumen)
    monkeypatch.setitem(sys.modules, "lumen.config", config_module)
    monkeypatch.setitem(sys.modules, "lumen.quantize", quantize_module)
    return calls, quantize_module
