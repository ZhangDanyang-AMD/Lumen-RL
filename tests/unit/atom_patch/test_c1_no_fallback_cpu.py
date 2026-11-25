"""Selected C1 must expose a hit or fail, never silently time the baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "lumenrl/engine/inference/atom_patch/install.py"


def _module(monkeypatch, selected):
    monkeypatch.setenv("ATOM_ALIGNED_KV_PACK", "1" if selected else "0")
    monkeypatch.delenv("ATOM_DECODE_PROFILE", raising=False)
    spec = importlib.util.spec_from_file_location("c1_gate_test", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attention():
    return SimpleNamespace(
        _gather_live_kv=lambda *args: "baseline",
        aligned_varlen_attention=lambda *args: "attention",
        _batch=lambda *args: "batch",
    )


def test_selected_failure_is_not_silent_baseline(monkeypatch):
    mod = _module(monkeypatch, True)
    mod._kv_pack = SimpleNamespace(
        fused_gather_live_kv=lambda *args: (_ for _ in ()).throw(RuntimeError("kernel"))
    )
    mod._decode_profile = SimpleNamespace(current=lambda: None)
    attention = _attention()
    mod._patch_aligned_attention(attention)
    with pytest.raises(RuntimeError, match="no silent fallback"):
        attention._gather_live_kv(None, None, [], [], 16, None)
    assert attention._lumen_c1_counters == {"hit": 0, "fallback": 1}


def test_selected_hit_recorded_only_on_success(monkeypatch):
    mod = _module(monkeypatch, True)
    mod._kv_pack = SimpleNamespace(fused_gather_live_kv=lambda *args: "C1")
    mod._decode_profile = SimpleNamespace(current=lambda: None)
    attention = _attention()
    mod._patch_aligned_attention(attention)
    assert attention._gather_live_kv(None, None, [], [], 16, None) == "C1"
    assert attention._lumen_c1_counters == {"hit": 1, "fallback": 0}


def test_off_stays_baseline(monkeypatch):
    mod = _module(monkeypatch, False)
    mod._kv_pack = SimpleNamespace(
        fused_gather_live_kv=lambda *args: (_ for _ in ()).throw(AssertionError("called"))
    )
    mod._decode_profile = SimpleNamespace(current=lambda: None)
    attention = _attention()
    mod._patch_aligned_attention(attention)
    assert attention._gather_live_kv(None, None, [], [], 16, None) == "baseline"
    assert attention._lumen_c1_counters == {"hit": 0, "fallback": 0}
