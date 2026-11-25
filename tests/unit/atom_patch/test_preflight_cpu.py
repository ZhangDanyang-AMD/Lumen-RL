"""Fail-closed identity/feature gates without importing ATOM or allocating a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "lumenrl"
    / "engine"
    / "inference"
    / "atom_patch"
    / "preflight.py"
)
_spec = importlib.util.spec_from_file_location("cleanmain_preflight_test", _SOURCE)
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)


@pytest.mark.parametrize(
    "missing",
    [
        "LUMEN_ATOM_RL_PATCH",
        "LUMENRL_ATOM_OP_ALIGNMENT",
        "LUMEN_ATOM_MXFP4_2D",
        "LUMEN_ATOM_MAIN_AITER_COMPAT",
        "LUMEN_ATOM_MAIN_LIFECYCLE",
    ],
)
def test_every_explicit_opt_in_required_before_import(monkeypatch, missing):
    required = [
        "LUMEN_ATOM_RL_PATCH",
        "LUMENRL_ATOM_OP_ALIGNMENT",
        "LUMEN_ATOM_MXFP4_2D",
        "LUMEN_ATOM_MAIN_AITER_COMPAT",
        "LUMEN_ATOM_MAIN_LIFECYCLE",
    ]
    for key in required:
        monkeypatch.setenv(key, "1")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match="requires"):
        preflight.require_clean_atom_main(atom_root="/does/not/exist")
