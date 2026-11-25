"""Pinned physical UUID/PCI gate on every Ray actor before rollout placement."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "lumenrl/workers/base_worker.py"


def _gate(monkeypatch, tmp_path, *, unique_id="c8fe46bfbdf4b35b"):
    payload = {
        "physical": {"0": {
            "unique_id": unique_id, "pci": "0000:05:00.0", "render": "renderD136"
        }}
    }
    manifest = tmp_path / "gpu-map.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("LUMENRL_SMCI_GPU_IDENTITY_MANIFEST", str(manifest))
    node = next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "_verified_smci_gpu_identity")
    props = SimpleNamespace(
        uuid="63386665-3436-6266-6264-663462333562", pci_domain_id=0,
        pci_bus_id=5, pci_device_id=0,
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_properties=lambda index: props,
    ))
    namespace = {"_os": os, "_torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_verified_smci_gpu_identity"]


def test_roc_uuid_and_pci_match_one_ray_rank(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, tmp_path)
    assert gate([0]) == {
        "ray_gpu_id": "0", "physical_gpu_id": 0,
        "physical_gpu_uuid": "c8fe46bfbdf4b35b",
        "physical_gpu_pci": "0000:05:00.0", "physical_gpu_render": "renderD136",
    }


def test_ray_ordinal_permutation_resolves_actual_physical_gpu(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, tmp_path)
    manifest = Path(os.environ["LUMENRL_SMCI_GPU_IDENTITY_MANIFEST"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["physical"]["3"] = {
        "unique_id": "d373c7db88d84636", "pci": "0000:75:00.0", "render": "renderD128"
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    observed = gate([3])
    assert observed["ray_gpu_id"] == "3"
    assert observed["physical_gpu_id"] == 0
    assert observed["physical_gpu_uuid"] == "c8fe46bfbdf4b35b"


def test_mismatched_or_duplicate_gpu_fails_closed(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, tmp_path, unique_id="0000000000000000")
    with pytest.raises(RuntimeError, match="hardware unmapped"):
        gate([0])
    with pytest.raises(RuntimeError, match="exactly one GPU"):
        gate([0, 1])
