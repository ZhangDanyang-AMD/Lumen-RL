"""Two-version TP1 MXFP4-2D refit regression (CPU tensors, actual worker seam)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

REFIT = Path(__file__).resolve().parents[3] / "lumenrl/engine/inference/atom_patch/refit_main.py"


class FakeModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.mxfp4_2d = True
        self._mxfp4_2d_shape = (32, 32)
        self.weight = torch.nn.Parameter(torch.zeros(32, 16, dtype=torch.uint8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(32, 1, dtype=torch.uint8), requires_grad=False)

    def weight_loader(self, param, value, shard_id=None):
        if shard_id is None:
            raise RuntimeError("old main loader cannot copy BF16 matrix into FP4 target")
        start = 0 if shard_id == 0 else 16
        param.data[start:start + 16, :] = value[:16, :]


class Runner:
    def __init__(self, rank):
        self.rank = rank
        self.world_size = 1
        self.device = torch.device("cpu")
        self.label = f"rank{rank}"
        self.model = torch.nn.Module()
        self.model.add_module("a", FakeModule("a"))
        self.model.add_module("b", FakeModule("b"))
        self.clear_count = 0
        self._packed_weight_accum = {}

    def _sync_target_model(self):
        return self.model

    def _get_param_to_module_mapping(self):
        return {
            (f"{name}.{leaf}" if name else leaf): (module, leaf, param)
            for name, module in self.model.named_modules()
            for leaf, param in module.named_parameters(recurse=False)
        }

    def _resolve_packed_name(self, name, mapping):
        if name == "a.w1":
            return "a.weight", 0, "a.weight"
        if name == "a.w3":
            return "a.weight", 1, "a.weight"
        return None

    def _get_packed_shard_order(self):
        return {"a.weight": [0, 1]}

    def _copy_into_param(self, param, tensor):
        param.data.copy_(tensor)

    def _load_into_param(self, param, loader, tensor, *args, **kwargs):
        loader(param, tensor, *args, **kwargs)

    @staticmethod
    def _is_fp8_param(module, param):
        return param.dtype == torch.float8_e4m3fn

    def _discard_failed_weight_sync(self):
        self._packed_weight_accum.clear()
        return True

    def _apply_packed_weight(self, name, tensor, mapping):
        return "skipped"

    def _apply_unmatched_weight(self, name, tensor, mapping):
        return self._apply_packed_weight(name, tensor, mapping)

    def clear_kv_cache(self):
        self.clear_count += 1
        return True

    def update_weights(self, named_tensors, clear_kv_cache=True):
        mapping = self._get_param_to_module_mapping()
        updated = 0
        for name, tensor in named_tensors:
            if name not in mapping:
                result = self._apply_unmatched_weight(name, tensor, mapping)
                if result == "updated":
                    updated += 1
                continue
            module, leaf, param = mapping[name]
            if tensor.shape == param.shape:
                self._copy_into_param(param, tensor.to(param.dtype))
            else:
                try:
                    self._load_into_param(param, module.weight_loader, tensor)
                except Exception:
                    continue
            updated += 1
        if clear_kv_cache:
            self.clear_kv_cache()
        self._packed_weight_accum.clear()
        return updated

    def update_weights_from_shm(self, shm_name, bucket_meta, is_last=True):
        return self._apply_fake_bucket(bucket_meta, is_last)

    def update_weights_from_ipc(self, ipc_handle, bucket_meta, is_last=True, ipc_handles=None):
        return self._apply_fake_bucket(bucket_meta, is_last)

    def _apply_fake_bucket(self, bucket_meta, is_last):
        mapping = self._get_param_to_module_mapping()
        updated = 0
        for name, meta in bucket_meta.items():
            tensor = meta["tensor"]
            if name not in mapping:
                updated += self._apply_unmatched_weight(name, tensor, mapping) == "updated"
                continue
            module, _, param = mapping[name]
            try:
                self._load_into_param(param, module.weight_loader, tensor)
                updated += 1
            except Exception:
                continue
        if is_last:
            self.clear_kv_cache()
            self._packed_weight_accum.clear()
        return updated


def _meta(tensor):
    return {"tensor": tensor, "shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}


def _quantize(weight):
    # Deterministic CPU stand-in for AITER: output bytes and E8M0-like scale
    # both change whenever the BF16 source changes. This checks routing/version,
    # not AITER's numerical quantization itself.
    scale = weight.float().abs().amax(dim=1).clamp(min=1).to(torch.uint8)[:, None]
    packed = weight[:, ::2].float().round().to(torch.uint8)
    return packed, scale


@pytest.fixture
def patch(monkeypatch):
    package = "lumenrl.engine.inference.atom_patch"
    fake_atom = types.ModuleType("atom")
    fake_atom.__path__ = []
    fake_rollout = types.ModuleType("atom.rollout")
    fake_rollout.__path__ = []
    updater = types.ModuleType("atom.rollout.weight_updater")
    updater.WeightUpdaterMixin = Runner
    fake_quant = types.ModuleType(package + ".quant_main")
    fake_quant.quantize_mxfp4_2d_weight = _quantize
    monkeypatch.setitem(sys.modules, "atom", fake_atom)
    monkeypatch.setitem(sys.modules, "atom.rollout", fake_rollout)
    monkeypatch.setitem(sys.modules, "atom.rollout.weight_updater", updater)
    for name in ("lumenrl", "lumenrl.engine", "lumenrl.engine.inference", package):
        stub = types.ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, package + ".quant_main", fake_quant)
    monkeypatch.setitem(sys.modules, package + ".alignment_main", types.ModuleType("alignment_main"))
    monkeypatch.setenv("LUMEN_ATOM_MXFP4_2D", "1")
    spec = importlib.util.spec_from_file_location("refit_under_test", REFIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.install()
    return module


def _check_consumed(runner, expected_a, expected_b):
    for name, source in (("a", expected_a), ("b", expected_b)):
        layer = getattr(runner.model, name)
        packed, scale = _quantize(source)
        assert torch.equal(layer.weight.data, packed)
        assert torch.equal(layer.weight_scale.data, scale)


def test_two_updates_requantize_actual_consumed_parameters(patch):
    for rank in range(4):
        worker = Runner(rank)
        a0 = torch.full((32, 32), 2, dtype=torch.bfloat16)
        b0 = torch.full((32, 32), 3, dtype=torch.bfloat16)
        a1 = torch.full((32, 32), 4, dtype=torch.bfloat16)
        b1 = torch.full((32, 32), 5, dtype=torch.bfloat16)
        assert worker.update_weights([("a.weight", a0), ("b.weight", b0)]) == 2
        _check_consumed(worker, a0, b0)
        first = worker._lumen_mxfp4_refit_state["committed"]
        assert worker.update_weights([("a.weight", a1), ("b.weight", b1)]) == 2
        _check_consumed(worker, a1, b1)
        assert worker._lumen_mxfp4_refit_state["committed"] == first + 1
        assert worker.clear_count == 2


def test_packed_shards_in_one_direct_update_are_requantized_once(patch):
    worker = Runner(0)
    part0 = torch.full((16, 32), 6, dtype=torch.bfloat16)
    part1 = torch.full((16, 32), 8, dtype=torch.bfloat16)
    b = torch.full((32, 32), 3, dtype=torch.bfloat16)
    assert worker.update_weights(
        [("a.w1", part0), ("a.w3", part1), ("b.weight", b)]
    ) == 2
    _check_consumed(worker, torch.cat((part0, part1), dim=0), b)
    assert worker._lumen_mxfp4_refit_state["committed"] == 1


@pytest.mark.parametrize("transport", ["ipc", "shm"])
def test_bucketed_two_versions_and_partial_final_refusal(patch, transport):
    worker = Runner(0)
    method = (worker.update_weights_from_ipc if transport == "ipc"
              else worker.update_weights_from_shm)
    src0 = torch.full((32, 32), 2, dtype=torch.bfloat16)
    src1 = torch.full((32, 32), 4, dtype=torch.bfloat16)

    def bucket(data, is_last):
        metadata = {name: _meta(tensor) for name, tensor in data.items()}
        return method(None, metadata, is_last)

    assert bucket({"a.weight": src0}, False) == 1
    assert worker._lumen_mxfp4_refit_state["committed"] is None
    assert bucket({"b.weight": src0}, True) == 1
    _check_consumed(worker, src0, src0)
    assert worker._lumen_mxfp4_refit_state["committed"] == 1
    assert bucket({"a.weight": src1}, False) == 1
    assert bucket({"b.weight": src1}, True) == 1
    _check_consumed(worker, src1, src1)
    assert worker._lumen_mxfp4_refit_state["committed"] == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        bucket({"a.weight": src0}, True)
    assert worker._lumen_mxfp4_refit_state["committed"] == 2


def test_incomplete_or_duplicate_version_refused(patch):
    worker = Runner(0)
    src = torch.full((32, 32), 7, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.update_weights([("a.weight", src)])
    assert worker._lumen_mxfp4_refit_state["committed"] is None
    with pytest.raises(ValueError, match="repeated|duplicate"):
        worker.update_weights([("a.weight", src), ("a.weight", src), ("b.weight", src)])
    assert worker._lumen_mxfp4_refit_state["committed"] is None


def test_only_selected_fp4_target_excluded_from_fp8_path(patch):
    worker = Runner(0)
    layer = worker.model.a
    assert worker._is_fp8_param(layer, layer.weight) is False
    other = torch.nn.Module()
    other.weight = torch.nn.Parameter(torch.zeros(32, 32, dtype=torch.float8_e4m3fn), requires_grad=False)
    assert worker._is_fp8_param(other, other.weight) is True


def test_worker_receipt_refuses_uncommitted_version(patch):
    worker = Runner(0)
    assert worker.get_lumen_refit_receipt(expect_committed=False)["refit_generation"] is None
    with pytest.raises(RuntimeError, match="no complete"):
        worker.get_lumen_refit_receipt()
    src = torch.full((32, 32), 9, dtype=torch.bfloat16)
    worker.update_weights_from_ipc(None, {"a.weight": _meta(src)}, False)
    with pytest.raises(RuntimeError, match="partial"):
        worker.get_lumen_refit_receipt()
    worker.update_weights_from_ipc(None, {"b.weight": _meta(src)}, True)
    receipt = worker.get_lumen_refit_receipt()
    assert receipt["rank"] == 0
    assert receipt["mxfp4_2d_target_count"] == 2
    assert receipt["refit_generation"] == 1
    assert receipt["c1_hit"] == receipt["c1_fallback"] == 0
    audit = worker.get_lumen_refit_receipt(audit_name="a.weight")["audit"]
    import hashlib

    weight, scale = _quantize(src)
    assert audit["weight_sha256"] == hashlib.sha256(weight.numpy().tobytes()).hexdigest()
    assert audit["scale_sha256"] == hashlib.sha256(scale.numpy().tobytes()).hexdigest()
    assert audit["generation"] == 1
    with pytest.raises(ValueError, match="unknown"):
        worker.get_lumen_refit_receipt(audit_name="not.a.weight")


def test_unsupported_source_refused_before_old_loader(patch):
    worker = Runner(0)
    with pytest.raises(ValueError, match="BF16"):
        worker.update_weights([("a.weight", torch.ones(32, 32, dtype=torch.float32))])
    assert worker.clear_count == 0
