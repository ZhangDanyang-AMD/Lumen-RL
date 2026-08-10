from __future__ import annotations

import json
import logging
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import pytest
import torch

import lumenrl.engine.inference.rdma_weight_transfer as rdma
import lumenrl.engine.inference.vllm_colocate_worker_ext as worker_ext
import lumenrl.engine.inference.vllm_fp8_utils as fp8_utils
from lumenrl.engine.inference.rdma_protocol import (
    RDMA_PROTOCOL_VERSION,
    RDMACapability,
)
from lumenrl.engine.inference.vllm_colocate_worker_ext import (
    vLLMColocateWorkerExtension,
)


def test_all_gather_lifecycle_diagnostic_syncs_before_and_after(monkeypatch, caplog):
    events = []

    class FakeGroupCoordinator:
        def all_gather(self, input_, use_custom=False, dim=-1):
            events.append(("collective", input_.numel(), use_custom, dim))
            return input_ + 1

    install = getattr(
        worker_ext,
        "_install_all_gather_lifecycle_diagnostics",
        None,
    )
    assert install is not None

    monkeypatch.setenv("LUMENRL_DIAG_ALL_GATHER", "1")
    monkeypatch.setenv("LUMENRL_DIAG_ALL_GATHER_NUMEL", "4")
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: events.append(("sync", device)),
    )
    caplog.set_level(logging.WARNING)

    install(FakeGroupCoordinator)
    install(FakeGroupCoordinator)
    input_tensor = torch.arange(4, dtype=torch.bfloat16)
    output = FakeGroupCoordinator().all_gather(input_tensor, dim=0)

    assert events == [
        ("sync", input_tensor.device),
        ("collective", 4, False, 0),
        ("sync", input_tensor.device),
    ]
    torch.testing.assert_close(output, input_tensor + 1)
    messages = [record.getMessage() for record in caplog.records]
    assert any("phase=before_presync" in message for message in messages)
    assert any("phase=after_presync" in message for message in messages)
    assert any("phase=after_all_gather" in message for message in messages)


def test_monkey_patch_model_installs_all_gather_lifecycle_diagnostic(monkeypatch):
    import lumenrl.moe.router_precision as router_precision

    events = []
    model = object()
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = SimpleNamespace(model=model)
    monkeypatch.setattr(
        worker_ext,
        "_install_all_gather_lifecycle_diagnostics",
        lambda: events.append("all_gather"),
    )
    monkeypatch.setattr(
        worker_ext,
        "_monkey_patch_compute_logits",
        lambda patched_model, vocab_size: events.append(
            ("compute_logits", patched_model, vocab_size)
        ),
    )
    monkeypatch.setattr(
        router_precision,
        "enable_fp32_moe_router",
        lambda patched_model: events.append(("router", patched_model)),
    )

    worker.monkey_patch_model(vocab_size=1024)

    assert events == [
        "all_gather",
        ("compute_logits", model, 1024),
        ("router", model),
    ]


def test_rdma_capability_is_immutable_and_json_serializable():
    capability = RDMACapability(
        protocol_version=RDMA_PROTOCOL_VERSION,
        module_path="example.worker",
        online_quant_reload=True,
        prequantized_stream=True,
    )

    with pytest.raises(FrozenInstanceError):
        capability.protocol_version = 1

    assert json.loads(json.dumps(capability.to_dict())) == {
        "protocol_version": 2,
        "module_path": "example.worker",
        "online_quant_reload": True,
        "prequantized_stream": True,
    }


def test_worker_reports_actual_rdma_capabilities():
    mixed_worker_type = type(
        "WorkerWithExtension",
        (vLLMColocateWorkerExtension,),
        {"__module__": "vllm.worker.worker"},
    )
    worker = object.__new__(mixed_worker_type)

    assert worker.get_rdma_capabilities() == {
        "protocol_version": 2,
        "module_path": (
            "lumenrl.engine.inference.vllm_colocate_worker_ext"
        ),
        "online_quant_reload": True,
        "prequantized_stream": True,
    }


def test_prequantized_fp8_metadata_restores_weight_and_block_scale_types():
    class OriginalWeight(torch.nn.Parameter):
        @property
        def output_dim(self):
            return self._output_dim

        @property
        def input_dim(self):
            return self._input_dim

        @property
        def weight_loader(self):
            return self._weight_loader

        @weight_loader.setter
        def weight_loader(self, value):
            self._weight_loader = value

    class BlockScale(torch.nn.Parameter):
        @property
        def output_dim(self):
            return self._output_dim

        @property
        def input_dim(self):
            return self._input_dim

        @property
        def weight_loader(self):
            return self._weight_loader

        @weight_loader.setter
        def weight_loader(self, value):
            self._weight_loader = value

    class LinearLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(
                torch.ones(1, 1),
                requires_grad=False,
            )
            self.weight_block_size = [128, 128]

    layer = LinearLayer()
    model = torch.nn.Module()
    model.add_module("linear", layer)
    recorded_loader = object()
    resident_loader = object()
    layer.weight.weight_loader = resident_loader
    template = OriginalWeight(torch.empty(4, 4, device="meta"), requires_grad=False)
    template._output_dim = 0
    template._input_dim = 1
    template._weight_loader = recorded_loader
    template.tp_rank = 0
    template.tp_size = 8
    infos = {
        module: SimpleNamespace(
            restore_metadata=(
                {"weight": template} if module is layer else {},
                {},
            )
        )
        for module in model.modules()
    }
    weight_ptr = layer.weight.data_ptr()
    scale_ptr = layer.weight_scale_inv.data_ptr()

    summary = fp8_utils._restore_prequantized_fp8_parameter_metadata(
        model,
        get_layerwise_info=lambda module: infos[module],
        block_scale_parameter_cls=BlockScale,
    )

    assert isinstance(layer.weight, OriginalWeight)
    assert layer.weight.data_ptr() == weight_ptr
    assert layer.weight.output_dim == 0
    assert layer.weight.input_dim == 1
    assert layer.weight.weight_loader is resident_loader
    assert isinstance(layer.weight_scale_inv, BlockScale)
    assert layer.weight_scale_inv.data_ptr() == scale_ptr
    assert layer.weight_scale_inv.output_dim == 0
    assert layer.weight_scale_inv.input_dim == 1
    assert layer.weight_scale_inv.weight_loader is resident_loader
    assert summary == {"weights": 1, "block_scales": 1, "moe_scales": 0}


def test_prequantized_fp8_metadata_restores_moe_scale_loaders():
    class RoutedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(
                torch.ones(2, 4, 4),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.ones(2, 4, 4),
                requires_grad=False,
            )
            self.w13_weight_scale_inv = torch.nn.Parameter(
                torch.ones(2, 1, 1),
                requires_grad=False,
            )
            self.w2_weight_scale_inv = torch.nn.Parameter(
                torch.ones(2, 1, 1),
                requires_grad=False,
            )

    model = RoutedExperts()
    w13_loader = object()
    w2_loader = object()
    model.w13_weight.weight_loader = w13_loader
    model.w2_weight.weight_loader = w2_loader
    infos = {
        module: SimpleNamespace(restore_metadata=({}, {}))
        for module in model.modules()
    }

    summary = fp8_utils._restore_prequantized_fp8_parameter_metadata(
        model,
        get_layerwise_info=lambda module: infos[module],
        block_scale_parameter_cls=type("BlockScale", (torch.nn.Parameter,), {}),
    )

    assert model.w13_weight_scale_inv.weight_loader is w13_loader
    assert model.w2_weight_scale_inv.weight_loader is w2_loader
    assert model.w13_weight_scale_inv.quant_method == "block"
    assert model.w2_weight_scale_inv.quant_method == "block"
    assert summary == {"weights": 0, "block_scales": 0, "moe_scales": 2}


def test_online_reload_runs_prepare_load_finalize_and_returns_summary(
    monkeypatch,
    caplog,
):
    events = []
    real_tracker = fp8_utils.ReloadFingerprintTracker

    class Tracker(real_tracker):
        def __init__(self, model):
            events.append("snapshot")
            super().__init__(model)

        def finalize(self):
            events.append("fingerprint")
            return super().finalize()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        def load_weights(self, weights):
            events.append("load")
            loaded = set()
            for name, tensor in weights:
                loaded.add(name)
                self.get_parameter(name).data.copy_(tensor)
            return loaded

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("attn", torch.tensor([3.0, 4.0]))])

    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    fake_transfer = ModuleType(
        "lumenrl.engine.inference.bucketed_weight_transfer"
    )
    fake_transfer.BucketedWeightReceiver = Receiver
    fake_moe_sync = ModuleType(
        "lumenrl.engine.inference.vllm_moe_weight_sync"
    )
    fake_moe_sync.FusedMoEWeightRouter = object
    fake_moe_sync.assert_weight_sync_coverage = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setitem(
        sys.modules,
        "lumenrl.engine.inference.bucketed_weight_transfer",
        fake_transfer,
    )
    monkeypatch.setitem(
        sys.modules,
        "lumenrl.engine.inference.vllm_moe_weight_sync",
        fake_moe_sync,
    )
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: True)
    monkeypatch.setattr(fp8_utils, "ReloadFingerprintTracker", Tracker)
    monkeypatch.setattr(
        fp8_utils,
        "prepare_online_quantized_weights_for_loading",
        lambda model: events.append("prepare"),
    )
    monkeypatch.setattr(
        fp8_utils,
        "finalize_online_quantized_weights_loading",
        lambda model, config: events.append("finalize"),
    )

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker.model_runner = SimpleNamespace(
        model=Model(),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    with caplog.at_level(logging.WARNING):
        summary = worker.update_weights_from_ipc()

    assert events == ["snapshot", "prepare", "load", "finalize", "fingerprint"]
    assert summary["online_quant"] is True
    assert summary["buckets"] == 1
    assert summary["weights"] == 1
    assert summary["fingerprints"]["snapshotted"] == 1
    assert (
        "manifest=not aggregated, static=checked, representatives=sampled"
        in caplog.text
    )
    assert "first_source_snapshot=True" in caplog.text
    assert "exact_name_change_checks=0" in caplog.text
    assert "reload verified" not in caplog.text


def test_ipc_online_load_error_still_finalizes_and_preserves_original(
    monkeypatch,
    caplog,
):
    events = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Parameter(torch.tensor([1.0]))

        def load_weights(self, weights):
            events.append("load")
            raise RuntimeError("load failed")

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("attn", torch.tensor([2.0]))])

    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    fake_transfer = ModuleType("lumenrl.engine.inference.bucketed_weight_transfer")
    fake_transfer.BucketedWeightReceiver = Receiver
    fake_moe_sync = ModuleType("lumenrl.engine.inference.vllm_moe_weight_sync")
    fake_moe_sync.FusedMoEWeightRouter = object
    fake_moe_sync.assert_weight_sync_coverage = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setitem(
        sys.modules,
        "lumenrl.engine.inference.bucketed_weight_transfer",
        fake_transfer,
    )
    monkeypatch.setitem(
        sys.modules,
        "lumenrl.engine.inference.vllm_moe_weight_sync",
        fake_moe_sync,
    )
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: True)
    monkeypatch.setattr(
        fp8_utils,
        "prepare_online_quantized_weights_for_loading",
        lambda model: events.append("prepare"),
    )

    def fail_finalize(model, config):
        events.append("finalize")
        raise RuntimeError("finalize failed")

    monkeypatch.setattr(
        fp8_utils,
        "finalize_online_quantized_weights_loading",
        fail_finalize,
    )

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker.model_runner = SimpleNamespace(
        model=Model(),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(
        RuntimeError,
        match="load failed",
    ):
        worker.update_weights_from_ipc()

    assert events == ["prepare", "load", "finalize"]
    assert "secondary online FP8 finalize failure" in caplog.text


def test_rdma_online_orders_snapshot_prepare_load_finalize_fingerprint(
    monkeypatch,
    caplog,
):
    events = []

    class Tracker:
        def __init__(self, model):
            events.append("snapshot")
            self.probes = 0

        def checkpoint_static_changed_names(self):
            self.probes += 1
            if self.probes == 1:
                raise RuntimeError("diagnostic probe failed")
            return ["model.layers.0.ffn.gate.tid2eid"]

        def restore_checkpoint_static_tensors(self):
            events.append("restore")

        def finalize(self):
            events.append("fingerprint")
            return {"snapshotted": 1}

    def receive_weight_stream(
        *args,
        fingerprint_tracker,
        finalize_fingerprints,
        **kwargs,
    ):
        events.append("load")
        assert isinstance(fingerprint_tracker, Tracker)
        assert finalize_fingerprints is False
        return {"weights": 1.0}

    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setattr(rdma, "receive_weight_stream", receive_weight_stream)
    monkeypatch.setattr(fp8_utils, "ReloadFingerprintTracker", Tracker)
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: True)
    monkeypatch.setattr(
        fp8_utils,
        "prepare_online_quantized_weights_for_loading",
        lambda model: events.append("prepare"),
    )
    monkeypatch.setattr(
        fp8_utils,
        "finalize_online_quantized_weights_loading",
        lambda model, config: events.append("finalize"),
    )

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker._rdma_weight_groups = {"group": object()}
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    with caplog.at_level(logging.WARNING):
        summary = worker.receive_weights_rdma("group", version=4)

    assert events == [
        "snapshot",
        "prepare",
        "load",
        "finalize",
        "restore",
        "fingerprint",
    ]
    assert summary["verification"] == {"snapshotted": 1}
    assert (
        "manifest=checked, static=checked, representatives=sampled"
        in caplog.text
    )
    assert "reload verified" not in caplog.text
    assert "phase=after_prepare" in caplog.text
    assert "diagnostic probe failed" in caplog.text
    assert "phase=after_load" in caplog.text
    assert "phase=after_finalize" in caplog.text


def test_rdma_online_load_error_still_finalizes_and_preserves_original(
    monkeypatch,
    caplog,
):
    events = []

    class Tracker:
        def __init__(self, model):
            events.append("snapshot")

        def checkpoint_static_changed_names(self):
            return []

        def restore_checkpoint_static_tensors(self):
            events.append("restore")

        def finalize(self):
            events.append("fingerprint")
            return {}

    def receive_weight_stream(*args, **kwargs):
        events.append("load")
        raise RuntimeError("RDMA load failed")

    def fail_finalize(model, config):
        events.append("finalize")
        raise RuntimeError("RDMA finalize failed")

    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setattr(rdma, "receive_weight_stream", receive_weight_stream)
    monkeypatch.setattr(fp8_utils, "ReloadFingerprintTracker", Tracker)
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: True)
    monkeypatch.setattr(
        fp8_utils,
        "prepare_online_quantized_weights_for_loading",
        lambda model: events.append("prepare"),
    )
    monkeypatch.setattr(
        fp8_utils,
        "finalize_online_quantized_weights_loading",
        fail_finalize,
    )

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker._rdma_weight_groups = {"group": object()}
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(
        RuntimeError,
        match="RDMA load failed",
    ):
        worker.receive_weights_rdma("group", version=4)

    assert events == ["snapshot", "prepare", "load", "finalize", "restore"]
    assert "secondary RDMA online FP8 finalize failure" in caplog.text


def test_prequantized_rdma_requires_online_fp8_model(monkeypatch):
    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: False)

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker._rdma_weight_groups = {"group": object()}
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    with pytest.raises(
        RuntimeError,
        match="prequantized FP8 RDMA stream requires",
    ):
        worker.receive_weights_rdma(
            "group",
            version=4,
            prequantized_fp8=True,
        )


def test_prequantized_rdma_restores_sharding_metadata_before_load(monkeypatch):
    events = []

    def prepare_prequantized(model):
        events.append("prepare_prequantized")
        model.metadata_restored = True
        return {"weights": 1, "block_scales": 1}

    def receive_weight_stream(*args, streamed_scales, **kwargs):
        model = args[1]
        assert model.metadata_restored is True
        assert streamed_scales is True
        events.append("load")
        return {"weights": 2.0, "verification": {}}

    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setattr(rdma, "receive_weight_stream", receive_weight_stream)
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: True)
    monkeypatch.setattr(
        fp8_utils,
        "prepare_prequantized_fp8_weights_for_loading",
        prepare_prequantized,
        raising=False,
    )

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker._rdma_weight_groups = {"group": object()}
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(metadata_restored=False),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    summary = worker.receive_weights_rdma(
        "group",
        version=4,
        prequantized_fp8=True,
    )

    assert events == ["prepare_prequantized", "load"]
    assert summary["weights"] == 2.0


def test_rdma_online_prepare_error_still_finalizes_and_restores(monkeypatch):
    events = []

    class Tracker:
        def __init__(self, model):
            events.append("snapshot")

        def checkpoint_static_changed_names(self):
            return []

        def restore_checkpoint_static_tensors(self):
            events.append("restore")

    def fail_prepare(model):
        events.append("prepare")
        raise RuntimeError("RDMA prepare failed")

    def receive_weight_stream(*args, **kwargs):
        events.append("load")
        return {}

    fake_platforms = ModuleType("vllm.platforms")
    fake_platforms.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setattr(rdma, "receive_weight_stream", receive_weight_stream)
    monkeypatch.setattr(fp8_utils, "ReloadFingerprintTracker", Tracker)
    monkeypatch.setattr(fp8_utils, "is_online_quant_model", lambda config: True)
    monkeypatch.setattr(
        fp8_utils,
        "prepare_online_quantized_weights_for_loading",
        fail_prepare,
    )
    monkeypatch.setattr(
        fp8_utils,
        "finalize_online_quantized_weights_loading",
        lambda model, config: events.append("finalize"),
    )

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.local_rank = 0
    worker.device = torch.device("cpu")
    worker._rdma_weight_groups = {"group": object()}
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        vllm_config=SimpleNamespace(model_config=object()),
    )

    with pytest.raises(RuntimeError, match="RDMA prepare failed"):
        worker.receive_weights_rdma("group", version=4)

    assert events == ["snapshot", "prepare", "finalize", "restore"]
