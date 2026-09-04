import json
from pathlib import Path

import pytest
import yaml

from geak_utils.local_templates import (
    VerifiedTemplateRecord,
    register_verified_template,
)
from multi_tune_agent import cli
from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.task_factory import GeneratedKernelTask


def _config(tmp_path, **updates):
    values = {
        "geak_root": tmp_path / "geak",
        "cases_path": tmp_path / "cases.yaml",
        "trajectory_root": tmp_path / "runs",
        "generated_template_root": tmp_path / "generated",
        "aiter_root": tmp_path / "aiter",
    }
    values.update(updates)
    return MultiTuneConfig(**values)


def _recognized(**updates):
    value = {
        "operator": "gemm",
        "target_gpu": "gfx942",
        "format": "fp8",
        "input_dtype": None,
        "weight_dtype": None,
        "output_dtype": None,
        "input_scale_granularity": "per_tensor",
        "weight_scale_granularity": "per_channel",
        "block_size": None,
        "dimensions": {"m": 16, "n": 32, "k": 64},
        "shapes": [[16, 32, 64]],
        "m": 16,
        "n": 32,
        "k": 64,
        "language": "triton",
        "explicit_fields": ["format"],
    }
    value.update(updates)
    return value


def _verified_bundle(path, contract):
    path.mkdir(parents=True)
    (path / "scripts").mkdir()
    (path / "kernel.py").write_text("def kernel():\n    pass\n")
    (path / "config.yaml").write_text("source_file_path: [kernel.py]\n")
    (path / "scripts" / "task_runner.py").write_text("pass\n")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                **contract.metadata,
                "provenance": {"generation_method": "test"},
                "trust": {"trusted": True},
            }
        )
    )


def test_local_verified_reuse_skips_generation_and_gpu(tmp_path, monkeypatch):
    config = _config(tmp_path)
    request = "gfx942 FP8 GEMM M=16 N=32 K=64 per-tensor input per-channel weight"
    recognized = _recognized()
    contract = cli._kernel_contract(recognized, request)
    template = config.generated_template_root / contract.contract_hash
    _verified_bundle(template, contract)
    record = VerifiedTemplateRecord(
        contract_hash=contract.contract_hash,
        operator="gemm",
        template_path=template,
        architecture="gfx942",
        language="triton",
        backend="triton",
        provenance={"generation_method": "test"},
        direction=request,
    )
    register_verified_template(
        config.generated_template_root / "templates.yaml",
        record,
        verified_root=config.generated_template_root,
    )
    monkeypatch.setattr(
        cli,
        "TemplateBootstrapper",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generated")),
    )
    monkeypatch.setattr(
        cli,
        "run_template_gpu_gate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GPU ran")),
    )

    task = cli._bootstrap_kernel_task(
        config,
        object(),
        request,
        recognized,
        input_fn=lambda _: "",
        output_fn=lambda _: None,
    )

    assert isinstance(task, GeneratedKernelTask)
    assert yaml.safe_load(config.cases_path.read_text())["tasks"][0][
        "contract_hash"
    ] == contract.contract_hash


def test_failed_gate_never_catalogs(tmp_path, monkeypatch):
    config = _config(tmp_path)
    draft = object()

    class Bootstrapper:
        def __init__(self, **kwargs):
            pass

        def generate(self, contract):
            return draft

    class Gate:
        trusted = False
        errors = ("correctness failed",)
        validation_workspace = tmp_path / "diagnostics"

    monkeypatch.setattr(cli, "TemplateBootstrapper", Bootstrapper)
    monkeypatch.setattr(cli, "run_template_gpu_gate", lambda *args, **kwargs: Gate())
    monkeypatch.setattr(
        cli,
        "promote_validated_template",
        lambda *args: (_ for _ in ()).throw(AssertionError("promoted")),
    )

    task = cli._bootstrap_kernel_task(
        config,
        object(),
        "request",
        _recognized(),
        input_fn=lambda _: "",
        output_fn=lambda _: None,
    )

    assert task is None
    assert not config.cases_path.exists()
    assert not (config.generated_template_root / "templates.yaml").exists()


def test_native_mxfp8_rejects_gfx942_before_generation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        cli,
        "TemplateBootstrapper",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generation must not start")
        ),
    )

    with pytest.raises(ValueError, match="requires native CDNA4"):
        cli._bootstrap_kernel_task(
            config,
            object(),
            "MI308 MXFP8 GEMM M=1 N=128 K=128",
            _recognized(
                format="mxfp8",
                input_scale_granularity="per_block",
                weight_scale_granularity="per_block",
                block_size=32,
            ),
            input_fn=lambda _: "",
            output_fn=lambda _: None,
        )


def test_trusted_gate_promotes_and_registers(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = []

    class Draft:
        def __init__(self, contract):
            self.contract = contract

    class Bootstrapper:
        def __init__(self, **kwargs):
            pass

        def generate(self, contract):
            return Draft(contract)

    class Gate:
        trusted = True
        errors = ()
        validation_workspace = tmp_path / "diagnostics"

    def promote(draft, gate, root):
        calls.append("promote")
        destination = Path(root) / draft.contract.contract_hash
        destination.mkdir(parents=True)
        (destination / "metadata.json").write_text(
            json.dumps(
                {
                    "contract_hash": draft.contract.contract_hash,
                    "provenance": {"generation_method": "llm_direct"},
                }
            )
        )
        return destination

    def register(path, record, **kwargs):
        calls.append("register")
        return record

    def materialize(config, record, **kwargs):
        calls.append("catalog")
        return GeneratedKernelTask(
            case_id="generated",
            task_dir=record.template_path,
            case_type="aiter_generated",
            operator=record.operator,
            architecture=record.architecture,
            backend=record.backend,
            request=kwargs["request"],
            contract_hash=record.contract_hash,
            provenance=record.provenance,
        )

    monkeypatch.setattr(cli, "TemplateBootstrapper", Bootstrapper)
    monkeypatch.setattr(cli, "run_template_gpu_gate", lambda *args, **kwargs: Gate())
    monkeypatch.setattr(cli, "promote_validated_template", promote)
    monkeypatch.setattr(cli, "register_verified_template", register)
    monkeypatch.setattr(cli, "materialize_verified_template_task", materialize)

    task = cli._bootstrap_kernel_task(
        config,
        object(),
        "request",
        _recognized(),
        input_fn=lambda _: (_ for _ in ()).throw(
            AssertionError("auto promotion must not prompt")
        ),
        output_fn=lambda _: None,
    )

    assert isinstance(task, GeneratedKernelTask)
    assert calls == ["promote", "register", "catalog"]


def test_generic_metadata_is_exposed_safely(tmp_path):
    from multi_tune_agent.geak_tool import GEAKToolEnvironment

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "config.yaml").write_text("source_file_path: [kernel.py]\n")
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "operator": "softmax",
                "architecture": "gfx942",
                "contract_hash": "a" * 64,
                "contract": {"input": {"dtype": "fp16"}},
                "provenance": {"generation_method": "llm_direct"},
                "trust": {"trusted": True},
            }
        )
    )
    catalog = tmp_path / "cases.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "generated",
                        "type": "aiter_generated",
                        "kernel_path": str(task_dir),
                    }
                ]
            }
        )
    )
    observation = GEAKToolEnvironment(
        _config(tmp_path, cases_path=catalog)
    ).case_observation("generated")
    assert observation["contract"]["input"]["dtype"] == "fp16"
    assert observation["provenance"]["generation_method"] == "llm_direct"
