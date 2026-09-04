import json
from pathlib import Path

import pytest
import yaml

from geak_utils import (
    GEMM_TEMPLATES,
    architecture_for_target,
    get_gemm_template,
    validate_template_target,
)
from geak_utils.catalog import load_tasks
from multi_tune_agent.agents import RolePromptLibrary
from multi_tune_agent.cli import _prompt_gemm_format
from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.request_parser import recognize_gemm_request
from multi_tune_agent.runtime import ModelTurn
from multi_tune_agent.task_factory import (
    generate_gemm_task,
    parse_gemm_request,
    register_generated_case,
)


ROOT = Path(__file__).resolve().parents[1]


class JsonBackend:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, tools=()):
        text = json.dumps(self.payload)
        return ModelTurn(text, {"role": "assistant", "content": text}, [], [], {})


def model_payload(**updates):
    payload = {
        "operator": "gemm",
        "target_gpu": "MI308X",
        "format": "fp16",
        "dtype": "fp16",
        "m": 16,
        "n": 32,
        "k": 64,
        "language": "triton",
        "confidence": 0.9,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("fp8", "fp8"),
        ("float8", "fp8"),
        ("e4m3", "fp8"),
        ("e4m3fnuz", "fp8"),
        ("a8w8", "fp8"),
        ("mxfp4", "mxfp4"),
        ("mx-fp4", "mxfp4"),
        ("fp4", "mxfp4"),
        ("e2m1", "mxfp4"),
    ],
)
def test_deterministic_parser_normalizes_quantized_aliases(alias, expected):
    result = parse_gemm_request("MI308 GEMM M=16 N=32 K=64 %s" % alias)
    assert result["dtype"] == expected


def test_model_cannot_hallucinate_format_or_language():
    result = recognize_gemm_request(
        "MI308 GEMM M=16 N=32 K=64",
        JsonBackend(model_payload(format="mxfp4", dtype="mxfp4", language="flydsl")),
    )
    assert result["dtype"] == "fp16"
    assert result["language"] == "triton"

    evidenced = recognize_gemm_request(
        "MI308 FP8 GEMM M=16 N=32 K=64 in HIP",
        JsonBackend(model_payload(format="fp16", dtype="fp16", language="triton")),
    )
    assert evidenced["dtype"] == "fp8"
    assert evidenced["language"] == "hip"


def test_template_registry_maps_and_rejects_architectures():
    assert tuple(GEMM_TEMPLATES) == ("fp16", "fp8", "mxfp4")
    assert architecture_for_target("MI300X") == "gfx942"
    assert architecture_for_target("mi308") == "gfx942"
    assert architecture_for_target("MI350X") == "gfx950"
    assert architecture_for_target("mi355") == "gfx950"
    assert get_gemm_template("e2m1").case_type == "quant_fp4_mxfp"
    with pytest.raises(ValueError, match="unsupported"):
        validate_template_target("mxfp4", "MI308X")
    with pytest.raises(ValueError, match="unsupported known"):
        architecture_for_target("MI250X")
    with pytest.raises(ValueError, match="unknown GPU"):
        architecture_for_target("mystery")


def test_format_prompt_default_invalid_and_unsupported_retry():
    answers = iter(["", "2"])
    messages = []
    selected = _prompt_gemm_format(
        "mxfp4",
        "MI308X",
        input_fn=lambda _: next(answers),
        output_fn=messages.append,
    )
    assert selected == "fp8"
    assert any("unsupported" in message.lower() for message in messages)

    assert (
        _prompt_gemm_format(
            "fp8",
            "MI308X",
            input_fn=lambda _: "",
            output_fn=lambda _: None,
        )
        == "fp8"
    )


def test_quantized_generation_manifest_catalog_and_single_shape_runner(tmp_path):
    config = MultiTuneConfig(
        geak_root=tmp_path,
        cases_path=tmp_path / "catalog.yaml",
        trajectory_root=tmp_path / "runs",
    )
    cases = (
        ("fp8", "MI308X", "scaled_quant_gemm", "triton", 64),
        ("mxfp4", "MI355X", "quant_fp4_mxfp", "aiter", 64),
    )
    for format_name, gpu, case_type, backend, k in cases:
        task = generate_gemm_task(
            config,
            "%s %s GEMM M=16 N=32 K=%d" % (gpu, format_name, k),
            task_root=tmp_path / "tasks",
        )
        assert task.case_type == case_type
        assert task.backend == backend
        manifest = json.loads((task.task_dir / "task_spec.json").read_text())
        assert manifest["format"] == format_name
        assert manifest["backend"] == backend
        assert manifest["architecture"] in get_gemm_template(
            format_name
        ).supported_architectures
        assert manifest["scale_contract"]
        runner = (task.task_dir / "scripts" / "task_runner.py").read_text()
        assert "task_spec.json" in runner
        assert "len(shapes) != 1" in runner
        register_generated_case(config.cases_path, task)

    entries = yaml.safe_load(config.cases_path.read_text())["tasks"]
    assert [entry["type"] for entry in entries] == [
        "scaled_quant_gemm",
        "quant_fp4_mxfp",
    ]
    assert all(entry["format"] in {"fp8", "mxfp4"} for entry in entries)


def test_generation_rejects_architecture_before_writing(tmp_path):
    config = MultiTuneConfig(
        geak_root=tmp_path,
        cases_path=tmp_path / "catalog.yaml",
        trajectory_root=tmp_path / "runs",
    )
    root = tmp_path / "tasks"
    with pytest.raises(ValueError, match="unsupported"):
        generate_gemm_task(
            config,
            "MI308X MXFP4 GEMM M=16 N=32 K=64",
            task_root=root,
        )
    assert not root.exists()


def test_canonical_quantized_catalog_and_knowledge_paths():
    tasks = {task.task_id: task for task in load_tasks(ROOT / "cases/examples_cases.yaml")}
    assert tasks["dense-gemm-fp8"].task_type == "scaled_quant_gemm"
    assert tasks["dense-gemm-mxfp4"].task_type == "quant_fp4_mxfp"

    geak_root = ROOT.parents[2] / "GEAK"
    prompts = RolePromptLibrary(geak_root)
    for case_type in ("scaled_quant_gemm", "quant_fp4_mxfp"):
        operator = geak_root / "perf_knowledge" / "operators" / case_type
        assert (operator / "overview.md").is_file()
        text = prompts.system("engineer", case_type)
        assert ("%s/overview.md" % case_type) in text
