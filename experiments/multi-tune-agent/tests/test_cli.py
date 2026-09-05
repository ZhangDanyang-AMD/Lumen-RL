import json
from pathlib import Path
from types import SimpleNamespace

import yaml
import multi_tune_agent.cli as cli

from multi_tune_agent.cli import (
    _default_case_id,
    _has_geak_contract,
    _prompt_gemm_supplement,
    _write_custom_catalog,
    build_parser,
)
from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.task_factory import GeneratedKernelTask


def test_parser_exposes_interactive_and_request_modes():
    parser = build_parser()
    interactive = parser.parse_args(["--config", "config.yaml", "interactive"])
    assert interactive.command == "interactive"

    run = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "run",
            "--case",
            "demo",
            "--request",
            "optimize M=1234 N=2048 K=4096",
            "--stream",
        ]
    )
    assert run.command == "run"
    assert run.cases == ["demo"]
    assert run.request.startswith("optimize")
    assert run.stream is True

    generate = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "generate",
            "--manifest",
            "requests.yaml",
            "--output-catalog",
            "cases.yaml",
            "--stream",
        ]
    )
    assert generate.command == "generate"
    assert generate.manifest == Path("requests.yaml")
    assert generate.output_catalog == Path("cases.yaml")
    assert generate.stream is True


def test_generation_manifest_records_each_noninteractive_result(tmp_path, monkeypatch):
    config = MultiTuneConfig(
        geak_root=tmp_path,
        cases_path=tmp_path / "unused.yaml",
        trajectory_root=tmp_path / "runs",
        bootstrap_auto_promote=True,
    )
    manifest = tmp_path / "requests.yaml"
    manifest.write_text(
        "version: 1\nrequests:\n"
        "  - id: hip-gfx942-demo\n"
        "    request: Generate HIP GEMM on gfx942 M=1 N=2 K=3 FP16\n"
        "    seed_provenance:\n"
        "      source_repo: https://github.com/ROCm/aiter.git\n"
        "      source_sha: abc123\n"
    )
    catalog = tmp_path / "cases.yaml"
    seen = []
    task = GeneratedKernelTask(
        case_id="hip-gfx942-demo",
        task_dir=tmp_path / "task",
        case_type="gemm",
        operator="gemm",
        architecture="gfx942",
        backend="hip",
        request="Generate HIP GEMM on gfx942 M=1 N=2 K=3 FP16",
        contract_hash="contract",
        provenance={"template": "generated"},
    )
    monkeypatch.setattr(
        cli,
        "_generate_kernel_task_noninteractive",
        lambda config, backend, **kwargs: task,
    )

    def register(path, value):
        seen.append((path, value))
        path.write_text("tasks:\n  - id: hip-gfx942-demo\n")

    monkeypatch.setattr(cli, "register_generated_case", register)
    assert (
        cli._run_generation_manifest(
            config, object(), manifest, catalog, stream=False
        )
        == 0
    )
    assert seen[0][0] == catalog.resolve()
    assert seen[0][1].provenance["template"] == "generated"
    assert seen[0][1].provenance["case_seed"]["source_sha"] == "abc123"
    result_path = (
        config.trajectory_root
        / "requests"
        / "requests-generation-results.jsonl"
    )
    result = json.loads(result_path.read_text().strip())
    assert result["case_id"] == "hip-gfx942-demo"
    assert result["status"] == "generated"
    assert result["seed_provenance"]["source_repo"] == (
        "https://github.com/ROCm/aiter.git"
    )


def test_custom_catalog_records_task_and_request(tmp_path):
    task = tmp_path / "MI308 GEMM"
    task.mkdir()
    (task / "config.yaml").write_text("source_file_path: [kernel.py]\n")
    config = MultiTuneConfig(
        geak_root=tmp_path,
        cases_path=tmp_path / "unused.yaml",
        trajectory_root=tmp_path / "runs",
    )

    assert _has_geak_contract(task)
    assert _default_case_id(task) == "MI308-GEMM"
    path = _write_custom_catalog(
        config,
        case_id="mi308-gemm",
        case_type="gemm",
        kernel_path=task,
        request="Optimize M=1234 N=2048 K=4096.",
    )
    payload = yaml.safe_load(path.read_text())
    case = payload["tasks"][0]
    assert case["id"] == "mi308-gemm"
    assert case["kernel_path"] == str(task.resolve())
    assert case["direction"] == "Optimize M=1234 N=2048 K=4096."


def test_missing_gemm_details_are_collected_as_a_supplement():
    messages = []
    answers = iter(["", "M=1 N=128 K=256"])
    request = _prompt_gemm_supplement(
        "MI308 FP8 GEMM",
        "recognizer could not identify M",
        input_fn=lambda _prompt: next(answers),
        output_fn=messages.append,
    )

    assert request == "MI308 FP8 GEMM; user supplement: M=1 N=128 K=256"
    assert messages[0].startswith("More information is needed:")
    assert "Please enter" in messages[1]


def test_missing_gemm_details_can_return_to_menu():
    assert (
        _prompt_gemm_supplement(
            "GEMM",
            "missing target GPU",
            input_fn=lambda _prompt: "back",
            output_fn=lambda _message: None,
        )
        is None
    )


def test_explicit_fp8_contract_skips_format_menu(tmp_path, monkeypatch):
    config = MultiTuneConfig(
        geak_root=tmp_path,
        cases_path=tmp_path / "cases.yaml",
        trajectory_root=tmp_path / "runs",
    )
    recognized = {
        "operator": "gemm",
        "target_gpu": "gfx942",
        "format": "fp8",
        "input_dtype": None,
        "weight_dtype": None,
        "output_dtype": None,
        "input_scale_granularity": "per_token",
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
    sentinel = object()
    monkeypatch.setattr(cli, "recognize_kernel_request", lambda *args: recognized)
    monkeypatch.setattr(cli, "generate_gemm_task", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        cli,
        "_prompt_gemm_format",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("format menu")),
    )
    answers = iter(["gfx942 FP8 GEMM M=16 N=32 K=64 scales", ""])
    assert (
        cli._generate_kernel_task_interactive(
            config,
            object(),
            input_fn=lambda _: next(answers),
            output_fn=lambda _: None,
        )
        is sentinel
    )


def test_missing_format_and_fp8_scales_are_reported():
    base = {
        "operator": "gemm",
        "target_gpu": "gfx942",
        "dimensions": {"m": 1, "n": 2, "k": 3},
        "shapes": [[1, 2, 3]],
        "format": None,
        "input_dtype": None,
    }
    assert cli._missing_kernel_fields(base) == ["format/input dtype"]
    assert cli._missing_kernel_fields({**base, "format": "fp8"}) == [
        "input scale granularity",
        "weight scale granularity",
    ]
    assert (
        cli._missing_kernel_fields(
            {
                **base,
                "format": "mxfp8",
                "input_scale_granularity": "per_block",
                "weight_scale_granularity": "per_block",
            }
        )
        == []
    )


def test_canonical_template_cannot_override_explicit_hip_language():
    descriptor = SimpleNamespace(backend="triton")
    assert not cli._template_matches_requested_language(
        descriptor, {"language": "hip"}
    )
    assert cli._template_matches_requested_language(
        descriptor, {"language": "triton"}
    )
