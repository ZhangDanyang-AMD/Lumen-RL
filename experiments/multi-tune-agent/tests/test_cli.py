from pathlib import Path

import yaml

from multi_tune_agent.cli import (
    _default_case_id,
    _has_geak_contract,
    _write_custom_catalog,
    build_parser,
)
from multi_tune_agent.config import MultiTuneConfig


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
