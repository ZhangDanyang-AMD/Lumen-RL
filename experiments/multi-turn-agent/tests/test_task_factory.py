from pathlib import Path

import yaml
from geak_utils.paths import example_task_path

from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.task_factory import (
    generate_gemm_task,
    parse_gemm_request,
    register_generated_case,
)


def _config(tmp_path):
    geak = tmp_path / "geak"
    geak.mkdir()
    return MultiTuneConfig(
        geak_root=geak,
        cases_path=tmp_path / "cases" / "catalog.yaml",
        trajectory_root=tmp_path / "runs",
    )


def test_parse_gemm_request_accepts_chinese_mnk_format():
    spec = parse_gemm_request(
        "优化 MI308X 上的 FP16 GEMM kernel，MNK分别是1234、2048 4096"
    )
    assert spec["target_gpu"] == "MI308X"
    assert (spec["m"], spec["n"], spec["k"]) == (1234, 2048, 4096)
    assert spec["dtype"] == "fp16"


def test_parse_gemm_request_tolerates_pinyin_and_loose_separators():
    spec = parse_gemm_request("MI308上做一个GEMM，MNK分别 shi 1/128/128")
    assert spec["target_gpu"] == "MI308"
    assert (spec["m"], spec["n"], spec["k"]) == (1, 128, 128)
    omitted_mnk = parse_gemm_request("请在MI300上做一个GEMM，分别是3/128/128")
    assert (omitted_mnk["m"], omitted_mnk["n"], omitted_mnk["k"]) == (3, 128, 128)


def test_parse_gemm_request_accepts_bare_correction_and_fullwidth_text():
    assert parse_gemm_request("１／１２８／１２８")["m"] == 1
    spec = parse_gemm_request("MNK fp16 are 1 x 128 x 256")
    assert (spec["m"], spec["n"], spec["k"]) == (1, 128, 256)


def test_generate_and_register_gemm_task(tmp_path):
    config = _config(tmp_path)
    task = generate_gemm_task(
        config,
        "Optimize MI308X FP16 GEMM M=1234, N=2048, K=4096",
        task_root=tmp_path / "local-tasks",
    )
    assert task.task_dir.is_dir()
    assert (task.task_dir / "kernel.py").is_file()
    assert (task.task_dir / "config.yaml").is_file()
    runner = (task.task_dir / "scripts" / "task_runner.py").read_text()
    assert "CASES = [('m1234_n2048_k4096', 1234, 2048, 4096)]" in runner
    compile(runner, str(task.task_dir / "scripts" / "task_runner.py"), "exec")

    register_generated_case(config.cases_path, task)
    payload = yaml.safe_load(config.cases_path.read_text())
    case = payload["tasks"][0]
    assert case["id"] == task.case_id
    assert case["type"] == "gemm"
    assert (config.cases_path.parent / case["kernel_path"]).resolve() == task.task_dir
    manifest = yaml.safe_load((task.task_dir / "task_spec.json").read_text())
    assert manifest["generated_from"] == str(example_task_path("gemm"))
    generated_config = (task.task_dir / "config.yaml").read_text()
    assert "${GEAK_CONTAINER_NAME:-geak-phase1-vllm}" in generated_config
