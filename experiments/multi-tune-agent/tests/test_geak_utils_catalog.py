from pathlib import Path

import pytest

import geak_utils
from geak_utils import TaskSpec, load_tasks
from geak_utils.paths import example_task_path, example_tasks_root
from geak_utils.task import normalize_task_type


ROOT = Path(__file__).resolve().parents[1]


def make_task(root: Path, name: str = "kernel") -> Path:
    task = root / name
    task.mkdir()
    (task / "config.yaml").write_text(
        "correctness_command: python3 check.py\n"
        "performance_command: python3 bench.py\n",
        encoding="utf-8",
    )
    (task / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    return task


def test_public_api_is_orchestration_neutral():
    assert set(geak_utils.__all__) == {
        "TaskSpec",
        "load_tasks",
        "KernelSandbox",
        "CommandResult",
        "EvaluationResult",
        "SandboxError",
        "FORMAT_ALIASES",
        "GEMM_TEMPLATES",
        "GemmTemplate",
        "architecture_for_target",
        "get_gemm_template",
        "normalize_gemm_format",
        "validate_template_target",
    }


def test_arbitrary_task_types_are_normalized_not_restricted(tmp_path):
    task = make_task(tmp_path)
    spec = TaskSpec("custom", " New-Kernel Family ", task)
    assert spec.task_type == "new_kernel_family"
    assert normalize_task_type("softmax") == "softmax"
    with pytest.raises(ValueError, match="non-empty"):
        TaskSpec("bad", "  ", task)


def test_load_tasks_resolves_relative_paths_and_aliases(tmp_path):
    task = make_task(tmp_path)
    catalog = tmp_path / "tasks.yaml"
    catalog.write_text(
        "tasks:\n"
        "  - id: custom-0\n"
        "    type: experimental op\n"
        "    kernel_path: kernel\n"
        "    max_turns: 2\n",
        encoding="utf-8",
    )
    loaded = load_tasks(catalog)
    assert loaded == [
        TaskSpec("custom-0", "experimental_op", task, max_turns=2)
    ]
    assert loaded[0].case_id == loaded[0].task_id
    assert TaskSpec(
        case_id="legacy", case_type="gemm", kernel_path=task
    ).task_id == "legacy"


def test_catalog_rejects_invalid_entries_and_duplicates(tmp_path):
    task = make_task(tmp_path)
    catalog = tmp_path / "tasks.yaml"
    for payload, error in (
        ("tasks: []\n", "non-empty"),
        ("tasks: [bad]\n", "mapping"),
        ("tasks:\n- {type: x, kernel_path: kernel}\n", "non-empty 'id'"),
        ("tasks:\n- {id: x, type: x}\n", "kernel_path"),
        (
            "tasks:\n"
            "- {id: x, type: a, kernel_path: kernel}\n"
            "- {id: x, type: b, kernel_path: kernel}\n",
            "duplicate",
        ),
    ):
        catalog.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            load_tasks(catalog)
    assert task.is_dir()


def test_task_contract_and_path_validation(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="runnable GEAK"):
        TaskSpec.from_mapping(
            {"id": "x", "type": "custom", "kernel_path": str(empty)}
        )
    with pytest.raises(ValueError, match="not a directory"):
        TaskSpec.from_mapping(
            {"id": "x", "type": "custom", "kernel_path": str(tmp_path / "missing")}
        )
    with pytest.raises(FileNotFoundError, match="task catalog not found"):
        load_tasks(tmp_path / "missing.yaml")

    commandment = tmp_path / "commandment"
    commandment.mkdir()
    (commandment / "COMMANDMENT.md").write_text("contract", encoding="utf-8")
    assert TaskSpec.from_mapping(
        {"id": "cmd", "type": "anything", "kernel_path": commandment}
    )

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "unittest.py").write_text("", encoding="utf-8")
    (extracted / "meta.json").write_text("{}", encoding="utf-8")
    assert TaskSpec.from_mapping(
        {"id": "unit", "type": "anything", "kernel_path": extracted}
    )


def test_migrated_catalog_points_to_three_clean_tasks():
    tasks = load_tasks(ROOT / "cases" / "examples_cases.yaml")
    canonical_ids = {
        "dense-gemm-fp16",
        "dense-gemm-fp8",
        "dense-gemm-mxfp4",
        "fused-attention-prefill",
        "grouped-gemm-moe",
    }
    canonical = [task for task in tasks if task.task_id in canonical_ids]
    assert [task.task_type for task in canonical] == [
        "gemm",
        "scaled_quant_gemm",
        "quant_fp4_mxfp",
        "fused_attention",
        "grouped_gemm",
    ]
    assert all((task.kernel_path / "config.yaml").is_file() for task in canonical)
    assert example_tasks_root() == (ROOT / "examples" / "tasks").resolve()
    assert example_task_path("gemm") == canonical[0].kernel_path
    assert all(
        "${GEAK_CONTAINER_NAME:-geak-phase1-vllm}"
        in (task.kernel_path / "config.yaml").read_text(encoding="utf-8")
        for task in canonical
    )
