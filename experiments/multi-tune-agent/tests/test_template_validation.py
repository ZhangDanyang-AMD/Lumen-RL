from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from geak_utils.template_validation import (
    ValidationIssue,
    ValidationReport,
    validate_generated_template,
)


VALID_KERNEL = """
def candidate(a, b):
    return a @ b
""".lstrip()

VALID_RUNNER = r'''
import argparse
import importlib
import json
from pathlib import Path

import torch

TASK_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_ARCH = "gfx942"
RTOL = 5.0e-3
ATOL = 5.0e-3
CASES = [("small", 16, 16, 16)]


def runtime_arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


def require_gfx942():
    if runtime_arch() != SUPPORTED_ARCH:
        raise RuntimeError("unsupported architecture")


def load_kernel():
    return importlib.import_module("kernel").candidate


def make_case(index):
    torch.manual_seed(1701 + index)
    a = torch.randn((16, 16), device="cuda")
    b = torch.randn((16, 16), device="cuda")
    return a, b


def torch_reference(a, b):
    """Independent documented PyTorch reference for dense GEMM."""
    return torch.matmul(a.float(), b.float())


def compile_kernel(candidate):
    a, b = make_case(0)
    candidate(a, b)


def check_correctness(candidate):
    a, b = make_case(0)
    expected = torch_reference(a, b)
    actual = candidate(a, b)
    torch.testing.assert_close(actual.float(), expected, rtol=RTOL, atol=ATOL)


def benchmark(candidate):
    results = []
    for index, (case_id, _m, _n, _k) in enumerate(CASES):
        a, b = make_case(index)
        latency = 0.125
        candidate(a, b)
        print("Perf: %.6f ms (%s)" % (latency, case_id))
        results.append({
            "test_case_id": case_id,
            "execution_time_ms": latency,
        })
    build = TASK_DIR / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps({"test_cases": results}), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    candidate = load_kernel()
    if mode == "compile":
        compile_kernel(candidate)
    elif mode == "correctness":
        check_correctness(candidate)
    else:
        benchmark(candidate)


if __name__ == "__main__":
    main()
'''.lstrip()


def valid_metadata() -> dict:
    return {
        "name": "generated-fp16-gemm",
        "operator": "dense_gemm",
        "format": "fp16",
        "supported_arches": ["gfx942"],
        "input_dtype": "float16",
        "output_dtype": "float16",
        "contract": {
            "inputs": {
                "a": {"shape": "[M,K]", "dtype": "float16"},
                "b": {"shape": "[K,N]", "dtype": "float16"},
            },
            "output": {"shape": "[M,N]", "dtype": "float16"},
            "reference": "torch.matmul in FP32",
        },
        "provenance": {
            "generator": "template-agent",
            "source_request": "request-123",
        },
    }


def write_template(
    root: Path,
    *,
    runner: str = VALID_RUNNER,
    kernel: str = VALID_KERNEL,
    metadata: dict | None = None,
    config: str | None = None,
) -> Path:
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "kernel.py").write_text(kernel, encoding="utf-8")
    (root / "scripts" / "task_runner.py").write_text(runner, encoding="utf-8")
    if config is None:
        config = """
source_file_path:
  - kernel.py
target_kernel_functions:
  - candidate
compile_command:
  - python3 scripts/task_runner.py compile
correctness_command:
  - python3 scripts/task_runner.py correctness
performance_command:
  - python3 scripts/task_runner.py performance
task_type: triton2triton
""".lstrip()
    (root / "config.yaml").write_text(config, encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps(metadata if metadata is not None else valid_metadata()),
        encoding="utf-8",
    )
    return root


def codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_fp_template_and_public_report_api(tmp_path):
    template = write_template(tmp_path / "task")
    report = validate_generated_template(
        template,
        expected_contract={
            "operator": "dense_gemm",
            "format": "fp16",
            "supported_arches": ["gfx942"],
            "contract": {"output": {"dtype": "float16"}},
        },
    )

    assert report.valid and report.ok and bool(report)
    assert report.root == template.resolve()
    assert report.issues == report.errors == ()
    assert report.warnings == ()
    assert "location" not in ValidationIssue("x", "message").__dict__
    assert "[x]" in str(ValidationIssue("x", "message", "kernel.py", 2))


@pytest.mark.parametrize(
    ("relative", "expected_code"),
    [
        ("kernel.py", "missing-required-file"),
        ("config.yaml", "missing-required-file"),
        ("scripts/task_runner.py", "missing-required-file"),
        ("metadata.json", "missing-required-file"),
    ],
)
def test_required_files(relative, expected_code, tmp_path):
    template = write_template(tmp_path / "task")
    (template / relative).unlink()
    assert expected_code in codes(validate_generated_template(template))


def test_rejects_root_and_nested_symlinks(tmp_path):
    template = write_template(tmp_path / "task")
    (template / "linked.py").symlink_to(template / "kernel.py")
    report = validate_generated_template(template)
    assert {"symlink", "unexpected-source-file"} & codes(report)

    root_link = tmp_path / "root-link"
    root_link.symlink_to(template, target_is_directory=True)
    assert "root-symlink" in codes(validate_generated_template(root_link))


def test_rejects_unexpected_sources_and_executables(tmp_path):
    template = write_template(tmp_path / "task")
    (template / "oracle.py").write_text("pass\n", encoding="utf-8")
    helper = template / "README.txt"
    helper.write_text("not executable", encoding="utf-8")
    helper.chmod(helper.stat().st_mode | 0o100)

    report = validate_generated_template(template)
    assert "unexpected-source-file" in codes(report)
    assert "unexpected-executable" in codes(report)


@pytest.mark.parametrize(
    ("filename", "content", "expected_code"),
    [
        ("config.yaml", "source_file_path: [\n", "invalid-yaml"),
        ("metadata.json", "{oops", "invalid-json"),
        ("kernel.py", "def broken(:\n", "invalid-python"),
        ("scripts/task_runner.py", "if ):\n", "invalid-python"),
    ],
)
def test_rejects_unparseable_files(filename, content, expected_code, tmp_path):
    template = write_template(tmp_path / "task")
    (template / filename).write_text(content, encoding="utf-8")
    assert expected_code in codes(validate_generated_template(template))


@pytest.mark.parametrize(
    ("config", "expected_code"),
    [
        (
            """
source_file_path: [kernel.py, scripts/task_runner.py]
target_kernel_functions: [candidate]
compile_command: [python3 scripts/task_runner.py compile]
correctness_command: [python3 scripts/task_runner.py correctness]
performance_command: [python3 scripts/task_runner.py performance]
""",
            "unsafe-source-paths",
        ),
        (
            """
source_file_path: [kernel.py]
target_kernel_functions: []
compile_command: [python3 scripts/task_runner.py compile]
correctness_command: [python3 scripts/task_runner.py correctness]
performance_command: [python3 scripts/task_runner.py performance]
""",
            "invalid-target-functions",
        ),
        (
            """
source_file_path: [kernel.py]
target_kernel_functions: [missing]
compile_command: [python3 scripts/task_runner.py compile]
correctness_command: [python3 scripts/task_runner.py correctness]
performance_command: [python3 scripts/task_runner.py performance]
""",
            "missing-target-function",
        ),
        (
            """
source_file_path: [kernel.py]
target_kernel_functions: [candidate]
compile_command: [python3 scripts/task_runner.py performance]
correctness_command: [python3 scripts/task_runner.py correctness]
performance_command: [python3 scripts/task_runner.py performance]
""",
            "invalid-command-mode",
        ),
        (
            """
source_file_path: [kernel.py]
target_kernel_functions: [candidate]
compile_command: python3 scripts/task_runner.py compile
correctness_command: [python3 scripts/task_runner.py correctness]
performance_command: [python3 scripts/task_runner.py performance]
""",
            "invalid-command",
        ),
    ],
)
def test_config_contract(config, expected_code, tmp_path):
    template = write_template(tmp_path / "task", config=config)
    assert expected_code in codes(validate_generated_template(template))


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda value: value.pop("name"), "incomplete-metadata"),
        (lambda value: value.pop("contract"), "incomplete-contract"),
        (lambda value: value.pop("provenance"), "incomplete-provenance"),
        (
            lambda value: value.__setitem__("supported_arches", []),
            "incomplete-metadata",
        ),
        (
            lambda value: value.__setitem__(
                "provenance", {"generator": "agent"}
            ),
            "incomplete-provenance",
        ),
    ],
)
def test_metadata_requires_complete_contract_and_provenance(
    mutation, expected_code, tmp_path
):
    metadata = valid_metadata()
    mutation(metadata)
    template = write_template(tmp_path / "task", metadata=metadata)
    assert expected_code in codes(validate_generated_template(template))


def test_expected_contract_is_recursive_subset(tmp_path):
    template = write_template(tmp_path / "task")
    good = validate_generated_template(
        template, {"contract": {"output": {"dtype": "float16"}}}
    )
    bad = validate_generated_template(
        template, {"contract": {"output": {"dtype": "bfloat16"}}}
    )
    missing = validate_generated_template(template, {"contract_version": 7})
    invalid = validate_generated_template(template, expected_contract="fp16")  # type: ignore[arg-type]

    assert good.valid
    assert "contract-mismatch" in codes(bad)
    assert "contract-mismatch" in codes(missing)
    assert "invalid-expected-contract" in codes(invalid)


@pytest.mark.parametrize(
    ("old", "new", "expected_code"),
    [
        (
            'choices=("compile", "correctness", "performance")',
            'choices=("compile", "performance")',
            "runner-modes",
        ),
        (
            "torch.manual_seed(1701 + index)",
            "torch.seed()",
            "runner-seed",
        ),
        (
            'print("Perf: %.6f ms (%s)" % (latency, case_id))',
            'print("Performance", latency, case_id)',
            "runner-perf-output",
        ),
        (
            '(build / "performance_report.json").write_text(',
            '(build / "results.json").write_text(',
            "runner-performance-report",
        ),
    ],
)
def test_runner_modes_seed_and_performance_contract(
    old, new, expected_code, tmp_path
):
    runner = VALID_RUNNER.replace(old, new)
    template = write_template(tmp_path / "task", runner=runner)
    assert expected_code in codes(validate_generated_template(template))


@pytest.mark.parametrize(
    ("replacement", "expected_code"),
    [
        (
            "expected = candidate(a, b)",
            "runner-kernel-reference",
        ),
        (
            "expected = actual.clone()",
            "runner-self-reference",
        ),
        (
            "expected = torch_reference(a, b)\n    actual = candidate(a, b)\n"
            "    torch.testing.assert_close(actual, actual, rtol=RTOL, atol=ATOL)",
            "runner-self-comparison",
        ),
        (
            "expected = torch_reference(a, b)\n    actual = candidate(a, b)\n"
            "    torch.testing.assert_close(actual, expected, rtol=0.9, atol=99)",
            "runner-permissive-tolerance",
        ),
    ],
)
def test_rejects_fake_or_permissive_correctness(
    replacement, expected_code, tmp_path
):
    if "\n" in replacement:
        old = (
            "expected = torch_reference(a, b)\n"
            "    actual = candidate(a, b)\n"
            "    torch.testing.assert_close(actual.float(), expected, "
            "rtol=RTOL, atol=ATOL)"
        )
    else:
        old = "expected = torch_reference(a, b)"
    runner = VALID_RUNNER.replace(old, replacement)
    template = write_template(tmp_path / "task", runner=runner)
    assert expected_code in codes(validate_generated_template(template))


def test_allows_kernel_only_as_actual_with_independent_reference(tmp_path):
    runner = VALID_RUNNER.replace(
        'def load_kernel():\n    return importlib.import_module("kernel").candidate',
        "def load_kernel():\n"
        "    from kernel import candidate\n"
        "    return candidate",
    )
    template = write_template(tmp_path / "task", runner=runner)
    assert validate_generated_template(template).valid


def test_rejects_direct_kernel_reference_import(tmp_path):
    runner = VALID_RUNNER.replace(
        "import importlib",
        "import importlib\nfrom kernel import candidate",
    ).replace(
        "expected = torch_reference(a, b)",
        "expected = candidate(a, b)",
    )
    template = write_template(tmp_path / "task", runner=runner)
    assert "runner-kernel-reference" in codes(validate_generated_template(template))


def test_rejects_correctness_bypass_and_swallowed_exception(tmp_path):
    runner = VALID_RUNNER.replace(
        "def check_correctness(candidate):\n",
        "def check_correctness(candidate):\n    return\n",
    ).replace(
        'if __name__ == "__main__":\n    main()',
        'if __name__ == "__main__":\n'
        "    try:\n"
        "        main()\n"
        "    except Exception:\n"
        "        pass",
    )
    template = write_template(tmp_path / "task", runner=runner)
    report = validate_generated_template(template)
    assert "runner-correctness-bypass" in codes(report)
    assert "runner-swallowed-exception" in codes(report)


def test_rejects_harness_mutation_through_source_paths(tmp_path):
    runner = VALID_RUNNER.replace(
        "def benchmark(candidate):",
        'def benchmark(candidate):\n'
        '    (TASK_DIR / "config.yaml").write_text("source_file_path: []")',
    )
    template = write_template(tmp_path / "task", runner=runner)
    assert "runner-harness-mutation" in codes(validate_generated_template(template))


def test_restricted_architecture_gate_precedes_import_and_allocation(tmp_path):
    early_aiter = VALID_RUNNER.replace("import torch", "import torch\nimport aiter")
    template = write_template(tmp_path / "early-import", runner=early_aiter)
    assert "runner-early-heavy-import" in codes(validate_generated_template(template))

    late_gate = VALID_RUNNER.replace(
        "    require_gfx942()\n    candidate = load_kernel()",
        "    candidate = load_kernel()\n    require_gfx942()",
    )
    template = write_template(tmp_path / "late-gate", runner=late_gate)
    assert "runner-late-architecture-gate" in codes(
        validate_generated_template(template)
    )


def test_validator_never_executes_template(tmp_path):
    marker = tmp_path / "executed"
    kernel = (
        "from pathlib import Path\n"
        "Path(%r).write_text('bad')\n" % os.fspath(marker)
        + VALID_KERNEL
    )
    runner = (
        "from pathlib import Path\n"
        "Path(%r).write_text('bad')\n" % os.fspath(marker)
        + VALID_RUNNER
    )
    template = write_template(tmp_path / "task", kernel=kernel, runner=runner)
    validate_generated_template(template)
    assert not marker.exists()
