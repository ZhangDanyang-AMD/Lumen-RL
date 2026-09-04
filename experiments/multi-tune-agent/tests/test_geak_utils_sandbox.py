import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import geak_utils.sandbox as sandbox_module
from geak_utils import CommandResult, KernelSandbox, SandboxError, TaskSpec


def make_upstream(root: Path) -> Path:
    scripts = root / "kernel_workflow" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "materialize_workspace.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (scripts / "gpu_lock.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return root


def make_task(root: Path, name: str = "task") -> Path:
    task = root / name
    task.mkdir()
    (task / "config.yaml").write_text(
        "source_file_path:\n"
        "- kernel.py\n"
        "compile_command: python3 runner.py compile\n"
        "correctness_command: python3 runner.py correctness\n"
        "performance_command: python3 runner.py performance\n",
        encoding="utf-8",
    )
    (task / "kernel.py").write_text("def kernel():\n    return 1\n", encoding="utf-8")
    (task / "test_kernel.py").write_text("def test_it():\n    pass\n", encoding="utf-8")
    scripts = task / "scripts"
    scripts.mkdir()
    (scripts / "task_runner.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    return task


def make_sandbox(tmp_path: Path) -> KernelSandbox:
    upstream = make_upstream(tmp_path / "upstream")
    task = make_task(tmp_path)
    sandbox = KernelSandbox(upstream, tmp_path / "runs", command_timeout=1)
    sandbox.workspace = task.resolve()
    sandbox.commands = sandbox._discover_commands(sandbox.workspace)
    sandbox.allowed_write_paths = sandbox._discover_write_paths(sandbox.workspace)
    return sandbox


def result(
    mode: str,
    *,
    ok: bool = True,
    cases: dict[str, float] | None = None,
) -> CommandResult:
    return CommandResult(
        mode,
        mode,
        0 if ok else 1,
        "",
        "" if ok else "failed",
        0.1,
        per_case_ms=cases or {},
    )


def test_upstream_path_errors_are_explicit(tmp_path):
    with pytest.raises(SandboxError, match="not a directory"):
        KernelSandbox(tmp_path / "missing")
    root = tmp_path / "partial"
    root.mkdir()
    with pytest.raises(SandboxError, match="materialize_workspace.sh"):
        KernelSandbox(root)
    with pytest.raises(TypeError, match="explicit upstream_root"):
        KernelSandbox()


def test_prepare_uses_upstream_materializer_and_discovers_contract(
    tmp_path, monkeypatch
):
    upstream = make_upstream(tmp_path / "upstream")
    source = make_task(tmp_path, "source")
    sandbox = KernelSandbox(upstream, tmp_path / "runs")
    calls = []

    def materialize(argv, **kwargs):
        calls.append((argv, kwargs))
        destination = Path(argv[argv.index("--dst") + 1])
        shutil.copytree(source, destination)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_module.subprocess, "run", materialize)
    workspace = sandbox.prepare(
        TaskSpec("task", "custom", source), tmp_path / "episode"
    )
    assert workspace.is_dir()
    assert calls[0][0][1] == str(sandbox.materialize_workspace)
    assert "--link-aiter" in calls[0][0]
    assert sandbox.commands["correctness"] == "python3 runner.py correctness"
    assert sandbox.allowed_write_paths == ["kernel.py"]
    with pytest.raises(SandboxError, match="already exists"):
        sandbox.prepare(TaskSpec("task", "custom", source), tmp_path / "episode")


def test_contract_discovery_and_validation(tmp_path):
    sandbox = KernelSandbox(make_upstream(tmp_path / "upstream"))
    assert sandbox._commands_from_mapping(
        {
            "correctness_command": ["python c.py"],
            "performance_command": ["python p.py", "--full"],
        }
    )["performance"] == "python p.py && --full"
    with pytest.raises(SandboxError, match="must declare"):
        sandbox._commands_from_mapping({})
    with pytest.raises(SandboxError, match="mapping"):
        sandbox._commands_from_mapping([])  # type: ignore[arg-type]

    commands = sandbox._commands_from_commandment(
        "**CORRECTNESS** — `cd /old/ws && bash /repo/gpu_lock.sh 7 python3 unittest.py --correctness`\n"
        "**FULL_BENCHMARK** — `python3 unittest.py --benchmark`\n"
    )
    assert commands["correctness"] == "python3 unittest.py --correctness"
    with pytest.raises(SandboxError, match="must expose"):
        sandbox._commands_from_commandment("nothing")

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "unittest.py").write_text("", encoding="utf-8")
    (extracted / "meta.json").write_text("{}", encoding="utf-8")
    assert sandbox._discover_commands(extracted) == {
        "correctness": "python3 unittest.py",
        "performance": "python3 unittest.py",
    }


def test_source_only_write_boundary_and_path_guards(tmp_path):
    sandbox = make_sandbox(tmp_path)
    assert "kernel.py" in sandbox.list_files()
    sandbox.write_file("kernel.py", "VALUE = 2\n")
    assert (sandbox.workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    with pytest.raises(SandboxError, match="write blocked"):
        sandbox.write_file("test_kernel.py", "bad\n")
    with pytest.raises(SandboxError, match="write blocked"):
        sandbox.write_file("scripts/task_runner.py", "bad\n")
    with pytest.raises(SandboxError, match="relative"):
        sandbox.read_file("/etc/passwd")
    with pytest.raises(SandboxError, match="escapes"):
        sandbox.list_files("../")
    with pytest.raises(SandboxError, match="not a file"):
        sandbox.read_file("missing.py")


def test_performance_output_and_report_parsing(tmp_path):
    sandbox = make_sandbox(tmp_path)
    parsed = sandbox.parse_performance(
        "Perf: 1.25 ms (small)\n"
        "GEAK_RESULT_LATENCY_MS=2.5 case=large\n"
        "Perf: 3 ms\n"
    )
    assert parsed == {"small": 1.25, "large": 2.5, "case_0": 3.0}

    report_dir = sandbox.workspace / "build"
    report_dir.mkdir()
    (report_dir / "performance_report.json").write_text(
        json.dumps(
            {
                "test_cases": [
                    {"test_case_id": "x", "execution_time_ms": 4.0},
                    {"test_case_id": "bad", "execution_time_ms": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert sandbox._parse_performance_report() == {"x": 4.0}


def test_baseline_median_and_full_evaluation(tmp_path, monkeypatch):
    sandbox = make_sandbox(tmp_path)
    results = iter(
        [
            result("compile"),
            result("correctness"),
            result("performance", cases={"x": 2.0, "y": 8.0}),
            result("performance", cases={"x": 4.0, "y": 4.0}),
            result("compile"),
            result("correctness"),
            result("performance", cases={"x": 1.5, "y": 3.0}),
        ]
    )
    monkeypatch.setattr(sandbox, "run_mode", lambda mode: next(results))
    baseline = sandbox.establish_baseline(2)
    assert baseline["per_case_ms"] == {"x": 3.0, "y": 6.0}
    assert baseline["repeats"] == 2

    evaluation = sandbox.evaluate()
    assert evaluation.compiled and evaluation.correct
    assert evaluation.speedup_geomean == pytest.approx(2.0)
    assert evaluation.speedup_arithmetic == pytest.approx(2.0)


def test_evaluation_failures_and_baseline_requirement(tmp_path, monkeypatch):
    sandbox = make_sandbox(tmp_path)
    with pytest.raises(SandboxError, match="establish_baseline"):
        sandbox.evaluate()

    sandbox.baseline_ms = {"x": 2.0}
    monkeypatch.setattr(sandbox, "run_mode", lambda mode: result("compile", ok=False))
    failed = sandbox.evaluate()
    assert not failed.compiled and failed.error == "compilation failed"

    calls = iter([result("compile"), result("correctness", ok=False)])
    monkeypatch.setattr(sandbox, "run_mode", lambda mode: next(calls))
    failed = sandbox.evaluate()
    assert failed.compiled and not failed.correct
    assert failed.error == "correctness failed"


def test_run_mode_wraps_gpu_lock_without_requiring_gpu(tmp_path, monkeypatch):
    sandbox = make_sandbox(tmp_path)
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="Perf: 5 ms (x)\n",
            stderr="",
        )

    monkeypatch.setattr(sandbox_module.subprocess, "run", run)
    measured = sandbox.run_mode("performance")
    assert measured.per_case_ms == {"x": 5.0}
    assert seen["argv"][:3] == ["bash", str(sandbox.gpu_lock), "0"]
    assert seen["kwargs"]["env"]["GEAK_GPU_ALLOWED"] == "0"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("x", 1, output=b"partial", stderr=b"late")

    monkeypatch.setattr(sandbox_module.subprocess, "run", timeout)
    timed = sandbox.run_mode("correctness")
    assert timed.timed_out and timed.returncode == 124
    assert timed.stdout == "partial"
