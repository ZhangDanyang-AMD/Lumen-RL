from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import multi_tune_agent.template_bootstrap as bootstrap_module
from multi_tune_agent.template_bootstrap import (
    BootstrapError,
    KernelContract,
    TemplateBootstrapper,
    TemplateGateResult,
    promote_validated_template,
    run_template_gpu_gate,
)


KERNEL = """\
def candidate(a, b):
    return a @ b
"""

RUNNER = """\
import argparse
import importlib
import json
from pathlib import Path

import torch

TASK_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_ARCH = "gfx942"
RTOL = 5.0e-3
ATOL = 5.0e-3


def runtime_arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


def require_gpu_arch():
    if runtime_arch() != SUPPORTED_ARCH:
        raise RuntimeError("unsupported architecture")


def load_kernel():
    return importlib.import_module("kernel").candidate


def make_case(index):
    torch.manual_seed(1701 + index)
    a = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    return a, b


def torch_reference(a, b):
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
    a, b = make_case(0)
    candidate(a, b)
    latency = 0.125
    case_id = "small"
    print("Perf: %.6f ms (%s)" % (latency, case_id))
    results = [{"test_case_id": case_id, "execution_time_ms": latency}]
    build = TASK_DIR / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps({"test_cases": results}), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gpu_arch()
    candidate = load_kernel()
    if mode == "compile":
        compile_kernel(candidate)
    elif mode == "correctness":
        check_correctness(candidate)
    else:
        benchmark(candidate)


if __name__ == "__main__":
    main()
"""

CONFIG = """\
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
"""


def bundle(*, runner: str = RUNNER, metadata: object | None = None) -> dict:
    return {
        "files": {
            "kernel.py": KERNEL,
            "config.yaml": CONFIG,
            "scripts/task_runner.py": runner,
            "metadata.json": metadata
            if metadata is not None
            else {
                "name": "generated-gemm",
                "operator": "wrong",
                "format": "fp16",
                "supported_arches": ["wrong"],
                "contract": {"inputs": {}, "output": {}},
                "provenance": {"generator": "wrong", "source_request": "wrong"},
            },
        }
    }


class FakeBackend:
    model = "fake-model"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict]] = []

    def generate(self, messages, tools=()):
        assert tools == ()
        self.messages.append([dict(message) for message in messages])
        return SimpleNamespace(text=json.dumps(self.responses.pop(0)))


@pytest.fixture
def contract() -> KernelContract:
    return KernelContract(
        operator=" Dense GEMM ",
        request="Multiply activation A by weight B",
        target_gpu="MI300X",
        architecture="MI300X",
        language=" Python ",
        input_dtype="float16",
        weight_dtype="half",
        output_dtype="FP16",
        input_format="row major",
        weight_format="column-major",
        shapes=[(16, 16, 16)],
    )


def write_aiter_tree(root: Path, *, high_confidence: bool = True) -> Path:
    wrapper = root / "aiter/ops/dense/gemm/dense_gemm.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# gfx942 wrapper\n", encoding="utf-8")
    if high_confidence:
        test = root / "op_tests/dense/gemm/test_dense_gemm.py"
        test.parent.mkdir(parents=True)
        test.write_text(
            "def reference(a, b):\n    return torch.matmul(a, b)\n",
            encoding="utf-8",
        )
        benchmark = root / "benchmarks/dense/gemm/benchmark_dense_gemm.py"
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text("# benchmark\n", encoding="utf-8")
    return root


def test_direct_success_and_metadata_override(
    tmp_path: Path, contract: KernelContract
) -> None:
    backend = FakeBackend(
        [
            bundle(
                metadata={
                    "name": "model-name",
                    "operator": "malicious-override",
                    "format": "fp16",
                    "contract_hash": "wrong",
                    "contract": {"inputs": {"bad": True}, "output": {"bad": True}},
                    "provenance": {"generator": "wrong", "source_request": "wrong"},
                }
            )
        ]
    )
    draft = TemplateBootstrapper(backend, tmp_path / "drafts").generate(contract)

    assert draft.valid
    assert draft.generation_method == "llm_direct"
    assert draft.path.name == contract.contract_hash
    metadata = json.loads((draft.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["operator"] == "dense_gemm"
    assert metadata["contract_hash"] == contract.contract_hash
    assert metadata["supported_arches"] == ["gfx942"]
    assert metadata["contract"] == contract.metadata["contract"]
    assert metadata["provenance"] == {
        "generator": "multi_tune_agent.template_bootstrap",
        "source_request": contract.request,
        "generation_method": "llm_direct",
        "model": "fake-model",
        "source_artifacts": [],
        "contract_hash": contract.contract_hash,
    }


def test_invalid_direct_is_repaired_with_aiter(
    tmp_path: Path, contract: KernelContract
) -> None:
    aiter = write_aiter_tree(tmp_path / "aiter")
    backend = FakeBackend([bundle(runner="not python"), bundle()])
    draft = TemplateBootstrapper(
        backend, tmp_path / "drafts", aiter_root=aiter
    ).generate(contract)

    assert draft.valid
    assert draft.generation_method == "llm_aiter_repair"
    assert len(backend.messages) == 2
    assert "Read-only AITER evidence" in backend.messages[1][1]["content"]
    metadata = json.loads((draft.path / "metadata.json").read_text(encoding="utf-8"))
    artifacts = metadata["provenance"]["source_artifacts"]
    assert artifacts
    for artifact in artifacts:
        data = (aiter / artifact["path"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(data).hexdigest()


def test_second_aiter_repair_uses_first_repair_validation_feedback(
    tmp_path: Path, contract: KernelContract
) -> None:
    aiter = write_aiter_tree(tmp_path / "aiter")
    backend = FakeBackend(
        [bundle(runner="not python"), bundle(runner="still not python"), bundle()]
    )
    events = []

    draft = TemplateBootstrapper(
        backend, tmp_path / "drafts", aiter_root=aiter, event_sink=events.append
    ).generate(contract)

    assert draft.valid
    assert draft.generation_method == "llm_aiter_repair"
    assert len(backend.messages) == 3
    assert "invalid-python" in backend.messages[2][1]["content"]
    assert [event["phase"] for event in events].count("repair_request") == 2
    assert events[-1]["phase"] == "repair_ready"


def test_malformed_direct_response_is_repaired_with_aiter(
    tmp_path: Path, contract: KernelContract
) -> None:
    aiter = write_aiter_tree(tmp_path / "aiter")
    backend = FakeBackend([{"files": {"unexpected.py": "pass\n"}}, bundle()])

    draft = TemplateBootstrapper(
        backend, tmp_path / "drafts", aiter_root=aiter
    ).generate(contract)

    assert draft.valid
    assert draft.generation_method == "llm_aiter_repair"
    assert len(backend.messages) == 2
    assert "generation-response" in backend.messages[1][1]["content"]


def test_bundle_normalizes_scalar_config_and_runner_import_path(
    tmp_path: Path, contract: KernelContract
) -> None:
    scalar_config = CONFIG.replace(
        "source_file_path:\n  - kernel.py", "source_file_path: kernel.py"
    ).replace(
        "target_kernel_functions:\n  - candidate",
        "target_kernel_functions: candidate",
    )
    for mode in ("compile", "correctness", "performance"):
        scalar_config = scalar_config.replace(
            f"{mode}_command:\n  - python3 scripts/task_runner.py {mode}",
            f"{mode}_command: python3 scripts/task_runner.py {mode}",
        )
    response = bundle()
    response["files"]["config.yaml"] = scalar_config

    draft = TemplateBootstrapper(
        FakeBackend([response]), tmp_path / "drafts"
    ).generate(contract)

    config = yaml.safe_load((draft.path / "config.yaml").read_text(encoding="utf-8"))
    assert config["source_file_path"] == ["kernel.py"]
    assert config["target_kernel_functions"] == ["candidate"]
    assert config["compile_command"] == ["python3 scripts/task_runner.py compile"]
    runner = (draft.path / "scripts/task_runner.py").read_text(encoding="utf-8")
    assert "_bootstrap_sys.path.insert" in runner
    assert "_BootstrapPath(__file__).resolve().parents[1]" in runner


def test_bundle_infers_safe_source_and_target_function_config(
    tmp_path: Path, contract: KernelContract
) -> None:
    response = bundle()
    response["files"]["kernel.py"] = (
        KERNEL + '\nTARGET_KERNEL_FUNCTIONS = {"candidate": candidate}\n'
    )
    response["files"]["config.yaml"] = CONFIG.replace(
        "source_file_path:\n  - kernel.py\n", ""
    ).replace("target_kernel_functions:\n  - candidate\n", "")

    draft = TemplateBootstrapper(
        FakeBackend([response]), tmp_path / "drafts"
    ).generate(contract)

    config = yaml.safe_load((draft.path / "config.yaml").read_text(encoding="utf-8"))
    assert config["source_file_path"] == ["kernel.py"]
    assert config["target_kernel_functions"] == ["candidate"]


def test_low_confidence_evidence_does_not_trigger_repair(
    tmp_path: Path, contract: KernelContract
) -> None:
    aiter = write_aiter_tree(tmp_path / "aiter", high_confidence=False)
    backend = FakeBackend([bundle(runner="not python")])
    with pytest.raises(BootstrapError) as caught:
        TemplateBootstrapper(
            backend, tmp_path / "drafts", aiter_root=aiter
        ).generate(contract)

    assert len(backend.messages) == 1
    assert caught.value.validation_report is not None
    assert caught.value.draft_path is not None
    assert caught.value.draft_path.name.startswith(".failed-" + contract.contract_hash)
    assert caught.value.draft_path.is_dir()


@pytest.mark.parametrize(
    "files",
    [
        {"../escape.py": "bad"},
        {
            "kernel.py": KERNEL,
            "config.yaml": CONFIG,
            "scripts/task_runner.py": RUNNER,
            "metadata.json": {},
            "scripts/../escape.py": "bad",
        },
    ],
)
def test_malicious_or_extra_bundle_keys_are_rejected(
    tmp_path: Path, contract: KernelContract, files: dict
) -> None:
    backend = FakeBackend([{"files": files}])
    with pytest.raises(BootstrapError, match="unsafe path|missing required"):
        TemplateBootstrapper(backend, tmp_path / "drafts").generate(contract)
    assert not (tmp_path / "escape.py").exists()


def test_benign_extra_file_is_ignored_and_metadata_can_be_synthesized(
    tmp_path: Path, contract: KernelContract
) -> None:
    response = bundle()
    files = response["files"]
    files.pop("metadata.json")
    files["README.md"] = "model explanation that must not be installed"
    files["task_runner.py"] = files.pop("scripts/task_runner.py")

    draft = TemplateBootstrapper(
        FakeBackend([response]), tmp_path / "drafts"
    ).generate(contract)

    assert draft.valid
    assert not (draft.path / "README.md").exists()
    assert (draft.path / "metadata.json").is_file()
    assert (draft.path / "scripts" / "task_runner.py").is_file()


def test_existing_valid_draft_is_idempotent(
    tmp_path: Path, contract: KernelContract
) -> None:
    backend = FakeBackend([bundle()])
    bootstrapper = TemplateBootstrapper(backend, tmp_path / "drafts")
    first = bootstrapper.generate(contract)
    second = bootstrapper.generate(contract)

    assert first.path == second.path
    assert second.valid
    assert len(backend.messages) == 1


def test_contract_normalization_stable_hash_and_validation() -> None:
    first = KernelContract(
        "Dense-GEMM",
        "request",
        "MI300X",
        "mi300x",
        input_dtype="float16",
        block_size=[16, 32],
        shapes=[16, 32, 64],
    )
    second = KernelContract(
        " dense_gemm ",
        "request",
        "mi300x",
        "gfx942",
        input_dtype="fp16",
        block_size=(16, 32),
        shapes=((16, 32, 64),),
    )
    assert first == second
    assert first.contract_hash == second.contract_hash
    assert len(first.contract_hash) == 64
    with pytest.raises(FrozenInstanceError):
        first.operator = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive"):
        KernelContract("gemm", "request", "mi300x", "gfx942", shapes=[(4, 0)])
    with pytest.raises(ValueError, match="integer"):
        KernelContract("gemm", "request", "mi300x", "gfx942", shapes=[(4, 1.5)])  # type: ignore[list-item]


def test_aiter_excerpts_are_bounded(
    tmp_path: Path, contract: KernelContract
) -> None:
    aiter = write_aiter_tree(tmp_path / "aiter")
    marker = "MUST_NOT_REACH_PROMPT"
    wrapper = aiter / "aiter/ops/dense/gemm/dense_gemm.py"
    wrapper.write_text("# gfx942\n" + ("x" * 20_000) + marker, encoding="utf-8")
    backend = FakeBackend([bundle(runner="not python"), bundle()])

    TemplateBootstrapper(backend, tmp_path / "drafts", aiter_root=aiter).generate(
        contract
    )

    repair_prompt = backend.messages[1][1]["content"]
    assert marker not in repair_prompt
    assert "...[truncated]" in repair_prompt
    assert len(repair_prompt) < 40_000


def make_draft(tmp_path: Path, contract: KernelContract):
    return TemplateBootstrapper(
        FakeBackend([bundle()]), tmp_path / "drafts"
    ).generate(contract)


def command_result(
    mode: str,
    *,
    ok: bool = True,
    per_case_ms: dict[str, float] | None = None,
):
    return SimpleNamespace(
        mode=mode,
        command="run-" + mode,
        returncode=0 if ok else 1,
        timed_out=False,
        stdout="output-" + mode,
        stderr="" if ok else "failed-" + mode,
        ok=ok,
        per_case_ms=per_case_ms or {},
    )


class FakeSandbox:
    outcomes: list[object] = []
    calls: list[str] = []
    init_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).init_kwargs = dict(kwargs)

    def prepare(self, task, episode):
        self.task = task
        self.episode = Path(episode)
        workspace = self.episode / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    def run_mode(self, mode):
        type(self).calls.append(mode)
        outcome = type(self).outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def install_fake_sandbox(monkeypatch, outcomes):
    FakeSandbox.outcomes = list(outcomes)
    FakeSandbox.calls = []
    FakeSandbox.init_kwargs = {}
    monkeypatch.setattr(bootstrap_module, "KernelSandbox", FakeSandbox)


def test_gpu_gate_runs_locked_modes_in_order(
    tmp_path: Path, contract: KernelContract, monkeypatch
) -> None:
    draft = make_draft(tmp_path, contract)
    install_fake_sandbox(
        monkeypatch,
        [
            command_result("compile"),
            command_result("correctness"),
            command_result("performance", per_case_ms={"large": 0.25, "small": 0.1}),
        ],
    )

    result = run_template_gpu_gate(
        draft,
        geak_root=tmp_path / "geak",
        run_root=tmp_path / "runs",
        gpu_ids="3",
        command_timeout=17,
    )

    assert result.trusted
    assert FakeSandbox.calls == ["compile", "correctness", "performance"]
    assert dict(result.per_case_ms) == {"large": 0.25, "small": 0.1}
    assert list(result.command_summaries) == [
        "compile",
        "correctness",
        "performance",
    ]
    assert FakeSandbox.init_kwargs["gpu_ids"] == "3"
    assert FakeSandbox.init_kwargs["command_timeout"] == 17
    assert result.validation_workspace is not None
    assert result.validation_workspace.parent.name.startswith(
        "template-validation-" + contract.contract_hash[:16]
    )


@pytest.mark.parametrize(
    ("outcomes", "expected_calls", "compiled", "correct"),
    [
        ([command_result("compile", ok=False)], ["compile"], False, False),
        (
            [command_result("compile"), command_result("correctness", ok=False)],
            ["compile", "correctness"],
            True,
            False,
        ),
    ],
)
def test_gpu_gate_stops_after_command_failure(
    tmp_path: Path,
    contract: KernelContract,
    monkeypatch,
    outcomes,
    expected_calls,
    compiled,
    correct,
) -> None:
    draft = make_draft(tmp_path, contract)
    install_fake_sandbox(monkeypatch, outcomes)

    result = run_template_gpu_gate(
        draft, geak_root=tmp_path / "geak", run_root=tmp_path / "runs"
    )

    assert FakeSandbox.calls == expected_calls
    assert result.compiled is compiled
    assert result.correct is correct
    assert not result.performance_valid
    assert not result.trusted
    assert result.errors


@pytest.mark.parametrize("latencies", [{}, {"bad": 0.0}, {"bad": float("nan")}])
def test_gpu_gate_rejects_invalid_performance_evidence(
    tmp_path: Path, contract: KernelContract, monkeypatch, latencies
) -> None:
    draft = make_draft(tmp_path, contract)
    install_fake_sandbox(
        monkeypatch,
        [
            command_result("compile"),
            command_result("correctness"),
            command_result("performance", per_case_ms=latencies),
        ],
    )

    result = run_template_gpu_gate(
        draft, geak_root=tmp_path / "geak", run_root=tmp_path / "runs"
    )

    assert result.compiled and result.correct
    assert not result.performance_valid
    assert not result.trusted
    assert result.errors


def test_gpu_gate_returns_sandbox_setup_error(
    tmp_path: Path, contract: KernelContract, monkeypatch
) -> None:
    draft = make_draft(tmp_path, contract)

    class BrokenSandbox:
        def __init__(self, **kwargs):
            raise RuntimeError("configuration unavailable")

    monkeypatch.setattr(bootstrap_module, "KernelSandbox", BrokenSandbox)
    result = run_template_gpu_gate(
        draft, geak_root=tmp_path / "geak", run_root=tmp_path / "runs"
    )

    assert result.static_valid
    assert not result.trusted
    assert result.command_summaries == {}
    assert result.errors == (
        "sandbox setup failed: RuntimeError: configuration unavailable",
    )


def trusted_gate(workspace: Path) -> TemplateGateResult:
    return TemplateGateResult(
        static_valid=True,
        compiled=True,
        correct=True,
        performance_valid=True,
        per_case_ms={"small": 0.125},
        command_summaries={
            "compile": {
                "mode": "compile",
                "command": "compile",
                "returncode": 0,
                "timed_out": False,
                "ok": True,
                "stdout": "",
                "stderr": "",
            },
            "correctness": {
                "mode": "correctness",
                "command": "correctness",
                "returncode": 0,
                "timed_out": False,
                "ok": True,
                "stdout": "",
                "stderr": "",
            },
            "performance": {
                "mode": "performance",
                "command": "performance",
                "returncode": 0,
                "timed_out": False,
                "ok": True,
                "stdout": "Perf: 0.125 ms (small)",
                "stderr": "",
            },
        },
        validation_workspace=workspace,
    )


def test_promotion_refuses_untrusted_gate(
    tmp_path: Path, contract: KernelContract
) -> None:
    draft = make_draft(tmp_path, contract)
    untrusted = TemplateGateResult(
        static_valid=True,
        compiled=True,
        correct=True,
        performance_valid=False,
        per_case_ms={},
        errors=("missing latency",),
    )
    with pytest.raises(BootstrapError, match="trusted"):
        promote_validated_template(draft, untrusted, tmp_path / "verified")
    assert not (tmp_path / "verified" / contract.contract_hash).exists()


def test_promotion_success_idempotency_and_mismatch_rejection(
    tmp_path: Path, contract: KernelContract
) -> None:
    draft = make_draft(tmp_path, contract)
    original_metadata = (draft.path / "metadata.json").read_bytes()
    gate = trusted_gate(tmp_path / "validation-workspace")
    verified_root = tmp_path / "verified"

    promoted = promote_validated_template(draft, gate, verified_root)
    repeated = promote_validated_template(draft, gate, verified_root)

    assert repeated == promoted == verified_root / contract.contract_hash
    assert (draft.path / "metadata.json").read_bytes() == original_metadata
    assert {
        path.relative_to(promoted).as_posix()
        for path in promoted.rglob("*")
        if path.is_file()
    } == {
        "kernel.py",
        "config.yaml",
        "scripts/task_runner.py",
        "metadata.json",
    }
    metadata = json.loads((promoted / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["trust"]["trusted"] is True
    assert metadata["trust"]["per_case_ms"] == {"small": 0.125}
    assert "timestamp" not in metadata["trust"]
    (promoted / "kernel.py").write_text("def different(): pass\n", encoding="utf-8")
    with pytest.raises(BootstrapError, match="mismatched"):
        promote_validated_template(draft, gate, verified_root)
