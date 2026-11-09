"""DSV4.1 debug flow: an unknown harness failure is patched, then retried once."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from lumenrl.agentic.core import ToolCall  # noqa: E402
from lumenrl.agentic.rollout import ScriptedPolicy  # noqa: E402
from micake_rsi.debug_env import DebugEnvironment, DebugSettings  # noqa: E402
from micake_rsi.loop import run_generation  # noqa: E402
from sandbox.micake import MicakeGemmEnvironment  # noqa: E402
from test_rsi_loop import BASELINE, DOMAIN, FACTS, REALIZATION, _cfg, _measured, _policies, _schedule  # noqa: E402


def _settings(root: Path) -> DebugSettings:
    return DebugSettings(
        enabled=True, max_repairs_per_episode=1, max_turns=6, max_read_bytes=4000, max_file_bytes=10000,
        editable_roots=(str(root),), check_python="python3", check_cwd=str(root),
        test_prefix="experiments.cake_gemm_optimization_workflow_v1.", check_timeout_s=5,
    )


def test_edit_outside_the_root_and_arbitrary_check_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "micake"
    root.mkdir()
    source = root / "gpu_runtime.py"
    source.write_text("status = 'unknown'\n")
    outside = tmp_path / "secret.key"
    outside.write_text("nope\n")
    env = DebugEnvironment(_settings(root), {"stage": "gpu_unknown", "diagnostic": {"code": "X", "message": "boom"}})

    async def scenario() -> tuple[Any, Any, Any]:
        await env.reset({})
        escaped = await env.step(ToolCall.from_raw("a", "edit_file", json.dumps({
            "path": str(outside), "old": "nope\n", "new": "yes\n",
        })))
        relative = await env.step(ToolCall.from_raw("b", "read_file", json.dumps({"path": "gpu_runtime.py"})))
        denied = await env.step(ToolCall.from_raw("c", "run_check", json.dumps({"test": "os.system"})))
        return escaped, relative, denied

    escaped, relative, denied = asyncio.run(scenario())
    assert escaped.observation["error"] == "PATH_DENIED"
    assert outside.read_text() == "nope\n"
    assert relative.observation["error"] == "PATH_NOT_ABSOLUTE"
    assert denied.observation["error"] == "CHECK_DENIED"
    assert source.read_text() == "status = 'unknown'\n"


def test_unknown_harness_failure_is_patched_and_retried(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg["debug"]["enabled"] = True
    cfg["rsi"]["episodes_per_task"] = 1
    root = tmp_path / "micake"
    root.mkdir()
    source = root / "gpu_runtime.py"
    source.write_text("status = 'unknown'\n")
    cfg["debug"]["editable_roots"] = [str(root)]
    cfg["debug"]["require_check_before_retry"] = False

    class Flaky:
        async def describe(self) -> Mapping[str, Any]:
            return {
                "candidate_profile": "C144", "portfolio_profile": "P",
                "portfolio": [{"workload_id": "w", "m": 1, "n": 1, "k": 1}],
                "baseline_config": BASELINE, "domain": DOMAIN,
                "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
            }

        async def evaluate(self, config: Mapping[str, Any], fidelity: str, measurement: Mapping[str, Any]) -> dict[str, Any]:
            if config == BASELINE:
                return {"stage": "compiled", "compiled_facts": FACTS}
            if "refused" in source.read_text():
                return _measured(1.1)
            return {"stage": "gpu_unknown", "diagnostic": {"code": "C144_TRITON_LAUNCH_UNKNOWN", "message": "shared memory 262144"}}

        async def close(self) -> None:
            return None

    policies = _policies()
    policies["manager"] = ScriptedPolicy([
        {"name": "submit_schedule", "arguments": {"schedule": _schedule(128), "realization": REALIZATION, "hypothesis": "probe"}},
        {"name": "finish", "arguments": {"summary": "harness patched"}},
    ], model="manager")
    policies["debugger"] = ScriptedPolicy([
        {"name": "read_file", "arguments": {"path": str(source)}},
        {"name": "edit_file", "arguments": {"path": str(source), "old": "status = 'unknown'\n", "new": "status = 'refused'\n"}},
        {"name": "report_repair", "arguments": {"summary": "classify the unclassified launch", "retry": True}},
    ], model="debugger")
    micake = MicakeGemmEnvironment(**dict(cfg["micake"]["kwargs"], evaluator=Flaky()))
    manifest = asyncio.run(run_generation(cfg, micake=micake, policies=policies))

    assert manifest["metrics"]["manager"]["terminal_states"] == {"AGENT_FINISHED": 1}
    assert manifest["metrics"]["manager"]["mean_best_speedup"] == pytest.approx(1.1)
    assert manifest["metrics"]["debug"]["retries"] == 1
    assert manifest["metrics"]["debug"]["files"] == [str(source.resolve())]
    assert "debugger" not in manifest["metrics"]["roles"]
    assert "debugger" not in manifest["datasets"]
    assert "refused" in source.read_text()
    assert manifest["episodes"]["specialist"] == 1
    [repair] = manifest["repairs"]
    assert repair["outcome"] == "kept" and repair["stage_after"] == "measured"
    assert "+status = 'refused'" in Path(repair["patch"]).read_text()


def test_unfixed_unknown_stays_terminal_after_one_repair(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg["debug"]["enabled"] = True
    cfg["rsi"]["episodes_per_task"] = 1
    root = tmp_path / "micake"
    root.mkdir()
    source = root / "gpu_runtime.py"
    source.write_text("status = 'unknown'\n")
    cfg["debug"]["editable_roots"] = [str(root)]
    cfg["debug"]["require_check_before_retry"] = False

    class AlwaysUnknown:
        async def describe(self) -> Mapping[str, Any]:
            return {
                "candidate_profile": "C144", "portfolio_profile": "P",
                "portfolio": [{"workload_id": "w", "m": 1, "n": 1, "k": 1}],
                "baseline_config": BASELINE, "domain": DOMAIN,
                "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
            }

        async def evaluate(self, config: Mapping[str, Any], fidelity: str, measurement: Mapping[str, Any]) -> dict[str, Any]:
            if config == BASELINE:
                return {"stage": "compiled", "compiled_facts": FACTS}
            return {"stage": "gpu_unknown", "diagnostic": {"code": "C144_TRITON_LAUNCH_UNKNOWN", "message": "still unknown"}}

        async def close(self) -> None:
            return None

    policies = _policies()
    policies["manager"] = ScriptedPolicy([
        {"name": "submit_schedule", "arguments": {"schedule": _schedule(128), "realization": REALIZATION, "hypothesis": "probe"}},
        {"name": "finish", "arguments": {"summary": "should not run"}},
    ], model="manager")
    policies["debugger"] = ScriptedPolicy([
        {"name": "edit_file", "arguments": {"path": str(source), "old": "status = 'unknown'\n", "new": "status = 'still unknown'\n"}},
        {"name": "report_repair", "arguments": {"summary": "could not classify it", "retry": True}},
        {"name": "report_repair", "arguments": {"summary": "second session must not start", "retry": True}},
    ], model="debugger")
    micake = MicakeGemmEnvironment(**dict(cfg["micake"]["kwargs"], evaluator=AlwaysUnknown()))
    manifest = asyncio.run(run_generation(cfg, micake=micake, policies=policies))

    assert manifest["metrics"]["manager"]["terminal_states"] == {"UNKNOWN_OUTCOME": 1}
    assert manifest["metrics"]["debug"]["episodes"] == 1
    assert manifest["episodes"]["specialist"] == 1
    assert manifest["repairs"][0]["outcome"] == "rolled_back"
    assert source.read_text() == "status = 'unknown'\n"


def test_patch_guards_protect_oracle_pins_and_require_a_passing_check(tmp_path: Path) -> None:
    root = tmp_path / "micake"
    root.mkdir()
    source = root / "gpu_runtime.py"
    source.write_text("abs_tolerance: float = 0.05\nstatus = 'unknown'\n")
    oracle = root / "test_gpu_runtime.py"
    oracle.write_text("assert True\n")
    settings = DebugSettings(**{
        **_settings(root).__dict__,
        "protected_globs": ("test_*.py",),
        "pinned": {str(source.resolve()): ("abs_tolerance: float = 0.05",)},
    })
    checks: list[int] = [1, 0]

    async def runner(argv: Any, cwd: Path, timeout_s: float) -> Mapping[str, Any]:
        return {"returncode": checks.pop(0), "stdout": "", "stderr": ""}

    env = DebugEnvironment(settings, {"stage": "gpu_unknown"}, check_runner=runner)

    def call(name: str, **arguments: Any) -> ToolCall:
        return ToolCall.from_raw("c", name, json.dumps(arguments))

    async def scenario() -> list[Any]:
        await env.reset({})
        return [
            await env.step(call("edit_file", path=str(oracle), old="assert True\n", new="pass\n")),
            await env.step(call("edit_file", path=str(source), old="abs_tolerance: float = 0.05", new="abs_tolerance: float = 9.0")),
            await env.step(call("edit_file", path=str(source), old="'unknown'", new="'refused'")),
            await env.step(call("report_repair", summary="fixed", retry=True)),
            await env.step(call("run_check", test="experiments.cake_gemm_optimization_workflow_v1.test_gpu_runtime")),
            await env.step(call("report_repair", summary="fixed", retry=True)),
            await env.step(call("run_check", test="experiments.cake_gemm_optimization_workflow_v1.test_gpu_runtime")),
            await env.step(call("report_repair", summary="fixed", retry=True)),
        ]

    protected, pinned, edited, unchecked, failed, still_unchecked, passed, reported = asyncio.run(scenario())
    assert protected.observation["error"] == "PATH_PROTECTED" and oracle.read_text() == "assert True\n"
    assert pinned.observation["error"] == "CONTRACT_PIN_VIOLATED"
    assert edited.observation["bytes"] > 0
    assert unchecked.observation["error"] == "REPAIR_UNCHECKED"
    assert failed.observation["returncode"] == 1 and still_unchecked.observation["error"] == "REPAIR_UNCHECKED"
    assert passed.observation["returncode"] == 0 and reported.done and env.retry
    assert "-status = 'unknown'" in env.diff()
    assert env.rollback() == [str(source.resolve())]
    assert source.read_text() == "abs_tolerance: float = 0.05\nstatus = 'unknown'\n"
