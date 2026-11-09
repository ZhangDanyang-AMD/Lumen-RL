from __future__ import annotations

import asyncio
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments" / "sandbox"))

from lumenrl.agentic.core import ToolCall, build_environment  # noqa: E402
from sandbox.micake import BACKEND_NAME, MicakeGemmEnvironment  # noqa: E402
from sandbox.micake.tool_runtime import ToolArgumentError, parse_submission, tool_schemas  # noqa: E402

DOMAIN = {
    "schedule": {
        "tile_m": [32, 64, 128], "tile_n": [32, 64, 128], "tile_k": [16, 32, 64],
        "pipeline_depth": [2, 3, 4], "unroll_factor": [2, 4, 8], "vectorize_factor": [2, 4, 8],
    },
    "realization": {
        "program_order": ["TWO_DIMENSIONAL", "GROUPED_M"], "group_m": [1, 4, 8], "num_warps": [4, 8],
        "num_stages": [2, 3, 4], "waves_per_eu": [1, 2], "matrix_instr_nonkdim": [0, 16], "kpack": [1],
    },
}
BASELINE = {
    "schedule": {"tile_m": 64, "tile_n": 64, "tile_k": 32, "pipeline_depth": 3, "unroll_factor": 4, "vectorize_factor": 4},
    "realization": {"program_order": "TWO_DIMENSIONAL", "group_m": 1, "num_warps": 4, "num_stages": 3,
                    "waves_per_eu": 1, "matrix_instr_nonkdim": 0, "kpack": 1},
}
REWARD = {
    "schema_invalid": -1.0, "verify_fail": -0.3, "compile_fail": -0.5, "compile_pass": 0.3,
    "correctness_fail": -0.8, "correctness_pass": 1.0, "performance_log_clip": 1.0, "new_best_bonus": 0.2,
    "repeat_penalty": -0.2, "unknown_outcome": -0.5, "terminal_best_speedup": 0.5,
}
FACTS = {"vgpr_count": 128, "agpr_count": 0, "sgpr_count": 60, "spill_count": 0, "scratch_size_bytes": 0,
         "shared_bytes": 65536, "reported_occupancy_waves": 4, "code_size_bytes": 12000}


class FakeEvaluator:
    """Returns a scripted worker result keyed by tile_m; counts real evaluations."""

    def __init__(self, outcomes: Mapping[int, dict[str, Any]]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def describe(self) -> Mapping[str, Any]:
        return {
            "candidate_profile": "C144", "portfolio_profile": "P", "portfolio": [{"workload_id": "w", "m": 1, "n": 1, "k": 1}],
            "baseline_config": BASELINE, "domain": DOMAIN, "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
        }

    async def evaluate(self, config: Mapping[str, Any], fidelity: str, measurement: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((json.dumps(config, sort_keys=True), fidelity))
        if config == BASELINE:
            return {"stage": "compiled", "compiled_facts": FACTS}
        return copy.deepcopy(self.outcomes[config["schedule"]["tile_m"]])

    async def close(self) -> None:
        self.closed = True


def _measured(speedup: float) -> dict[str, Any]:
    return {
        "stage": "measured", "candidate_digest": "d" * 64, "source_sha256": "s" * 64, "compiled_facts": FACTS,
        "correctness": {"status": "PASS", "cases": [{"workload_id": "w", "status": "PASS"}]},
        "performance": {"validity": "VALID", "geomean_speedup": speedup, "per_workload_speedup": {"w": speedup}},
    }


def _env(outcomes: Mapping[int, dict[str, Any]], max_submissions: int = 3) -> tuple[MicakeGemmEnvironment, FakeEvaluator]:
    fake = FakeEvaluator(outcomes)
    env = build_environment(
        BACKEND_NAME, micake_root="/unused", micake_python="/unused", gpu_id=0, rocm_path="/opt/rocm",
        request_timeout_s=1.0, fidelity="performance", max_submissions=max_submissions,
        measurement={"warmup_count": 1, "paired_repeats": 1, "dispatches_per_block": 1}, reward=REWARD,
        evaluator=fake,
    )
    return env, fake


def _submit(tile_m: int, call_id: str = "c", **overrides: Any) -> ToolCall:
    schedule = dict(BASELINE["schedule"], tile_m=tile_m)
    realization = {k: v for k, v in BASELINE["realization"].items() if k != "kpack"}
    arguments = {"schedule": schedule, "realization": realization, "hypothesis": "test", **overrides}
    return ToolCall.from_raw(call_id, "submit_schedule", json.dumps(arguments))


def test_tool_schema_hides_fixed_fields_and_is_closed() -> None:
    submit = tool_schemas(DOMAIN)[0]["function"]["parameters"]
    assert "kpack" not in submit["properties"]["realization"]["properties"]
    assert submit["properties"]["schedule"]["additionalProperties"] is False
    assert submit["properties"]["schedule"]["properties"]["tile_m"]["enum"] == [32, 64, 128]


def test_parse_submission_rejects_out_of_domain_and_extra_fields() -> None:
    good = json.loads(_submit(128).raw_arguments)
    assert parse_submission(good, DOMAIN).config["realization"]["kpack"] == 1
    with pytest.raises(ToolArgumentError, match="VALUE_OUT_OF_DOMAIN"):
        parse_submission({**good, "schedule": dict(good["schedule"], tile_m=48)}, DOMAIN)
    with pytest.raises(ToolArgumentError, match="UNEXPECTED_FIELD"):
        parse_submission({**good, "source": "kernel"}, DOMAIN)
    with pytest.raises(ToolArgumentError, match="VALUE_OUT_OF_DOMAIN"):
        parse_submission({**good, "schedule": dict(good["schedule"], tile_m="128")}, DOMAIN)
    with pytest.raises(ToolArgumentError, match="ARGUMENTS_NOT_JSON_OBJECT"):
        parse_submission(None, DOMAIN)


def test_reward_mapping_duplicate_and_best_tracking() -> None:
    outcomes = {
        32: {"stage": "compile_fail", "diagnostic": {"code": "C144_TRITON_COMPILE_REJECTED:X", "message": "boom"}},
        128: _measured(1.5),
    }
    env, fake = _env(outcomes, max_submissions=3)

    async def scenario() -> list[Any]:
        reset = await env.reset({"task_id": "t"})
        assert reset.observation["baseline_compiled_facts"] == FACTS
        results = [
            await env.step(ToolCall.from_raw("bad", "submit_schedule", "{oops")),
            await env.step(_submit(32)),
            await env.step(_submit(128)),
            await env.step(_submit(128, call_id="dup")),
        ]
        return results + [await env.finalize("AGENT_FINISHED")]

    invalid, compile_fail, measured, duplicate, (terminal, summary) = asyncio.run(scenario())
    assert invalid.reward == -1.0 and invalid.observation["submission_counted"] is False
    assert compile_fail.reward == -0.5 and compile_fail.observation["diagnostic"]["message"] == "boom"
    assert measured.reward == pytest.approx(1.0 + math.log(1.5) + 0.2)
    assert measured.observation["new_best"] is True
    assert measured.observation["performance"]["geomean_speedup_vs_baseline"] == 1.5
    assert {r.kind for r in measured.receipt_refs} >= {"c144_candidate", "c144_paired_measurement"}
    assert duplicate.reward == -0.2 and duplicate.observation["duplicate"] is True
    assert len([c for c in fake.calls if c[1] == "performance"]) == 2
    assert terminal == pytest.approx(0.5 * math.log(1.5))
    assert summary["best_speedup"] == 1.5 and summary["best_config"]["schedule"]["tile_m"] == 128


def test_resource_rejection_continues_the_episode() -> None:
    env, _fake = _env({
        128: {
            "stage": "resource_rejected",
            "diagnostic": {
                "code": "C144_GROUP_SEGMENT_EXCEEDED",
                "message": "shared_bytes 262144 exceeds gfx950 group segment limit 163840",
            },
            "compiled_facts": {"shared_bytes": 262144},
        },
    }, max_submissions=2)

    async def scenario() -> Any:
        await env.reset({})
        rejected = await env.step(_submit(128))
        return rejected

    rejected = asyncio.run(scenario())
    assert rejected.done is False and rejected.terminal_state == "RUNNING"
    assert rejected.reward == -0.5
    assert rejected.observation["diagnostic"]["code"] == "C144_GROUP_SEGMENT_EXCEEDED"
    assert env.remaining_submissions() == 1


def test_unknown_outcome_terminates_without_caching() -> None:
    env, fake = _env({32: {"stage": "evaluation_unknown", "diagnostic": {"code": "TimeoutError"}}})

    async def scenario() -> Any:
        await env.reset({})
        return await env.step(_submit(32))

    result = asyncio.run(scenario())
    assert result.done and result.terminal_state == "UNKNOWN_OUTCOME" and result.reward == -0.5
    assert all(key[1] != "performance" for key in env._result_cache)


def test_budget_and_cross_episode_cache() -> None:
    env, fake = _env({128: _measured(0.8)}, max_submissions=1)

    async def scenario() -> tuple[Any, Any]:
        await env.reset({})
        first = await env.step(_submit(128))
        await env.reset({})
        second = await env.step(_submit(128))
        return first, second

    first, second = asyncio.run(scenario())
    assert first.done and first.terminal_state == "BUDGET_EXHAUSTED"
    assert first.observation["new_best"] is False
    assert first.reward == pytest.approx(1.0 + math.log(0.8))
    assert second.metadata["cached"] is True
    assert len([c for c in fake.calls if c[1] == "performance"]) == 1


def test_reward_config_must_be_complete() -> None:
    with pytest.raises(ValueError, match="reward keys mismatch"):
        _env_kwargs = dict(REWARD)
        _env_kwargs.pop("repeat_penalty")
        MicakeGemmEnvironment("/u", "/u", 0, "/opt/rocm", 1.0, "compile", 1, {}, _env_kwargs, evaluator=FakeEvaluator({}))
