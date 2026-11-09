from __future__ import annotations

import asyncio
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from lumenrl.agentic.core import ToolCall  # noqa: E402
from lumenrl.agentic.rollout import ScriptedPolicy  # noqa: E402
from micake_rsi.generation import PromotionGate, promotion_decision  # noqa: E402
from micake_rsi.loop import run_generation  # noqa: E402
from micake_rsi.subagent_envs import CriticEnvironment, CriticRewards, PlannerEnvironment  # noqa: E402
from sandbox.micake import MicakeGemmEnvironment  # noqa: E402

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
FACTS = {"vgpr_count": 128, "shared_bytes": 65536, "reported_occupancy_waves": 4}
CONFIG_PATH = EXPERIMENT_DIR / "configs" / "gen0_planner_critic.yaml"


def _measured(speedup: float) -> dict[str, Any]:
    return {
        "stage": "measured", "compiled_facts": FACTS,
        "correctness": {"status": "PASS", "cases": [{"workload_id": "w", "status": "PASS"}]},
        "performance": {"validity": "VALID", "geomean_speedup": speedup, "per_workload_speedup": {"w": speedup}},
    }


OUTCOMES = {
    (128, 64): _measured(1.2),
    (32, 64): {"stage": "compile_fail", "diagnostic": {"code": "C144_TRITON_COMPILE_REJECTED:X", "message": "boom"}},
    (128, 128): _measured(1.5),
}


class FakeEvaluator:
    """Scripted worker results keyed by (tile_m, tile_n)."""

    async def describe(self) -> Mapping[str, Any]:
        return {
            "candidate_profile": "C144", "portfolio_profile": "P", "portfolio": [{"workload_id": "w", "m": 1, "n": 1, "k": 1}],
            "baseline_config": BASELINE, "domain": DOMAIN, "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
        }

    async def evaluate(self, config: Mapping[str, Any], fidelity: str, measurement: Mapping[str, Any]) -> dict[str, Any]:
        if config == BASELINE:
            return {"stage": "compiled", "compiled_facts": FACTS}
        return copy.deepcopy(OUTCOMES[(config["schedule"]["tile_m"], config["schedule"]["tile_n"])])

    async def close(self) -> None:
        pass


def _cfg(tmp_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["rsi"]["root"] = str(tmp_path / "rsi")
    cfg["specialists"]["planner"]["max_submissions"] = 2
    cfg["debug"]["enabled"] = False
    return cfg


def _micake(cfg: Mapping[str, Any]) -> MicakeGemmEnvironment:
    kwargs = dict(cfg["micake"]["kwargs"], evaluator=FakeEvaluator())
    return MicakeGemmEnvironment(**kwargs)


def _schedule(tile_m: int, tile_n: int = 64) -> dict[str, Any]:
    return dict(BASELINE["schedule"], tile_m=tile_m, tile_n=tile_n)


REALIZATION = {k: v for k, v in BASELINE["realization"].items() if k != "kpack"}


def _policies() -> dict[str, ScriptedPolicy]:
    manager = ScriptedPolicy([
        {"name": "delegate", "arguments": {"role": "planner", "instruction": "try larger M tiles"}},
        {"name": "delegate", "arguments": {"role": "critic", "instruction": "what limits the best config?"}},
        {"name": "finish", "arguments": {"summary": "done"}},
    ], model="manager")
    planner = ScriptedPolicy([
        {"name": "submit_schedule", "arguments": {"schedule": _schedule(128), "realization": REALIZATION, "hypothesis": "bigger M"}},
        {"name": "submit_schedule", "arguments": {"schedule": _schedule(32), "realization": REALIZATION, "hypothesis": "smaller M"}},
        {"name": "submit_schedule", "arguments": {"schedule": _schedule(128, 128), "realization": REALIZATION, "hypothesis": "over budget"}},
        {"name": "report", "arguments": {"summary": "128 helps, 32 fails to compile"}},
    ], model="planner")
    critic = ScriptedPolicy([
        {"name": "report_diagnosis", "arguments": {"bottleneck": "nope", "evidence": "x", "recommended_schedule": _schedule(128, 128),
                                                   "recommended_realization": REALIZATION, "expected_effect": "y"}},
        {"name": "report_diagnosis", "arguments": {"bottleneck": "underfilled_grid", "evidence": "wide workload gains from N",
                                                   "recommended_schedule": _schedule(128, 128),
                                                   "recommended_realization": REALIZATION, "expected_effect": "faster wide"}},
    ], model="critic")
    return {"manager": manager, "planner": planner, "critic": critic}


def test_generation_end_to_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    manifest = asyncio.run(run_generation(cfg, micake=_micake(cfg), policies=_policies()))

    assert manifest["promotion"]["decision"] == "BOOTSTRAP"
    assert manifest["episodes"] == {"path": str(tmp_path / "rsi" / "gen0" / "episodes.jsonl"), "manager": 2, "specialist": 4}
    m = manifest["metrics"]
    assert m["manager"]["mean_best_speedup"] == pytest.approx(1.5)
    assert m["manager"]["delegations"] == {"planner": 2, "critic": 2}
    planner, critic = m["roles"]["planner"], m["roles"]["critic"]
    assert planner["submissions"] == 4 and planner["correctness_rate"] == pytest.approx(0.5)
    assert planner["schema_invalid_rate"] == pytest.approx(1 / 4)
    assert critic["submissions"] == 2 and critic["new_best_rate"] == pytest.approx(1.0)

    rows = [json.loads(line) for line in Path(manifest["datasets"]["critic"]["rl"]).read_text().splitlines()]
    assert len(rows) == 2
    row = rows[0]
    assert row["binding"]["policy_version"] == "critic-g0" and row["requires_logprob_recompute"] is True
    assert row["parent_episode_id"] and row["tools"][0]["function"]["name"] == "report_diagnosis"
    assert [s["execution_result"] for s in row["steps"]] == ["schema_invalid", "diagnosis_probed"]
    for step in row["steps"]:
        assert row["messages"][step["assistant_message_index"]]["role"] == "assistant"
    contribution = math.log(1.5 / 1.2)
    assert row["reward"]["global"] == pytest.approx(math.log(1.5))
    assert row["reward"]["contribution"] == pytest.approx(contribution)
    probe_reward = 0.2 + math.log(1.5) + 0.3
    assert row["reward"]["local"] == pytest.approx(-0.3 + 0.1 + probe_reward)
    assert row["reward"]["total"] == pytest.approx(row["reward"]["local"] + 0.5 * math.log(1.5) + contribution)

    champion = json.loads((tmp_path / "rsi" / "champion.json").read_text())
    assert champion["generation_id"] == "gen0"
    with pytest.raises(FileExistsError):
        asyncio.run(run_generation(cfg, micake=_micake(cfg), policies=_policies()))


def test_planner_sub_budget_and_shared_state() -> None:
    cfg = _cfg(Path("/unused"))
    micake = _micake(cfg)
    planner = PlannerEnvironment(micake, history_limit=5, report_reward=0.0)

    def call(tile_m: int, tile_n: int = 64) -> ToolCall:
        args = {"schedule": _schedule(tile_m, tile_n), "realization": REALIZATION, "hypothesis": "h"}
        return ToolCall.from_raw("c", "submit_schedule", json.dumps(args))

    async def scenario() -> list[Any]:
        await micake.reset({"task_id": "t"})
        reset = await planner.reset({"instruction": "go", "max_submissions": 1})
        assert reset.observation["sub_budget"] == 1
        first = await planner.step(call(128))
        second = await planner.step(call(32))
        return [first, second]

    first, second = asyncio.run(scenario())
    assert first.observation["sub_budget_remaining"] == 0
    assert second.observation["error"] == "SUB_BUDGET_EXHAUSTED"
    assert micake.remaining_submissions() == 9 and len(micake.history()) == 1


def test_critic_without_probe_only_scores_validity() -> None:
    cfg = _cfg(Path("/unused"))
    micake = _micake(cfg)
    critic = CriticEnvironment(micake, history_limit=5, rewards=CriticRewards(0.1, -0.3, probe=False))
    args = {"bottleneck": "lds_capacity", "evidence": "64 KiB LDS", "recommended_schedule": _schedule(128),
            "recommended_realization": REALIZATION, "expected_effect": "e"}

    async def scenario() -> Any:
        await micake.reset({})
        await critic.reset({"instruction": "diagnose"})
        return await critic.step(ToolCall.from_raw("c", "report_diagnosis", json.dumps(args)))

    result = asyncio.run(scenario())
    assert result.done and result.reward == pytest.approx(0.1)
    assert micake.remaining_submissions() == 10
    assert critic.report["recommended_config"]["realization"]["kpack"] == 1


def test_promotion_gate() -> None:
    gate = PromotionGate(min_role_episodes=1, max_schema_invalid_rate=0.3, max_correctness_regression=0.05, min_speedup_gain=0.01)
    role = {"episodes": 4, "schema_invalid_rate": 0.1, "correctness_rate": 0.8}
    champion = {"manager": {"mean_best_speedup": 1.10}, "roles": {"planner": role}}
    better = {"manager": {"mean_best_speedup": 1.15}, "roles": {"planner": role}}
    regressed = {"manager": {"mean_best_speedup": 1.20}, "roles": {"planner": dict(role, correctness_rate=0.6)}}
    assert promotion_decision(better, champion, gate)["decision"] == "PROMOTE"
    hold = promotion_decision(regressed, champion, gate)
    assert hold["decision"] == "HOLD" and "correctness" in hold["reasons"][0]
    assert promotion_decision(better, None, gate)["decision"] == "BOOTSTRAP"
