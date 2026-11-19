from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR, EXPERIMENT_DIR / "scripts"):
    sys.path.insert(0, str(entry))

from lumenrl.agentic.core import ToolCall  # noqa: E402
from lumenrl.agentic.core.backends import build_environment  # noqa: E402
from micake_rsi.subagent_envs import PlannerEnvironment  # noqa: E402
from sandbox.mmkca import MmkcaGemmEnvironment  # noqa: E402
from sandbox.mmkca.environment import PLANNER_PROMPT  # noqa: E402

DOMAIN = {
    "schedule": {"tile_m": [16, 32, 64, 128, 256], "tile_n": [64, 128, 256], "tile_k": [32, 64, 128, 256],
                 "stages": [1, 2], "split_k": [1, 2, 4, 8], "group_m": [1, 4, 8], "lds_pad": [0, 16, 32]},
    "realization": {"mfma": ["32x32x16", "16x16x32", "32x32x64", "16x16x128"], "waves_m": [1, 2, 4], "waves_n": [1, 2, 4]},
}
BASELINE = {"schedule": {"tile_m": 128, "tile_n": 128, "tile_k": 64, "stages": 2, "split_k": 1, "group_m": 8, "lds_pad": 16},
            "realization": {"mfma": "32x32x16", "waves_m": 2, "waves_n": 2}}
OPENING = {"schedule": {**BASELINE["schedule"], "tile_m": 32, "split_k": 2, "group_m": 4},
           "realization": {"mfma": "32x32x16", "waves_m": 1, "waves_n": 2}}
CONFIG = yaml.safe_load((EXPERIMENT_DIR / "configs" / "mmkca_planner.yaml").read_text(encoding="utf-8"))


class FakeWorker:
    """Speedup depends on the regime and bundle, so a cache that ignores them would leak results."""

    def __init__(self) -> None:
        self.describes: list[tuple] = []
        self.evaluations: list[tuple] = []

    async def describe(self, dtype: str, portfolio: str, split: str, bundle: int | None) -> Mapping[str, Any]:
        self.describes.append((dtype, portfolio, split, bundle))
        return {"candidate_profile": "mmkca-gemm-schedule", "portfolio_profile": f"{split}-{portfolio}",
                "regime": portfolio, "portfolio": [{"workload_id": f"{split}.{portfolio}.{bundle}", "m": 4, "n": 256, "k": 512}],
                "baseline_config": BASELINE, "opening": OPENING if portfolio == "decode" else BASELINE, "domain": DOMAIN,
                "objective": f"{portfolio} only", "target": {"chipset": "gfx950"},
                "resource_limits": {"max_group_segment_bytes": 163840, "element_bytes": 1 if dtype == "fp8" else 2}}

    async def evaluate(self, config, fidelity, measurement, dtype, portfolio, split, bundle) -> dict[str, Any]:
        self.evaluations.append((fidelity, dtype, portfolio, split, bundle))
        if fidelity == "compile":
            return {"stage": "compiled", "compiled_facts": {"vgpr_count": 100}}
        speedup = 1.5 + (0.1 if portfolio == "prefill" else 0.0) + 0.01 * (bundle or 0)
        return {"stage": "measured", "compiled_facts": {"vgpr_count": 100},
                "correctness": {"status": "PASS", "cases": []},
                "performance": {"validity": "VALID", "geomean_speedup": speedup, "per_workload_speedup": {}}}

    async def close(self) -> None:
        return None


def make_env(worker: FakeWorker) -> MmkcaGemmEnvironment:
    kwargs = {k: v for k, v in CONFIG["micake"]["kwargs"].items() if k != "gpu_id"}
    return MmkcaGemmEnvironment(gpu_id=0, evaluator=worker, **kwargs)


def submit(config: Mapping[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(call_id=call_id, name="submit_schedule",
                    arguments={"schedule": config["schedule"], "realization": config["realization"], "hypothesis": "try"},
                    raw_arguments="")


def test_backend_registers_under_its_own_name() -> None:
    env = build_environment("mmkca_gemm", "sandbox.mmkca", **{**CONFIG["micake"]["kwargs"], "evaluator": FakeWorker()})
    assert isinstance(env, MmkcaGemmEnvironment)
    assert env.capabilities().backend == "mmkca_gemm"


def test_reset_scores_one_regime_and_bundle() -> None:
    worker = FakeWorker()
    env = make_env(worker)

    async def run() -> None:
        reset = await env.reset({"portfolio": "decode", "dtype": "fp8", "split": "train", "bundle": 3, "max_submissions": 2})
        assert reset.observation["regime"] == "decode"
        assert reset.observation["opening"] == OPENING
        assert reset.observation["resource_limits"]["lds_element_bytes"] == 1
        assert reset.metadata["bundle"] == 3
        first = await env.step(submit(OPENING))
        assert first.execution_result == "measured" and first.observation["new_best"]
        assert first.metadata["config"]["realization"] == OPENING["realization"]
        duplicate = await env.step(submit(OPENING, "c2"))
        assert duplicate.execution_result == "duplicate"
        await env.reset({"portfolio": "prefill", "dtype": "bf16", "split": "train", "bundle": 3})
        again = await env.step(submit(OPENING))
        assert again.observation["performance"]["geomean_speedup_vs_baseline"] == pytest.approx(1.63)
        _, info = await env.finalize("BUDGET_EXHAUSTED")
        assert info["portfolio"] == "prefill" and info["dtype"] == "bf16"

    asyncio.run(run())
    measured = [e for e in worker.evaluations if e[0] == "performance"]
    assert measured == [("performance", "fp8", "decode", "train", 3), ("performance", "bf16", "prefill", "train", 3)]
    assert worker.describes == [("fp8", "decode", "train", 3), ("bf16", "prefill", "train", 3)]


def test_heldout_takes_no_bundle() -> None:
    env = make_env(FakeWorker())
    with pytest.raises(ValueError):
        asyncio.run(env.reset({"portfolio": "decode", "dtype": "bf16", "split": "heldout", "bundle": 1}))
    with pytest.raises(ValueError):
        asyncio.run(env.reset({"portfolio": "decode", "dtype": "bf16", "split": "train"}))


def _job(arm: str, sample: int, split: str = "train") -> dict[str, Any]:
    bundle = None if split == "heldout" else 1000
    group = "heldout:decode:bf16" if split == "heldout" else "train:decode:bf16:1000"
    return {"key": f"{group}:{arm}:{sample}", "arm": arm, "portfolio": "decode", "dtype": "bf16", "split": split,
            "bundle": bundle, "group": group, "sample": sample}


def test_driver_episode_becomes_an_sft_sample_and_a_gate_record() -> None:
    from lumenrl.agentic.rollout import ScriptedPolicy
    from micake_rsi.planner_gate import verified_best
    from rsi_mmkca import episode_sample, kept_samples, run_job

    tweaked = {"schedule": {**OPENING["schedule"], "split_k": 4}, "realization": OPENING["realization"]}
    actions = [{"name": "submit_schedule", "arguments": {**OPENING, "hypothesis": "opening"}},
               {"name": "submit_schedule", "arguments": {**OPENING, "hypothesis": "again"}},
               {"name": "submit_schedule", "arguments": {**tweaked, "hypothesis": "split"}},
               {"name": "report", "arguments": {"summary": "done"}}]
    cfg = {**CONFIG, "rsi": {**CONFIG["rsi"], "max_submissions": 3}}
    env = make_env(FakeWorker())
    record = asyncio.run(run_job(env, ScriptedPolicy(actions), cfg, _job("rl0", 0, "heldout"), verify=True))
    assert record["metrics"]["final_best"] == pytest.approx(1.5)
    assert record["metrics"]["reach"]["1.04"]["submissions"] == 1.0
    assert [row["stage"] for row in record["trace"]] == ["measured", "duplicate", "measured", "report"]
    assert verified_best(record) == pytest.approx(1.5)

    sample = episode_sample(record)
    advantages = [(s["tool"], s["execution_result"], s["advantage"]) for s in sample["steps"]]
    assert advantages == [("submit_schedule", "measured", 1.0), ("submit_schedule", "duplicate", -1.0),
                          ("submit_schedule", "measured", -1.0), ("report", "report", 1.0)]
    for step in sample["steps"]:
        assert sample["messages"][step["assistant_message_index"]]["role"] == "assistant"

    low = {**record, "key": "low", "metrics": {**record["metrics"], "final_best": 1.2}}
    high = {**record, "key": "high", "metrics": {**record["metrics"], "final_best": 1.6}}
    flat = {**record, "key": "flat", "metrics": {**record["metrics"], "final_best": 1.0}}
    kept = {s["source_key"] for s in kept_samples([low, record, high, flat], 0.005)}
    assert kept == {"high", record["key"]}


def test_illegal_submit_trains_the_repaired_legal_schedule() -> None:
    sys.path.insert(0, "/home/danyzhan/MICake-MMKCA/scripts")
    from gemm_portfolios import legal_mfma, schedule_violation, workloads
    from rsi_mmkca import legality_samples

    rows = workloads("decode", "heldout", "fp8")
    illegal = {"schedule": dict(OPENING["schedule"]), "realization": {"mfma": "32x32x16", "waves_m": 4, "waves_n": 4}}
    record = {"key": "train:decode:fp8:1:rl1:0", "dtype": "fp8", "episode": {
        "episode_id": "e",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": json.dumps({"portfolio": rows})},
            {"role": "assistant", "content": "", "tool_calls": []},
        ],
        "environment": {"tools": [{"type": "function", "function": {"name": "submit_schedule", "parameters": {
            "properties": {"realization": {"properties": {"mfma": {"enum": ["16x16x32", "32x32x64"]}}}}}}}]} ,
        "steps": [{"step_index": 0, "turn_index": 0, "action": {"name": "submit_schedule", "arguments": illegal},
                   "result": {"execution_result": "verify_fail"}}],
    }}
    sample = legality_samples(record)[0]
    arguments = json.loads(sample["messages"][-1]["tool_calls"][0]["function"]["arguments"])
    repaired = {"schedule": arguments["schedule"], "realization": arguments["realization"]}
    assert schedule_violation(repaired, "fp8", rows) is None
    assert sample["steps"][0]["advantage"] == 1.0
    assert sample["steps"][0]["assistant_message_index"] == 2
    enum = sample["tools"][0]["function"]["parameters"]["properties"]["realization"]["properties"]["mfma"]["enum"]
    assert enum == legal_mfma("fp8")
    assert "16x16x32" not in enum


def test_planner_child_uses_the_mmkca_prompt() -> None:
    env = make_env(FakeWorker())

    async def run():
        await env.reset({"portfolio": "decode", "dtype": "bf16", "split": "heldout", "max_submissions": 4})
        planner = PlannerEnvironment(env, 12, 0.0)
        return await planner.reset({"instruction": "go", "max_submissions": 4})

    reset = asyncio.run(run())
    assert reset.system_prompt == PLANNER_PROMPT
    assert reset.observation["regime"] == "decode" and reset.observation["opening"] == OPENING
    properties = reset.tools[0]["function"]["parameters"]["properties"]["realization"]["properties"]
    assert "kpack" not in properties and set(properties) == {"mfma", "waves_m", "waves_n"}
