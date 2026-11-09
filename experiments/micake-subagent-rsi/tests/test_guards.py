"""GEAK-style measurement guards: re-measurement, arbitration, noise band, champion re-verification."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from micake_rsi.generation import PromotionGate, promotion_decision  # noqa: E402
from micake_rsi.guards import harness_fingerprint  # noqa: E402
from micake_rsi.loop import run_generation  # noqa: E402
from sandbox.micake import MicakeGemmEnvironment  # noqa: E402
from test_rsi_loop import FakeEvaluator, _cfg, _measured, _micake, _policies  # noqa: E402


class InflatedEvaluator(FakeEvaluator):
    """In-episode 128x128 measures 1.5; on a fresh worker it reproduces only at 1.2."""

    def __init__(self) -> None:
        self.fresh = False

    async def evaluate(self, config: Mapping[str, Any], fidelity: str, measurement: Mapping[str, Any]) -> dict[str, Any]:
        if self.fresh and (config["schedule"]["tile_m"], config["schedule"]["tile_n"]) == (128, 128):
            return copy.deepcopy(_measured(1.2))
        return await super().evaluate(config, fidelity, measurement)

    async def close(self) -> None:
        self.fresh = True


def test_unreproduced_claim_is_flagged_and_kept_out_of_sft(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    micake = MicakeGemmEnvironment(**dict(cfg["micake"]["kwargs"], evaluator=InflatedEvaluator()))
    manifest = asyncio.run(run_generation(cfg, micake=micake, policies=_policies()))

    guards = manifest["guards"]
    assert [r["status"] for r in guards["records"]] == ["flagged", "flagged"]
    assert guards["records"][0]["runs"] == [1.2, 1.2]
    assert guards["summary"]["flagged_rate"] == pytest.approx(1.0)
    manager = manifest["metrics"]["manager"]
    assert manager["mean_claimed_best_speedup"] == pytest.approx(1.5)
    assert manager["mean_best_speedup"] == pytest.approx(1.2)
    assert any("flagged_rate" in reason for reason in manifest["promotion"]["reasons"])
    assert all(d["sft_count"] == 0 for d in manifest["datasets"].values())
    rows = [json.loads(line) for line in Path(manifest["datasets"]["critic"]["rl"]).read_text().splitlines()]
    assert {row["manager_verification"] for row in rows} == {"flagged"}
    assert (tmp_path / "rsi" / "gen0" / "verification.json").exists()


def test_champion_is_remeasured_when_the_harness_moved(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    gen0 = asyncio.run(run_generation(cfg, micake=_micake(cfg), policies=_policies()))
    assert gen0["guards"]["harness"] == harness_fingerprint(cfg["guards"]["harness_roots"])
    assert [r["status"] for r in gen0["guards"]["records"]] == ["verified", "verified"]
    assert "champion_reverified" not in gen0["guards"]["summary"]

    champion_path = tmp_path / "rsi" / "champion.json"
    champion = json.loads(champion_path.read_text())
    champion["guards"]["harness"]["sha256"] = "harness-before-repair"
    champion_path.write_text(json.dumps(champion))

    cfg["rsi"]["generation_id"] = "gen1"
    micake = MicakeGemmEnvironment(**dict(cfg["micake"]["kwargs"], evaluator=InflatedEvaluator()))
    gen1 = asyncio.run(run_generation(cfg, micake=micake, policies=_policies()))
    summary = gen1["guards"]["summary"]
    assert summary["champion_mean_best_speedup"] == pytest.approx(1.2)
    assert gen1["promotion"]["champion_reference"] == pytest.approx(1.2)


def test_noise_band_raises_the_promotion_threshold() -> None:
    gate = PromotionGate(min_role_episodes=1, max_schema_invalid_rate=0.3, max_correctness_regression=0.05,
                         min_speedup_gain=0.01, max_flagged_rate=0.5)
    role = {"episodes": 4, "schema_invalid_rate": 0.1, "correctness_rate": 0.8}
    champion = {"manager": {"mean_best_speedup": 1.10}, "roles": {"planner": role}}
    challenger = {"manager": {"mean_best_speedup": 1.15}, "roles": {"planner": role}}
    assert promotion_decision(challenger, champion, gate, {"noise_band": 0.02})["decision"] == "PROMOTE"
    noisy = promotion_decision(challenger, champion, gate, {"noise_band": 0.08})
    assert noisy["decision"] == "HOLD" and noisy["gain_threshold"] == pytest.approx(0.08)
    remeasured = promotion_decision(challenger, champion, gate, {"champion_mean_best_speedup": 1.20})
    assert remeasured["decision"] == "HOLD" and remeasured["champion_reference"] == pytest.approx(1.20)


def test_new_best_must_clear_the_noise_band(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    micake = MicakeGemmEnvironment(**dict(cfg["micake"]["kwargs"], evaluator=FakeEvaluator(), new_best_min_gain=0.3))
    asyncio.run(run_generation(cfg, micake=micake, policies=_policies()))
    episodes = [json.loads(line) for line in (tmp_path / "rsi" / "gen0" / "episodes.jsonl").read_text().splitlines()]
    best = {e["environment"]["final_summary"]["best_speedup"] for e in episodes if e["role"] == "manager"}
    assert best == {1.5}
