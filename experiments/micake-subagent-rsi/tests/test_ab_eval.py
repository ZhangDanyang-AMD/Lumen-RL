"""A/B benchmark metrics, statistics and the rule-based search harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from micake_rsi.ab_eval import compare, critic_metrics, e2e_metrics, planner_metrics, step_class  # noqa: E402
from micake_rsi.search_harness import GreedySearch, fits_lds, submission_arguments  # noqa: E402

DOMAIN = {
    "schedule": {"tile_m": [32, 64, 128], "tile_n": [32, 64, 128], "tile_k": [16, 32, 64], "pipeline_depth": [2, 3, 4],
                 "unroll_factor": [4], "vectorize_factor": [4]},
    "realization": {"num_stages": [2, 3, 4], "num_warps": [4, 8], "waves_per_eu": [1], "matrix_instr_nonkdim": [0, 16],
                    "program_order": ["TWO_DIMENSIONAL"], "group_m": [1], "kpack": [1]},
}
BASELINE = {
    "schedule": {"tile_m": 64, "tile_n": 64, "tile_k": 32, "pipeline_depth": 3, "unroll_factor": 4, "vectorize_factor": 4},
    "realization": {"num_stages": 3, "num_warps": 4, "waves_per_eu": 1, "matrix_instr_nonkdim": 0,
                    "program_order": "TWO_DIMENSIONAL", "group_m": 1, "kpack": 1},
}


def _step(label, speedup=None, validity="VALID", probe=None):
    observation = {}
    if speedup is not None:
        observation["performance"] = {"geomean_speedup_vs_baseline": speedup, "validity": validity}
    if probe is not None:
        observation["probe"] = probe
    return {"result": {"execution_result": label, "observation": observation}}


def test_planner_metrics_count_every_non_measured_action_as_wasted():
    episode = {"steps": [_step("measured", 1.04), _step("resource_rejected"), _step("duplicate"), _step("report")],
               "terminal_state": "AGENT_FINISHED", "turns": [{}] * 4}
    m = planner_metrics(episode)
    assert (m["actions"], m["wasted"], m["resource_rejected"], m["duplicate"]) == (3, 2, 1, 1)
    assert m["best_speedup"] == pytest.approx(1.04) and m["clean_finish"] == 1.0


def test_invalid_measurement_is_not_productive():
    assert step_class(_step("measured", 1.2, validity="NOISY")) == "invalid_measurement"


def test_critic_metrics_read_the_probe():
    probe = {"stage": "measured", "performance": {"geomean_speedup_vs_baseline": 1.06, "validity": "VALID"}}
    m = critic_metrics({"steps": [_step("schema_invalid"), _step("diagnosis_probed", probe=probe)]}, context_best=1.045)
    assert m["first_try_valid"] == 0.0 and m["probe_productive"] == 1.0 and m["probe_beats_context_best"] == 1.0


def test_e2e_metrics_reach_and_auc_skip_uncounted_calls():
    trace = [
        {"t": 10, "index": 1, "stage": "resource_rejected", "speedup": None, "counted": True},
        {"t": 12, "index": 1, "stage": "duplicate", "speedup": None, "counted": False},
        {"t": 30, "index": 2, "stage": "measured", "speedup": 1.04, "counted": True},
    ]
    m = e2e_metrics(trace, budget=4)
    assert (m["actions"], m["submissions"], m["wasted"]) == (3, 2, 2)
    assert m["reach"]["1.03"] == {"submissions": 2.0, "seconds": 30.0}
    assert m["reach"]["1.05"] == {"submissions": None, "seconds": None}
    assert m["final_best"] == pytest.approx(1.04)


def test_compare_detects_a_clear_shift_and_not_a_null_one():
    shifted = compare([0.0, 0.1, 0.0, 0.1, 0.0, 0.1], [1.0, 0.9, 1.0, 0.9, 1.0, 0.9])
    assert shifted["ci_low"] > 0 and shifted["p"] < 0.05
    null = compare([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0])
    assert null["ci_low"] <= 0 <= null["ci_high"] and null["p"] > 0.5


def test_greedy_search_respects_lds_and_moves_to_improvements():
    search = GreedySearch(BASELINE, DOMAIN)
    first = search.propose()
    assert first["realization"]["num_warps"] == 8 and fits_lds(first)
    search.observe(first, 1.03)
    assert search.incumbent == first and search.best == 1.03
    seen = [search.propose() for _ in range(10)]
    assert all(c is None or fits_lds(c) for c in seen)
    assert not any(c and c["schedule"]["tile_m"] == 128 and c["schedule"]["tile_k"] == 64 for c in seen)


def test_stage_move_keeps_pipeline_depth_linked_and_hides_kpack():
    search = GreedySearch(BASELINE, DOMAIN)
    moves = []
    while (c := search.propose()) is not None:
        moves.append(c)
    stage_moves = [c for c in moves if c["realization"]["num_stages"] != 3]
    assert stage_moves and all(c["schedule"]["pipeline_depth"] == c["realization"]["num_stages"] for c in stage_moves)
    assert "kpack" not in submission_arguments(BASELINE, "h")["realization"]


def test_stratified_compare_removes_between_instruction_variance():
    from micake_rsi.ab_eval import stratified_compare

    a = {0: [0.0, 0.1, 0.0], 1: [2.0, 2.1, 2.0], 2: [5.0, 5.1, 5.0]}
    b = {k: [v - 0.5 for v in vs] for k, vs in a.items()}
    result = stratified_compare(a, b, resamples=2000)
    assert result["strata"] == 3 and result["diff"] == pytest.approx(-0.5)
    assert result["ci_high"] < 0 and result["p"] < 0.05
