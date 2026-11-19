"""Improvement targets and the Planner champion/challenger gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from micake_rsi.improve import (  # noqa: E402
    config_key, exploit_candidates, exploit_sample, improvement_sample, install_opening, opening_candidate, walk_steps,
)
from micake_rsi.planner_gate import PlannerGate, gate_decision  # noqa: E402
from micake_rsi.search_harness import fits_lds  # noqa: E402

DOMAIN = {
    "schedule": {"tile_m": [32, 64, 128], "tile_n": [32, 64, 128], "tile_k": [16, 32, 64], "pipeline_depth": [2, 3],
                 "unroll_factor": [2, 4], "vectorize_factor": [4]},
    "realization": {"num_stages": [2, 3], "num_warps": [4, 8], "kpack": [1]},
}
BASE = {"schedule": {"tile_m": 64, "tile_n": 64, "tile_k": 32, "pipeline_depth": 2, "unroll_factor": 2, "vectorize_factor": 4},
        "realization": {"num_stages": 2, "num_warps": 4, "kpack": 1}}


def _with(**schedule):
    return {"schedule": {**BASE["schedule"], **schedule}, "realization": dict(BASE["realization"])}


def _call(config):
    return {"role": "assistant", "content": "LDS check: wrong.", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "submit_schedule", "arguments": json.dumps(config)}}]}


def _child():
    first, second = _with(tile_m=128), _with(tile_n=128)
    packet = {"best": {"geomean_speedup_vs_baseline": 1.0, "config": BASE}, "baseline_config": BASE, "history": []}
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": json.dumps(packet)},
                _call(first), {"role": "tool", "content": "{}"}, _call(second), {"role": "tool", "content": "{}"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "r", "type": "function",
                                                                       "function": {"name": "report", "arguments": "{}"}}]}]
    steps = [
        {"step_index": 0, "turn_index": 0, "action": {"name": "submit_schedule", "arguments": first},
         "result": {"execution_result": "measured", "metadata": {"config": first},
                    "observation": {"new_best": True, "best": {"geomean_speedup_vs_baseline": 1.03, "config": first}}}},
        {"step_index": 1, "turn_index": 1, "action": {"name": "submit_schedule", "arguments": second},
         "result": {"execution_result": "measured", "metadata": {"config": second},
                    "observation": {"new_best": False, "best": {"geomean_speedup_vs_baseline": 1.03, "config": first}}}},
        {"step_index": 2, "turn_index": 2, "action": {"name": "report", "arguments": {}}, "result": {"execution_result": "report"}},
    ]
    return {"role": "planner", "episode_id": "e", "messages": messages, "steps": steps, "environment": {"tools": []}}


def test_walk_steps_tracks_session_best_and_submitted():
    entries = list(walk_steps(_child()))
    assert [e["best_speedup"] for e in entries] == [1.0, 1.03]
    assert entries[1]["best_config"] == _with(tile_m=128)
    assert config_key(_with(tile_m=128)) in entries[1]["submitted"] and config_key(BASE) in entries[0]["submitted"]


def test_improvement_sample_keeps_new_best_and_report_with_exact_lds():
    sample = improvement_sample(_child(), "k")
    assert [s["advantage"] for s in sample["steps"]] == [1.0, -1.0, 1.0]
    content = sample["messages"][2]["content"]
    assert content.startswith("LDS check") and "wrong" not in content and "fits" in content


def test_exploit_candidates_are_untested_fitting_neighbours():
    entry = list(walk_steps(_child()))[1]
    candidates = list(exploit_candidates(entry["best_config"], DOMAIN, entry["submitted"]))
    assert candidates and all(fits_lds(c) and config_key(c) not in entry["submitted"] for c in candidates)
    sample = exploit_sample(_child(), entry, candidates[0], "k")
    target = sample["messages"][-1]
    assert sample["steps"][0]["assistant_message_index"] == len(sample["messages"]) - 1 == 4
    assert json.loads(target["tool_calls"][0]["function"]["arguments"])["schedule"] == candidates[0]["schedule"]
    assert target["content"].count("LDS check") == 1


def test_opening_candidate_lifts_both_axes_and_rejects_an_illegal_tile():
    candidate = opening_candidate(BASE, element=2, domain=DOMAIN)
    assert candidate["schedule"]["tile_m"] == candidate["schedule"]["tile_n"] == 128
    assert candidate["schedule"]["tile_k"] == 32 and fits_lds(candidate)
    assert opening_candidate(candidate, domain=DOMAIN) is None
    wide = _with(tile_k=64, pipeline_depth=3, unroll_factor=4)
    # (128+128)*64*2*(3-1)*4 = 262144 B, over the 163840 B cap.
    assert opening_candidate(wide) is None


def test_install_opening_replaces_only_a_small_first_submit():
    child = _child()
    child["messages"][2] = _call(BASE)
    child["steps"][0]["action"]["arguments"] = BASE
    child["steps"][0]["result"]["metadata"]["config"] = BASE
    child["steps"][0]["result"]["observation"] = {"new_best": True, "best": {"geomean_speedup_vs_baseline": 1.02, "config": BASE}}
    candidate = opening_candidate(BASE, domain=DOMAIN)
    sample = install_opening(child, candidate, "k")
    submitted = json.loads(sample["messages"][2]["tool_calls"][0]["function"]["arguments"])
    assert submitted["schedule"]["tile_m"] == submitted["schedule"]["tile_n"] == 128
    assert sample["steps"][0]["execution_result"] == "opening_tile" and sample["steps"][0]["advantage"] == 1.0
    assert sample["steps"][1]["advantage"] == -1.0
    already = _child()
    assert install_opening(already, candidate, "k") is None


def test_fp8_packet_halves_lds_in_candidates_and_annotations():
    child = _child()
    packet = json.loads(child["messages"][1]["content"])
    child["messages"][1]["content"] = json.dumps({**packet, "input_dtype": "FP8", "resource_limits": {"lds_element_bytes": 1}})
    big = _with(tile_m=128, tile_n=128, tile_k=64, pipeline_depth=3, unroll_factor=4)
    assert not fits_lds(big) and fits_lds(big, 1)
    content = improvement_sample(child, "k")["messages"][2]["content"]
    assert "* 1 B" in content and "* 2 B" not in content


def test_training_experience_excludes_held_out_shapes(tmp_path):
    sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))
    from build_improvements import experience

    def legacy(ids):
        packet = {"portfolio": [{"workload_id": i} for i in ids]}
        return {"key": "k", "episode": {"messages": [{}, {"content": json.dumps(packet)}]}}

    c144 = ["gemm.small-square", "gemm.large-square", "gemm.tall", "gemm.wide", "gemm.deep-k"]
    rows = [legacy(c144), legacy(c144 + ["gemm.decode"]), {"key": "t", "portfolio": "heldout"},
            {"key": "c", "portfolio": "c144"}, {"key": "l", "portfolio": "large"}]
    (tmp_path / "e2e_s0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = experience([tmp_path])
    assert [r.get("portfolio") for r in kept] == [None, "c144", "large"]


def _record(arm, best, t104, wasted=0.1, wall=300.0):
    failed = round(10 * wasted)
    trace = [{"stage": "resource_rejected"}] * failed + [{"stage": "measured"}] * (10 - failed) + [{"stage": "duplicate"}] * 5
    return {"arm": arm, "best": {"source": "agent"}, "verified": {"speedup": best}, "wall_s": wall, "trace": trace,
            "metrics": {"final_best": best, "wasted_rate": wasted, "best_at": {"seconds": t104 or wall},
                        "reach": {"1.04": {"submissions": 3 if t104 else None, "seconds": t104}}}}


def test_gate_promotes_faster_kernel_or_faster_path_and_holds_otherwise():
    gate = PlannerGate()
    champ = [_record("c", 1.040, 200.0) for _ in range(4)]
    assert gate_decision(champ + [_record("n", 1.050, 200.0) for _ in range(4)], "c", "n", gate)["route"] == "faster_kernel"
    assert gate_decision(champ + [_record("n", 1.039, 120.0) for _ in range(4)], "c", "n", gate)["route"] == "faster_path"
    held = gate_decision(champ + [_record("n", 1.041, 190.0) for _ in range(4)], "c", "n", gate)
    assert held["decision"] == "HOLD"
    wasteful = gate_decision(champ + [_record("n", 1.06, 100.0, wasted=0.3) for _ in range(4)], "c", "n", gate)
    assert wasteful["decision"] == "HOLD"
