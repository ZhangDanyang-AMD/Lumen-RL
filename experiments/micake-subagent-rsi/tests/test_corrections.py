"""MICake-verified corrections of resource-rejected turns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from micake_rsi.corrections import correction_sample, repairs  # noqa: E402
from micake_rsi.search_harness import fits_lds, lds_check, shared_bytes  # noqa: E402

DOMAIN = {
    "schedule": {"tile_m": [32, 64, 128], "tile_n": [32, 64, 128], "tile_k": [16, 32, 64], "pipeline_depth": [2, 3, 4],
                 "unroll_factor": [4, 8], "vectorize_factor": [4, 8]},
    "realization": {"num_stages": [2, 3, 4], "num_warps": [4, 8], "kpack": [1]},
}
REJECTED = {"schedule": {"tile_m": 128, "tile_n": 128, "tile_k": 64, "pipeline_depth": 4, "unroll_factor": 4, "vectorize_factor": 4},
            "realization": {"num_stages": 4, "num_warps": 8, "kpack": 1}}


def test_shared_bytes_matches_compiled_facts():
    def config(tm, tn, tk, pd, u):
        return {"schedule": {"tile_m": tm, "tile_n": tn, "tile_k": tk, "pipeline_depth": pd, "unroll_factor": u}}

    assert shared_bytes(config(128, 32, 32, 4, 2)) == 20480
    assert shared_bytes(config(64, 128, 32, 4, 2)) == 24576
    assert shared_bytes(config(64, 64, 32, 3, 4)) == 65536
    assert not fits_lds(config(128, 128, 64, 3, 4))
    assert "= 262144 B > 163840 B, too big" in lds_check(config(128, 128, 64, 3, 4))
    assert "buffers = (pipeline_depth 3 - 1) * unroll 4 = 8" in lds_check(config(128, 128, 64, 3, 4))


def test_repairs_fit_exactly_and_change_one_field_first():
    candidates = list(repairs(REJECTED, DOMAIN))
    first = candidates[0]
    changed = [(sec, k) for sec in ("schedule", "realization") for k in first[sec] if first[sec][k] != REJECTED[sec][k]]
    assert len(changed) == 1
    for c in candidates:
        assert fits_lds(c) and c["realization"] == REJECTED["realization"]
        assert c["schedule"]["vectorize_factor"] == REJECTED["schedule"]["vectorize_factor"]
    assert len({json.dumps(c, sort_keys=True) for c in candidates}) == len(candidates)


def test_correction_replaces_only_the_rejected_turn():
    call = {"id": "c1", "type": "function", "function": {"name": "submit_schedule", "arguments": "{}"}}
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "big tiles", "tool_calls": [call]}, {"role": "tool", "content": "rejected"}]
    record = {"key": "k", "episode": {"episode_id": "e", "messages": messages, "environment": {"tools": []},
                                       "steps": [{"step_index": 0, "turn_index": 0}]}}
    repaired = next(repairs(REJECTED, DOMAIN))
    sample = correction_sample(record, 0, repaired, REJECTED)
    assert len(sample["messages"]) == 3 and sample["steps"][0]["assistant_message_index"] == 2
    args = json.loads(sample["messages"][2]["tool_calls"][0]["function"]["arguments"])
    assert args["schedule"] == repaired["schedule"] and "kpack" not in args["realization"]
    assert "shared" in sample["messages"][2]["content"].lower() and messages[2]["content"] == "big tiles"
    assert lds_check(repaired) in sample["messages"][2]["content"]


def test_duplicate_repairs_move_one_axis_and_message_names_the_repeat():
    from micake_rsi.corrections import corrected_message, duplicate_repairs

    base = {"schedule": {"tile_m": 64, "tile_n": 64, "tile_k": 32, "pipeline_depth": 3, "unroll_factor": 4, "vectorize_factor": 4},
            "realization": {"num_stages": 3, "num_warps": 4, "kpack": 1}}
    first = next(duplicate_repairs(base, DOMAIN))
    changed = [k for sec in ("schedule", "realization") for k in first[sec] if first[sec][k] != base[sec][k]]
    assert 1 <= len(changed) <= 2
    call = {"id": "c", "type": "function", "function": {"name": "submit_schedule", "arguments": "{}"}}
    message = corrected_message({"role": "assistant", "content": "", "tool_calls": [call]}, base, first, kind="duplicate")
    assert "already" in message["content"]


def test_annotate_lds_replaces_model_written_checks():
    from micake_rsi.corrections import annotate_lds

    config = {"schedule": {"tile_m": 64, "tile_n": 64, "tile_k": 32, "pipeline_depth": 3, "unroll_factor": 4}}
    message = {"role": "assistant", "content": "LDS check: LDS = wrong = 1 B <= 163840 B.\nKeep the tile."}
    content = annotate_lds(message, config)["content"]
    assert content.splitlines() == [f"LDS check: {lds_check(config)}.", "Keep the tile."]
