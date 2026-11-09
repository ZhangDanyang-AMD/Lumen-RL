"""Step-level rejection-sampling selection."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from micake_rsi.rft import RftFilter, rft_sample, step_advantages  # noqa: E402


def _step(index, name, label, speedup=None, probe=False):
    perf = {"geomean_speedup_vs_baseline": speedup, "validity": "VALID"} if speedup is not None else None
    observation = {"probe": {"stage": label, "performance": perf}} if probe else ({"performance": perf} if perf else {})
    return {"step_index": index, "turn_index": index, "action": {"name": name},
            "result": {"execution_result": "diagnosis_probed" if probe else label, "observation": observation}}


def _episode(steps):
    messages = [{"role": "system"}, {"role": "user"}]
    for _ in steps:
        messages += [{"role": "assistant"}, {"role": "tool"}]
    return {"episode_id": "e", "messages": messages, "steps": steps, "environment": {"tools": []}}


def test_planner_keeps_only_fast_valid_submissions_and_the_report():
    steps = [_step(0, "submit_schedule", "measured", 1.04), _step(1, "submit_schedule", "resource_rejected"),
             _step(2, "submit_schedule", "measured", 0.70), _step(3, "report", "report")]
    assert step_advantages({"episode": _episode(steps)}, "planner", RftFilter()) == [1.0, -1.0, -1.0, 1.0]


def test_planner_report_is_dropped_when_nothing_was_kept():
    steps = [_step(0, "submit_schedule", "resource_rejected"), _step(1, "report", "report")]
    assert step_advantages({"episode": _episode(steps)}, "planner", RftFilter()) == [-1.0, -1.0]


def test_parallel_tool_call_episode_is_dropped():
    steps = [_step(0, "submit_schedule", "measured", 1.04), _step(1, "report", "report")]
    episode = _episode(steps)
    episode["messages"][2]["tool_calls"] = [{}, {}]
    assert step_advantages({"episode": episode}, "planner", RftFilter()) == [-1.0, -1.0]


def test_critic_keeps_probes_close_to_the_context_best():
    good = {"episode": _episode([_step(0, "report_diagnosis", "measured", 1.03, probe=True)]), "context_best": 1.045}
    weak = {"episode": _episode([_step(0, "report_diagnosis", "measured", 0.90, probe=True)]), "context_best": 1.045}
    assert step_advantages(good, "critic", RftFilter()) == [1.0]
    assert step_advantages(weak, "critic", RftFilter()) == [-1.0]


def test_sample_points_targets_at_assistant_messages():
    steps = [_step(0, "submit_schedule", "measured", 1.04), _step(1, "report", "report")]
    sample = rft_sample({"key": "k", "episode": _episode(steps)}, "planner", RftFilter())
    assert [s["assistant_message_index"] for s in sample["steps"]] == [2, 4]
