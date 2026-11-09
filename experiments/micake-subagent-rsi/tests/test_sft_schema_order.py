"""Training prompts must render tools and arguments in the order live rollouts send."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from micake_rsi.search_harness import submission_arguments  # noqa: E402
from micake_rsi.sft_data import _template_messages, canonical_schema  # noqa: E402
from sandbox.micake.tool_runtime import tool_schemas  # noqa: E402

DOMAIN = {
    "schedule": {"pipeline_depth": [2, 3], "tile_k": [16, 32], "tile_m": [32, 64]},
    "realization": {"kpack": [1], "num_warps": [4, 8], "program_order": ["TWO_DIMENSIONAL", "GROUPED_M"]},
}


def test_sorted_round_trip_restores_live_schema_order():
    live = tool_schemas(DOMAIN)
    stored = json.loads(json.dumps(live, sort_keys=True))
    assert json.dumps(stored) != json.dumps(live)
    assert json.dumps(canonical_schema(stored)) == json.dumps(live)


def test_arguments_follow_required_order():
    tools = canonical_schema(json.loads(json.dumps(tool_schemas(DOMAIN), sort_keys=True)))
    arguments = {"hypothesis": "h", "realization": {"num_warps": 4}, "schedule": {"tile_k": 16}}
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "submit_schedule", "arguments": json.dumps(arguments)}}]}
    rendered = _template_messages([message], tools)[0]["tool_calls"][0]["function"]["arguments"]
    assert list(rendered) == ["schedule", "realization", "hypothesis"]


def test_submission_arguments_put_hypothesis_last():
    config = {"schedule": {"tile_k": 16}, "realization": {"kpack": 1, "num_warps": 4}}
    assert list(submission_arguments(config, "h")) == ["schedule", "realization", "hypothesis"]
