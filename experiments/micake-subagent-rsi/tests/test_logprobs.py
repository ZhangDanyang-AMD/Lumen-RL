"""Aligned rollout logprobs are kept; a tool-call turn without them is recomputed."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    sys.path.insert(0, str(entry))

from lumenrl.agentic.core.protocols import StepResult, ToolCall  # noqa: E402
from lumenrl.agentic.core.trajectory import Episode, PolicyTurn, Step  # noqa: E402
from lumenrl.agentic.rollout.token_alignment import ALIGNED, TOOL_TOKENS_MISSING  # noqa: E402
from micake_rsi.export import specialist_sample  # noqa: E402
from micake_rsi.rewards import SpecialistCredit  # noqa: E402


def _episode(status: str) -> Episode:
    turn = PolicyTurn(
        0, "m", "", None, [ToolCall.from_raw("c", "submit", "{}")], "tool_calls",
        prompt_token_ids=[1, 2], response_token_ids=[3, 4], response_logprobs=[-0.1, -0.2],
        logprob_status=status, token_ids_cover="completion" if status == ALIGNED else None,
    )
    child = Episode("e", "g", {}, {}, {}, role="planner", messages=[{"role": "assistant", "content": ""}])
    child.turns = [turn]
    child.steps = [Step(0, 0, turn.tool_calls[0], StepResult({}, 0.1, execution_result="measured"))]
    return child


def test_export_keeps_aligned_logprobs_and_recomputes_the_rest() -> None:
    manager = Episode("m", "g", {"task_id": "t"}, {}, {})
    credit = SpecialistCredit("e", "planner", 0.1, 0.0, 0.0, 0.1, [0.1], [0.1])
    kept = specialist_sample("g", _episode(ALIGNED), manager, credit)
    assert kept["requires_logprob_recompute"] is False
    assert kept["rollout_tokens"][0]["response_logprobs"] == [-0.1, -0.2]
    assert kept["rollout_tokens"][0]["response_token_ids"] == [3, 4]
    dropped = specialist_sample("g", _episode(TOOL_TOKENS_MISSING), manager, credit)
    assert dropped["requires_logprob_recompute"] is True
    assert dropped["rollout_tokens"][0]["logprob_status"] == TOOL_TOKENS_MISSING
