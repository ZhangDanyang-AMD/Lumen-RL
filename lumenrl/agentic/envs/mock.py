"""CPU-only mock environment: find a hidden integer with higher/lower feedback.

Exercises the full AgentEnvironment protocol (tools, rewards, budget, snapshot)
without any GPU or external backend.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Mapping

from lumenrl.agentic.core.backends import register_environment
from lumenrl.agentic.core.protocols import (
    AgentEnvironment,
    EnvironmentCapabilities,
    ResetResult,
    StepResult,
    ToolCall,
)

BACKEND_NAME = "mock"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_guess",
            "description": "Submit an integer guess for the hidden target.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the task.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class MockGuessEnvironment(AgentEnvironment):
    """Hidden-integer search with ``higher``/``lower`` observations."""

    def __init__(self, max_submissions: int) -> None:
        self._max_submissions = max_submissions
        self._state: dict[str, Any] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(backend=BACKEND_NAME, snapshot=True, fidelity_levels=("cpu",))

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        low, high = int(task["low"]), int(task["high"])
        self._state = {"target": int(task["target"]), "low": low, "high": high, "submissions": 0, "solved": False}
        return ResetResult(
            system_prompt="Find the hidden integer. Use submit_guess; each result says higher, lower or correct.",
            observation={"range": [low, high], "max_submissions": self._max_submissions},
            tools=copy.deepcopy(TOOLS),
            metadata={"backend": BACKEND_NAME, "range": [low, high]},
        )

    async def step(self, action: ToolCall) -> StepResult:
        if action.name == "finish":
            return StepResult({"status": "finished"}, 0.0, True, "AGENT_FINISHED", "finish")
        if action.name != "submit_guess" or action.arguments is None or type(action.arguments.get("value")) is not int:
            return StepResult({"error": "INVALID_ACTION"}, -1.0, False, "RUNNING", "invalid")
        self._state["submissions"] += 1
        value, target = action.arguments["value"], self._state["target"]
        if value == target:
            self._state["solved"] = True
            return StepResult({"result": "correct"}, 1.0, True, "SUCCESS", "correct")
        done = self._state["submissions"] >= self._max_submissions
        hint = "higher" if value < target else "lower"
        return StepResult(
            {"result": hint, "remaining": self._max_submissions - self._state["submissions"]},
            -0.1, done, "BUDGET_EXHAUSTED" if done else "RUNNING", "wrong",
        )

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        return 0.0, {"solved": self._state["solved"], "submissions": self._state["submissions"]}

    async def snapshot(self) -> str:
        ref = uuid.uuid4().hex
        self._snapshots[ref] = copy.deepcopy(self._state)
        return ref

    async def restore(self, state_ref: str) -> None:
        self._state = copy.deepcopy(self._snapshots[state_ref])


register_environment(BACKEND_NAME, MockGuessEnvironment)
