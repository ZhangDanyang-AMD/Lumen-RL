"""Episode / Trajectory / Step records and a JSONL trajectory store.

The schema is backend-neutral: environment-specific facts live in
``StepResult.observation``/``metadata`` and ``ReceiptRef`` handles.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from lumenrl.agentic.core.protocols import StepResult, ToolCall

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "agent_episode_v1"


@dataclass
class PolicyTurn:
    """One model response inside an episode.

    Attributes:
        turn_index: Zero-based model-call index.
        model: Model identifier reported by the rollout backend.
        content: Assistant text content.
        reasoning: Reasoning text if the backend exposes it.
        tool_calls: Parsed tool calls issued in this turn.
        finish_reason: Backend finish reason.
        usage: Token usage reported by the backend.
        prompt_token_ids: Prompt token IDs if the backend returns them.
        response_token_ids: Response token IDs if the backend returns them.
        response_logprobs: Per-token rollout logprobs if the backend returns them.
        logprob_status: ``aligned`` when ids, logprobs and usage describe one completion.
        token_ids_cover: ``completion`` when ``response_token_ids`` is the sampled
            completion, including tool-call tokens.
        latency_s: Wall-clock latency of the model call.
    """

    turn_index: int
    model: str
    content: str
    reasoning: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: list[int] | None = None
    response_token_ids: list[int] | None = None
    response_logprobs: list[float] | None = None
    logprob_status: str | None = None
    token_ids_cover: str | None = None
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "model": self.model,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "prompt_token_ids": self.prompt_token_ids,
            "response_token_ids": self.response_token_ids,
            "response_logprobs": self.response_logprobs,
            "logprob_status": self.logprob_status,
            "token_ids_cover": self.token_ids_cover,
            "latency_s": self.latency_s,
        }


@dataclass
class Step:
    """One environment transition caused by one tool call."""

    step_index: int
    turn_index: int
    action: ToolCall
    result: StepResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "turn_index": self.turn_index,
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass
class Episode:
    """Complete multi-turn rollout of one policy on one task.

    Attributes:
        episode_id: Unique identifier.
        group_id: Identifier shared by episodes sampled for the same task (for
            group-relative advantages).
        task: Task specification passed to ``AgentEnvironment.reset``.
        environment: Frozen environment/task identity from ``ResetResult.metadata``.
        policy: Policy identity (model id, sampling settings).
        messages: Full chat transcript sent to / received from the policy.
        turns: Model responses.
        steps: Environment transitions.
        terminal_state: Final state (see ``TERMINAL_STATES``).
        total_reward: Sum of step rewards plus any terminal reward.
        terminal_reward: Reward assigned at episode end by the environment.
        parent_episode_id: Episode that spawned this one (hierarchical agents).
        role: Agent role that produced this episode (e.g. a specialist role name).
    """

    episode_id: str
    group_id: str
    task: dict[str, Any]
    environment: dict[str, Any]
    policy: dict[str, Any]
    parent_episode_id: str | None = None
    role: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: list[PolicyTurn] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    terminal_state: str = "RUNNING"
    total_reward: float = 0.0
    terminal_reward: float = 0.0
    wall_time_s: float = 0.0

    @property
    def step_rewards(self) -> list[float]:
        return [step.result.reward for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "parent_episode_id": self.parent_episode_id,
            "role": self.role,
            "task": self.task,
            "environment": self.environment,
            "policy": self.policy,
            "messages": self.messages,
            "turns": [turn.to_dict() for turn in self.turns],
            "steps": [step.to_dict() for step in self.steps],
            "terminal_state": self.terminal_state,
            "total_reward": self.total_reward,
            "terminal_reward": self.terminal_reward,
            "wall_time_s": self.wall_time_s,
        }


class TrajectoryStore:
    """Append-only JSONL store for episodes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, episode: Episode) -> None:
        """Append one episode as a single JSON line."""
        line = json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        logger.info("stored episode %s (%s) -> %s", episode.episode_id, episode.terminal_state, self.path)

    def load(self) -> list[Mapping[str, Any]]:
        """Load all stored episodes as dictionaries."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
