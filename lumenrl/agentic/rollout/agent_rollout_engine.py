"""Multi-turn agentic rollout: policy <-> AgentEnvironment -> Episode."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from lumenrl.agentic.core.protocols import AgentEnvironment, StepResult, ToolCall
from lumenrl.agentic.core.trajectory import Episode, PolicyTurn, Step
from lumenrl.agentic.rollout.policy import AgentPolicy, PolicyError

logger = logging.getLogger(__name__)

NO_TOOL_CALL_NUDGE = (
    "You did not call a tool. Continue by calling one of the provided tools; "
    "call the finish tool when you want to end the task."
)
REJECTED_PARALLEL_CALL = {"error": "PARALLEL_TOOL_CALL_REJECTED", "detail": "only one tool call per turn is executed"}


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout limits.

    Attributes:
        max_turns: Maximum model calls per episode.
        max_no_tool_nudges: Times a turn without tool calls is answered with a
            reminder before the episode ends as ``AGENT_FINISHED``.
    """

    max_turns: int
    max_no_tool_nudges: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutConfig":
        return cls(max_turns=int(value["max_turns"]), max_no_tool_nudges=int(value["max_no_tool_nudges"]))


def _assistant_message(turn: PolicyTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in turn.tool_calls
        ]
    return message


def _tool_message(call: ToolCall, observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.call_id,
        "content": json.dumps(observation, ensure_ascii=False, sort_keys=True),
    }


class AgenticRolloutEngine:
    """Runs episodes of one policy against environments, one tool call per turn."""

    def __init__(self, policy: AgentPolicy, config: RolloutConfig) -> None:
        self.policy = policy
        self.config = config

    async def run_episode(
        self,
        env: AgentEnvironment,
        task: Mapping[str, Any],
        group_id: str | None = None,
        parent_episode_id: str | None = None,
        role: str | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        """Run one episode to completion.

        Args:
            env: Environment instance (reset by this call).
            task: Task specification for ``env.reset``.
            group_id: Shared id for episodes sampled on the same task.
            parent_episode_id: Spawning episode for hierarchical rollouts.
            role: Role of the acting policy.
            episode_id: Pre-assigned id (lets environments reference the running episode).

        Returns:
            The completed episode including terminal reward.
        """
        start = time.monotonic()
        reset = await env.reset(task)
        episode = Episode(
            episode_id=episode_id or uuid.uuid4().hex,
            group_id=group_id or uuid.uuid4().hex,
            task=dict(task),
            environment=dict(reset.metadata),
            policy=self.policy.describe(),
            parent_episode_id=parent_episode_id,
            role=role,
        )
        episode.messages = [
            {"role": "system", "content": reset.system_prompt},
            {"role": "user", "content": json.dumps(reset.observation, ensure_ascii=False, sort_keys=True)},
        ]
        terminal_state = "BUDGET_EXHAUSTED"
        nudges = 0
        for turn_index in range(self.config.max_turns):
            try:
                turn = await self.policy.act(episode.messages, reset.tools, turn_index)
            except PolicyError as exc:
                logger.error("episode %s: policy failed: %s", episode.episode_id, exc)
                terminal_state = "ERROR"
                break
            episode.turns.append(turn)
            episode.messages.append(_assistant_message(turn))

            if not turn.tool_calls:
                if nudges < self.config.max_no_tool_nudges:
                    nudges += 1
                    episode.messages.append({"role": "user", "content": NO_TOOL_CALL_NUDGE})
                    continue
                terminal_state = "AGENT_FINISHED"
                break

            result = await self._execute(env, episode, turn)
            if result.done:
                terminal_state = result.terminal_state
                break

        terminal_reward, summary = await env.finalize(terminal_state)
        episode.terminal_state = terminal_state
        episode.terminal_reward = terminal_reward
        episode.environment["final_summary"] = summary
        episode.total_reward = sum(episode.step_rewards) + terminal_reward
        episode.wall_time_s = time.monotonic() - start
        logger.info(
            "episode %s finished: %s, %d turns, %d steps, reward %.3f",
            episode.episode_id, terminal_state, len(episode.turns), len(episode.steps), episode.total_reward,
        )
        return episode

    async def _execute(self, env: AgentEnvironment, episode: Episode, turn: PolicyTurn) -> StepResult:
        first, *rest = turn.tool_calls
        result = await env.step(first)
        episode.steps.append(Step(len(episode.steps), turn.turn_index, first, result))
        episode.messages.append(_tool_message(first, result.observation))
        for call in rest:
            episode.messages.append(_tool_message(call, REJECTED_PARALLEL_CALL))
        return result
