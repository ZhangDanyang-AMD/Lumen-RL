"""Multi-turn trajectory management for kernel agent RL training.

Structures raw JSONL events from :class:`TrajectoryWriter` into RL-consumable
multi-turn episodes compatible with LumenRL's :class:`DataProto`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """One agent turn: action → observation → reward."""

    turn_index: int
    action_text: str
    observation_text: str
    step_reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """One complete multi-turn episode."""

    episode_id: str
    task_id: str
    turns: list[Turn] = field(default_factory=list)
    final_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def turn_rewards(self) -> list[float]:
        return [t.step_reward for t in self.turns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "num_turns": self.num_turns,
            "final_reward": self.final_reward,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "action_text": t.action_text,
                    "observation_text": t.observation_text,
                    "step_reward": t.step_reward,
                    "done": t.done,
                }
                for t in self.turns
            ],
            "metadata": self.metadata,
        }


class TrajectoryCollector:
    """Accumulates turns during an episode and produces an :class:`Episode`."""

    def __init__(self, episode_id: str, task_id: str) -> None:
        self.episode_id = episode_id
        self.task_id = task_id
        self._turns: list[Turn] = []
        self._metadata: dict[str, Any] = {}

    def add_turn(
        self,
        action_text: str,
        observation_text: str,
        step_reward: float = 0.0,
        done: bool = False,
        **metadata: Any,
    ) -> Turn:
        turn = Turn(
            turn_index=len(self._turns),
            action_text=action_text,
            observation_text=observation_text,
            step_reward=step_reward,
            done=done,
            metadata=dict(metadata),
        )
        self._turns.append(turn)
        return turn

    def finalize(self, final_reward: float = 0.0, **metadata: Any) -> Episode:
        self._metadata.update(metadata)
        return Episode(
            episode_id=self.episode_id,
            task_id=self.task_id,
            turns=list(self._turns),
            final_reward=final_reward,
            metadata=self._metadata,
        )


def load_episodes_from_jsonl(path: Path | str) -> list[Episode]:
    """Parse a ``TrajectoryWriter`` JSONL file into :class:`Episode` objects.

    Groups events by ``run_start`` / ``round_end`` / ``run_end`` boundaries.
    Tool-call events within a round become individual turns.
    """
    path = Path(path)
    if not path.is_file():
        return []

    episodes: list[Episode] = []
    current_collector: TrajectoryCollector | None = None
    run_case_id = "unknown"

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = record.get("event", "")
        payload = record.get("payload", {})

        if event == "run_start":
            case_info = payload.get("case", {})
            run_case_id = str(case_info.get("case_id") or case_info.get("task_id") or "unknown")
            ep_id = f"{run_case_id}_{int(record.get('timestamp', 0))}"
            current_collector = TrajectoryCollector(ep_id, run_case_id)

        elif event == "tool_result" and current_collector is not None:
            params = payload.get("parameters", {})
            result = payload.get("result", {})
            reward_data = payload.get("reward", {})
            action_text = json.dumps(params, default=str)
            obs_text = json.dumps(result, default=str)
            step_reward = float(reward_data.get("total", 0.0) if isinstance(reward_data, dict) else 0.0)
            current_collector.add_turn(
                action_text=action_text,
                observation_text=obs_text,
                step_reward=step_reward,
            )

        elif event == "run_end" and current_collector is not None:
            final_reward = float(payload.get("final_reward", {}).get("total", 0.0)
                                 if isinstance(payload.get("final_reward"), dict) else 0.0)
            episode = current_collector.finalize(
                final_reward=final_reward,
                status=payload.get("status", "unknown"),
            )
            episodes.append(episode)
            current_collector = None

    if current_collector is not None:
        episodes.append(current_collector.finalize())

    return episodes


def episodes_to_tensors(
    episodes: Sequence[Episode],
    max_turns: int = 20,
) -> dict[str, torch.Tensor]:
    """Convert episodes into tensors for RL training.

    Returns dict with:
      - ``rewards`` [B]: final episode reward
      - ``turn_rewards`` [B, max_turns]: per-turn reward (zero-padded)
      - ``num_turns`` [B]: actual number of turns per episode
    """
    B = len(episodes)
    rewards = torch.zeros(B, dtype=torch.float32)
    turn_rewards = torch.zeros(B, max_turns, dtype=torch.float32)
    num_turns_t = torch.zeros(B, dtype=torch.long)

    for i, ep in enumerate(episodes):
        rewards[i] = ep.final_reward
        n = min(ep.num_turns, max_turns)
        num_turns_t[i] = n
        for t in range(n):
            turn_rewards[i, t] = ep.turns[t].step_reward

    return {
        "rewards": rewards,
        "turn_rewards": turn_rewards,
        "num_turns": num_turns_t,
    }
