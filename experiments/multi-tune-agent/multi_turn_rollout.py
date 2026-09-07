"""Multi-turn rollout support for kernel agent RL training.

Wraps LumenRL's single-turn generation in an outer loop that feeds GEAK
sandbox observations back into the model, producing multi-turn episodes
with per-turn log-probs and token-level turn boundary markers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class RolloutTurn:
    """One turn within a rollout episode."""

    turn_index: int
    action_token_ids: Tensor       # [T_action]
    action_log_probs: Tensor       # [T_action]
    observation_text: str = ""
    reward: float = 0.0
    done: bool = False


@dataclass
class RolloutEpisode:
    """Complete multi-turn rollout with token sequences and turn boundaries."""

    episode_id: str
    task_id: str
    prompt_token_ids: Tensor        # [T_prompt]
    turns: list[RolloutTurn] = field(default_factory=list)
    final_reward: float = 0.0

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def total_tokens(self) -> int:
        n = self.prompt_token_ids.shape[0]
        for t in self.turns:
            n += t.action_token_ids.shape[0]
        return n


def concat_turns_for_training(
    episodes: Sequence[RolloutEpisode],
    max_seq_len: int = 8192,
    pad_token_id: int = 0,
    max_turns: int = 20,
) -> dict[str, Tensor]:
    """Concatenate multi-turn episodes into padded training tensors.

    Returns dict with:
      - ``input_ids`` [B, max_seq_len]
      - ``log_probs`` [B, max_seq_len] (0 for prompt/obs tokens)
      - ``response_mask`` [B, max_seq_len] (1 for action tokens only)
      - ``turn_ids`` [B, max_seq_len] (turn index per token, 0 for prompt)
      - ``rewards`` [B]
      - ``turn_rewards`` [B, max_turns]
    """
    B = len(episodes)
    input_ids = torch.full((B, max_seq_len), pad_token_id, dtype=torch.long)
    log_probs = torch.zeros(B, max_seq_len, dtype=torch.float32)
    response_mask = torch.zeros(B, max_seq_len, dtype=torch.float32)
    turn_ids = torch.zeros(B, max_seq_len, dtype=torch.long)
    rewards = torch.zeros(B, dtype=torch.float32)
    turn_rewards = torch.zeros(B, max_turns, dtype=torch.float32)

    for i, ep in enumerate(episodes):
        pos = 0
        prompt_len = min(ep.prompt_token_ids.shape[0], max_seq_len)
        input_ids[i, :prompt_len] = ep.prompt_token_ids[:prompt_len]
        pos = prompt_len

        for t in ep.turns:
            action_len = t.action_token_ids.shape[0]
            end = min(pos + action_len, max_seq_len)
            actual = end - pos
            if actual <= 0:
                break

            input_ids[i, pos:end] = t.action_token_ids[:actual]
            log_probs[i, pos:end] = t.action_log_probs[:actual]
            response_mask[i, pos:end] = 1.0
            turn_ids[i, pos:end] = t.turn_index + 1  # 0 reserved for prompt
            pos = end

            if t.turn_index < max_turns:
                turn_rewards[i, t.turn_index] = t.reward

        rewards[i] = ep.final_reward

    return {
        "input_ids": input_ids,
        "log_probs": log_probs,
        "response_mask": response_mask,
        "turn_ids": turn_ids,
        "rewards": rewards,
        "turn_rewards": turn_rewards,
    }


class MultiTurnRolloutManager:
    """Orchestrates multi-turn generation with a GEAK gym environment.

    Parameters:
        generate_fn: Callable that takes (token_ids [1, T]) and returns
            (new_token_ids [1, T'], log_probs [1, T']).
        tokenizer: Object with ``encode(text) -> list[int]`` and
            ``decode(ids) -> str`` methods.
        env: A ``GEAKGymEnv`` instance.
        max_turns: Maximum turns per episode.
        max_seq_len: Maximum total sequence length.
    """

    def __init__(
        self,
        generate_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
        tokenizer: Any,
        env: Any,
        max_turns: int = 20,
        max_seq_len: int = 8192,
    ) -> None:
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer
        self.env = env
        self.max_turns = max_turns
        self.max_seq_len = max_seq_len

    def rollout(
        self, prompt_text: str, task_options: dict[str, Any] | None = None
    ) -> RolloutEpisode:
        """Run one multi-turn episode.

        1. Reset env → initial observation
        2. Build prompt with observation → generate action
        3. Step env with action → next observation
        4. Repeat until done or max_turns
        """
        obs, info = self.env.reset(options=task_options)
        task_id = info.get("task_id", "unknown")
        episode_id = f"{task_id}_{id(obs)}"

        context = prompt_text + "\n\n" + obs.to_text()
        prompt_ids = torch.tensor(
            self.tokenizer.encode(context), dtype=torch.long
        )

        episode = RolloutEpisode(
            episode_id=episode_id,
            task_id=task_id,
            prompt_token_ids=prompt_ids,
        )

        for turn_idx in range(self.max_turns):
            context_ids = self._build_context(episode)
            if context_ids.shape[-1] >= self.max_seq_len:
                break

            action_ids, action_logp = self.generate_fn(context_ids.unsqueeze(0))
            action_ids = action_ids.squeeze(0)
            action_logp = action_logp.squeeze(0)

            action_text = self.tokenizer.decode(action_ids.tolist())
            obs, reward, terminated, truncated, step_info = self.env.step(action_text)

            turn = RolloutTurn(
                turn_index=turn_idx,
                action_token_ids=action_ids,
                action_log_probs=action_logp,
                observation_text=obs.to_text(),
                reward=reward,
                done=terminated or truncated,
            )
            episode.turns.append(turn)

            if terminated or truncated:
                episode.final_reward = reward
                break

        return episode

    def _build_context(self, episode: RolloutEpisode) -> Tensor:
        """Build the full context sequence from prompt + previous turns."""
        parts = [episode.prompt_token_ids]
        for turn in episode.turns:
            parts.append(turn.action_token_ids)
            obs_ids = torch.tensor(
                self.tokenizer.encode(turn.observation_text),
                dtype=torch.long,
            )
            parts.append(obs_ids)
        return torch.cat(parts, dim=0)

    def batch_rollout(
        self,
        prompts: Sequence[str],
        task_options_list: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[RolloutEpisode]:
        """Run rollouts for multiple prompts sequentially."""
        options_list = task_options_list or [None] * len(prompts)
        return [
            self.rollout(prompt, opts)
            for prompt, opts in zip(prompts, options_list)
        ]
