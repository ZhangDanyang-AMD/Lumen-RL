"""One multi-turn rollout: the model's turns interleaved with what the tools wrote."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    """Token ids of a multi-turn rollout, in the order the engine saw them.

    ``response_ids`` holds everything after the prompt: model turns and the
    environment tokens between them (tool responses and the chat-template glue
    around them). ``response_mask`` is 1 where the model sampled the token and 0
    where the environment wrote it, so only the model's tokens carry loss.
    ``response_logprobs`` is the sampling log-prob of each model token and 0.0
    for environment tokens.
    """

    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    assistant_turns: int = 0
    # Rounds of tool output appended; verl counts them as user turns.
    tool_rounds: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    malformed_tool_calls: int = 0
    # Why the rollout ended: "answer" (a turn without tool calls), "max_turns",
    # "length" (out of response budget) or "abort" (the engine aborted it).
    stop_reason: str = ""
    generate_s: float = 0.0
    tool_s: float = 0.0
    # The experts the engine routed each token to, from the last turn: its input
    # was the whole trajectory, so this is [prompt + response - 1, layers,
    # top_k] as a single-turn completion's. Set when the engine returns them.
    routed_experts: Any = None

    def add_model_turn(self, token_ids: list[int], logprobs: list[float] | None) -> None:
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(
                f"{len(logprobs)} log-probs for {len(token_ids)} generated tokens"
            )
        self.response_ids.extend(token_ids)
        self.response_mask.extend([1] * len(token_ids))
        self.response_logprobs.extend(logprobs if logprobs is not None else [0.0] * len(token_ids))
        self.assistant_turns += 1

    def add_environment(self, token_ids: list[int]) -> None:
        self.response_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        self.response_logprobs.extend([0.0] * len(token_ids))
        self.tool_rounds += 1

    @property
    def model_tokens(self) -> int:
        return sum(self.response_mask)

    def stats(self) -> dict[str, Any]:
        return {
            "assistant_turns": self.assistant_turns,
            "tool_rounds": self.tool_rounds,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "malformed_tool_calls": self.malformed_tool_calls,
            "stop_reason": self.stop_reason,
            "model_tokens": self.model_tokens,
            "environment_tokens": len(self.response_ids) - self.model_tokens,
            "generate_s": self.generate_s,
            "tool_s": self.tool_s,
        }

    def to_rollout_result(self) -> dict[str, Any]:
        """The trajectory in the shape ``VLLMRayServer.generate`` returns, plus
        its ``response_mask`` and ``agent_stats``."""
        result = {
            "prompt_token_ids": list(self.prompt_ids),
            "token_ids": list(self.response_ids),
            "logprobs": list(self.response_logprobs),
            "response_mask": list(self.response_mask),
            "agent_stats": self.stats(),
        }
        if self.routed_experts is not None:
            result["routed_experts"] = self.routed_experts
        return result
