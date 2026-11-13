"""Multi-turn, tool-calling rollouts of prompt groups on the Ray vLLM or ATOM replicas."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import torch

from lumenrl.agent.chat import ChatTemplateTokenizer
from lumenrl.agent.deepseek_v4 import (
    V41_TAGS,
    DeepSeekV41ChatTokenizer,
    DeepSeekV4ChatTokenizer,
    DsmlToolParser,
)
from lumenrl.agent.loop import ToolAgentLoop
from lumenrl.agent.runner import AgentRunner
from lumenrl.agent.tool_parser import HermesToolParser
from lumenrl.agent.tools import SandboxFusionCodeTool, StatefulSandboxFusionCodeTool
from lumenrl.agent.trajectory import Trajectory

STOP_REASONS = ("answer", "length", "max_turns", "abort")


class AgentGroupRollout:
    """One ``ToolAgentLoop`` rollout per completion of a prompt group.

    A group's completions share their prompt, so they all go to one replica,
    the one with the fewest groups in flight, and share its prefix cache.
    Rollouts run on a private event loop; ``submit`` may be called from any
    thread.
    """

    def __init__(
        self,
        agent: ToolAgentLoop,
        servers: list[Any],
        *,
        max_response_tokens: int,
        max_sequence_tokens: int | None = None,
        system_prompt: str = "",
        cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if not servers:
            raise ValueError("AgentGroupRollout needs at least one rollout replica")
        self._agent = agent
        self._servers = list(servers)
        self._max_response_tokens = int(max_response_tokens)
        self._max_sequence_tokens = max_sequence_tokens
        self._system_prompt = system_prompt
        self._cleanup = cleanup
        self._runner = AgentRunner()
        self._lock = threading.Lock()
        self._load = [0] * len(self._servers)

    def submit(
        self,
        messages: list[dict[str, Any]],
        num_generations: int,
        sampling_params: dict[str, Any],
    ) -> concurrent.futures.Future[list[Trajectory]]:
        if self._system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self._system_prompt}, *messages]
        with self._lock:
            replica = min(range(len(self._load)), key=self._load.__getitem__)
            self._load[replica] += 1
        future = self._runner.submit(
            self._group(self._servers[replica], messages, num_generations, dict(sampling_params)),
        )
        future.add_done_callback(lambda _done: self._release(replica))
        return future

    def run(
        self,
        messages_list: list[list[dict[str, Any]]],
        num_generations: int,
        sampling_params: dict[str, Any],
    ) -> list[list[Trajectory]]:
        """Roll out every prompt group and return them in order."""
        futures = [self.submit(m, num_generations, sampling_params) for m in messages_list]
        return [future.result() for future in futures]

    def close(self) -> None:
        self._runner.close(self._cleanup)

    def _release(self, replica: int) -> None:
        with self._lock:
            self._load[replica] -= 1

    async def _group(
        self,
        server: Any,
        messages: list[dict[str, Any]],
        num_generations: int,
        sampling_params: dict[str, Any],
    ) -> list[Trajectory]:
        async def generate(prompt_ids: list[int], params: dict[str, Any]) -> dict[str, Any]:
            return await server.generate.remote(prompt_ids, params)

        return list(await asyncio.gather(*(
            self._agent.run(
                generate, messages, sampling_params,
                self._max_response_tokens, self._max_sequence_tokens,
            )
            for _ in range(num_generations)
        )))


def _chat_and_parser(mt: Any, tokenizer: Any) -> tuple[Any, Any]:
    """The tokenization and tool-call parser of ``multi_turn.format``."""
    kwargs = dict(mt.chat_template_kwargs or {})
    if mt.format == "hermes":
        return ChatTemplateTokenizer(tokenizer, kwargs), HermesToolParser()
    if mt.format == "dsml":
        return DeepSeekV4ChatTokenizer(tokenizer, **kwargs), DsmlToolParser()
    if mt.format == "dsml_v41":
        return DeepSeekV41ChatTokenizer(tokenizer, **kwargs), DsmlToolParser(V41_TAGS)
    raise ValueError(
        f"multi_turn.format={mt.format!r}: supported are 'hermes' (Qwen), 'dsml' (DeepSeek-V4) "
        "and 'dsml_v41' (DeepSeek-V4.1)"
    )


def build_agent_rollout(config: Any, tokenizer: Any, servers: list[Any]) -> AgentGroupRollout:
    """The agent rollout ``policy.generation.multi_turn`` describes."""
    mt = config.policy.generation.multi_turn
    chat, parser = _chat_and_parser(mt, tokenizer)
    tools = []
    ci = mt.code_interpreter
    if ci.enabled:
        tool_class = StatefulSandboxFusionCodeTool if ci.stateful else SandboxFusionCodeTool
        tools.append(tool_class(
            ci.url,
            language=ci.language,
            run_timeout_s=ci.run_timeout_s,
            memory_limit_mb=ci.memory_limit_mb,
            max_concurrency=ci.max_concurrency,
        ))
    agent = ToolAgentLoop(
        chat,
        tools,
        parser=parser,
        max_assistant_turns=mt.max_assistant_turns,
        max_parallel_calls=mt.max_parallel_calls,
        max_tool_response_length=mt.max_tool_response_length,
        tool_response_truncate_side=mt.tool_response_truncate_side,
        scripted_first_turn=mt.scripted_first_turn,
    )

    async def close_tools() -> None:
        for tool in tools:
            await tool.close()

    return AgentGroupRollout(
        agent,
        servers,
        max_response_tokens=int(config.policy.max_response_length),
        max_sequence_tokens=int(config.policy.max_total_sequence_length),
        system_prompt=mt.system_prompt,
        cleanup=close_tools,
    )


def mask_environment_tokens(
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    token_masks: list[list[int]],
) -> torch.Tensor:
    """Zero the environment tokens of left-padded rows in a ``[B, S-1]`` response mask.

    ``token_masks[i]`` has one entry per response token of row ``i`` (1 for a
    model token); column ``c`` of ``response_mask`` scores sequence position
    ``c + 1``, as ``RLTrainer._build_response_mask`` builds it.
    """
    if len(token_masks) != response_mask.shape[0]:
        raise ValueError(
            f"{len(token_masks)} agent token masks for {response_mask.shape[0]} rows"
        )
    masked = response_mask.clone()
    for i, (plen, tokens) in enumerate(zip(prompt_lengths, token_masks)):
        start = int((attention_mask[i] == 1).nonzero(as_tuple=True)[0][0].item())
        first = start + int(plen) - 1
        if first + len(tokens) > masked.shape[1]:
            raise ValueError(
                f"row {i}: {len(tokens)} agent mask entries past the response end"
            )
        keep = torch.tensor(tokens, dtype=masked.dtype, device=masked.device)
        masked[i, first:first + len(tokens)] *= keep
    return masked


def num_turns(stats: dict[str, Any]) -> int:
    """verl's ``num_turns``: the prompt, every assistant turn and every tool round."""
    return int(stats["assistant_turns"]) + int(stats["tool_rounds"]) + 1


def verl_agent_loop_metrics(
    stats: list[dict[str, Any]],
    prompt_lengths: list[int],
    response_lengths: list[int],
) -> dict[str, float]:
    """verl's per-rollout ``num_turns`` and ``timing_s/agent_loop/*`` metrics.

    The slowest rollout is the one with the most generate plus tool time, as
    verl picks it; rows are in the order of ``stats``.
    """
    if not stats:
        return {}
    turns = [float(num_turns(s)) for s in stats]
    generate = [float(s["generate_s"]) for s in stats]
    tool = [float(s["tool_s"]) for s in stats]
    metrics: dict[str, float] = {}
    for name, values in (
        ("num_turns", turns),
        ("timing_s/agent_loop/generate_sequences", generate),
        ("timing_s/agent_loop/tool_calls", tool),
    ):
        metrics[f"{name}/max"] = max(values)
        metrics[f"{name}/mean"] = sum(values) / len(values)
        metrics[f"{name}/min"] = min(values)
    slowest = max(range(len(stats)), key=lambda i: generate[i] + tool[i])
    metrics["timing_s/agent_loop/slowest/generate_sequences"] = generate[slowest]
    metrics["timing_s/agent_loop/slowest/tool_calls"] = tool[slowest]
    metrics["timing_s/agent_loop/slowest/prompt_length"] = float(prompt_lengths[slowest])
    metrics["timing_s/agent_loop/slowest/response_length"] = float(response_lengths[slowest])
    return metrics


def agent_metrics(stats: list[dict[str, Any]]) -> dict[str, float]:
    """Step metrics from the ``Trajectory.stats()`` of every trained row."""
    if not stats:
        return {}
    n = len(stats)

    def mean(key: str) -> float:
        return sum(float(s[key]) for s in stats) / n

    calls = sum(int(s["tool_calls"]) for s in stats)
    metrics = {
        "agent/assistant_turns_mean": mean("assistant_turns"),
        "agent/assistant_turns_max": float(max(int(s["assistant_turns"]) for s in stats)),
        "agent/tool_calls_mean": mean("tool_calls"),
        "agent/tool_call_frac": sum(int(s["tool_calls"]) > 0 for s in stats) / n,
        "agent/tool_error_rate": (
            sum(int(s["tool_errors"]) for s in stats) / calls if calls else 0.0
        ),
        "agent/malformed_tool_calls_mean": mean("malformed_tool_calls"),
        "agent/model_tokens_mean": mean("model_tokens"),
        "agent/environment_tokens_mean": mean("environment_tokens"),
        "agent/generate_s_mean": mean("generate_s"),
        "agent/tool_s_mean": mean("tool_s"),
    }
    for reason in STOP_REASONS:
        metrics[f"agent/stop_{reason}_frac"] = sum(s["stop_reason"] == reason for s in stats) / n
    return metrics
