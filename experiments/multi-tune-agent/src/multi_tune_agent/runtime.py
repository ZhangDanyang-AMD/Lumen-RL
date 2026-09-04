"""Framework-independent long-horizon agent and tool runtime.

The lifecycle and state machine mirror the useful ideas from verl/Uni-Agent,
but this module has no dependency on either project.
"""

from __future__ import annotations

import abc
import asyncio
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence

import requests


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    text: str
    assistant_message: dict[str, Any]
    tool_calls: list[ToolCall]
    logprobs: list[float]
    usage: dict[str, Any]


class ModelBackend(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelTurn: ...


def _messages_for_generation(
    messages: Sequence[Mapping[str, Any]], max_chars: int = 80000
) -> list[dict[str, Any]]:
    """Compact replayable tool history while retaining recent evidence."""

    prepared = [copy.deepcopy(dict(message)) for message in messages]
    for index, message in enumerate(prepared):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                raw = function.get("arguments")
                if not isinstance(raw, str):
                    continue
                try:
                    arguments = json.loads(raw)
                except ValueError:
                    continue
                content = arguments.get("content") if isinstance(arguments, dict) else None
                if (
                    isinstance(arguments, dict)
                    and arguments.get("action") == "write_file"
                    and isinstance(content, str)
                    and len(content) > 512
                ):
                    arguments["content"] = (
                        "<omitted %d written characters; use read_file for current source>"
                        % len(content)
                    )
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        if message.get("role") == "tool" and index < len(prepared) - 2:
            content = message.get("content")
            if isinstance(content, str) and len(content) > 6000:
                message["content"] = (
                    content[:3000]
                    + "...<older tool observation compacted>..."
                    + content[-3000:]
                )

    if len(json.dumps(prepared, default=str, ensure_ascii=False)) <= max_chars:
        return prepared
    prefix_count = min(3, len(prepared))
    tail_start = max(prefix_count, len(prepared) - 8)
    while tail_start > prefix_count and prepared[tail_start].get("role") == "tool":
        tail_start -= 1
    omitted = prepared[prefix_count:tail_start]
    summary = {
        "role": "user",
        "content": (
            "[Context checkpoint] %d older messages were compacted. Their full "
            "tool calls and results remain in the trajectory. Inspect the current "
            "workspace with read_file and continue from the latest visible error."
            % len(omitted)
        ),
    }
    return prepared[:prefix_count] + [summary] + prepared[tail_start:]


class OpenAIModelBackend:
    """Small OpenAI-compatible backend with tool calls and token logprobs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "unused",
        timeout: float = 600.0,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.session = session or requests.Session()

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_generation(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        response = self.session.post(
            self.base_url + "/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer " + self.api_key},
            timeout=self.timeout,
        )
        if response.status_code == 400:
            detail = response.text
            match = re.search(
                r"maximum context length is (\d+) tokens.*request has (\d+) input tokens",
                detail,
                re.IGNORECASE,
            )
            if match:
                available = int(match.group(1)) - int(match.group(2)) - 64
                retry_tokens = min(self.max_tokens, available)
                if 128 <= retry_tokens < self.max_tokens:
                    retry_payload = dict(payload)
                    retry_payload["max_tokens"] = retry_tokens
                    response = self.session.post(
                        self.base_url + "/chat/completions",
                        json=retry_payload,
                        headers={"Authorization": "Bearer " + self.api_key},
                        timeout=self.timeout,
                    )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if len(detail) > 2000:
                detail = detail[:2000] + "...(truncated)"
            raise requests.HTTPError(
                "%s; response=%s" % (exc, detail), response=response
            ) from exc
        body = response.json()
        choice = body["choices"][0]
        message = dict(choice.get("message") or {})
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must be an object")
            except (TypeError, ValueError) as exc:
                parsed = {"_malformed_json": str(exc), "_raw": arguments}
            calls.append(
                ToolCall(
                    call_id=str(raw.get("id") or uuid.uuid4().hex),
                    name=str(function.get("name") or ""),
                    arguments=parsed,
                )
            )
        logprobs = []
        for token in (choice.get("logprobs") or {}).get("content") or []:
            value = token.get("logprob")
            if isinstance(value, (int, float)):
                logprobs.append(float(value))
        return ModelTurn(
            text=str(message.get("content") or ""),
            assistant_message=message,
            tool_calls=calls,
            logprobs=logprobs,
            usage=dict(body.get("usage") or {}),
        )


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass
class AgentLoopMetrics:
    total_seconds: float = 0.0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    assistant_turns: int = 0
    tool_calls: int = 0


@dataclass
class AgentLoopOutput:
    messages: list[dict[str, Any]]
    final_text: str
    policy_logprobs: list[float]
    tool_rewards: list[float]
    reward_score: float
    metrics: AgentLoopMetrics
    session_id: str
    events: list[dict[str, Any]] = field(default_factory=list)


class StatefulTool(abc.ABC):
    name: str

    @abc.abstractmethod
    def schemas(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def create(self, create_kwargs: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]: ...

    @abc.abstractmethod
    def execute(
        self, instance_id: str, parameters: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float, Mapping[str, Any]]: ...

    @abc.abstractmethod
    def calc_reward(self, instance_id: str) -> float: ...

    @abc.abstractmethod
    def release(self, instance_id: str) -> None: ...


class AgentLoopBase(abc.ABC):
    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> AgentLoopOutput: ...


class ToolAgentLoop(AgentLoopBase):
    """One policy with one long-lived tool session across all turns."""

    def __init__(
        self,
        backend: ModelBackend,
        tool: StatefulTool,
        *,
        max_assistant_turns: int = 16,
        retain_session: bool = True,
    ) -> None:
        self.backend = backend
        self.tool = tool
        self.max_assistant_turns = int(max_assistant_turns)
        self.retain_session = bool(retain_session)

    async def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        create_kwargs: Mapping[str, Any],
    ) -> AgentLoopOutput:
        started = time.monotonic()
        session_id, initial = await asyncio.to_thread(self.tool.create, create_kwargs)
        history = [dict(message) for message in messages]
        history.append(
            {
                "role": "user",
                "content": "Environment initialized:\n"
                + json.dumps(initial, sort_keys=True, default=str),
            }
        )
        metrics = AgentLoopMetrics()
        policy_logprobs: list[float] = []
        tool_rewards: list[float] = []
        events: list[dict[str, Any]] = []
        final_text = ""
        state = AgentState.PENDING
        pending_calls: list[ToolCall] = []

        try:
            while state is not AgentState.TERMINATED:
                if state is AgentState.PENDING:
                    state = AgentState.GENERATING
                    continue
                if state is AgentState.GENERATING:
                    model_started = time.monotonic()
                    turn = await asyncio.to_thread(
                        self.backend.generate, history, self.tool.schemas()
                    )
                    elapsed = time.monotonic() - model_started
                    metrics.model_seconds += elapsed
                    metrics.assistant_turns += 1
                    policy_logprobs.extend(turn.logprobs)
                    final_text = turn.text
                    history.append(turn.assistant_message)
                    events.append(
                        {
                            "type": "model",
                            "elapsed_seconds": elapsed,
                            "text": turn.text,
                            "tool_calls": [call.__dict__ for call in turn.tool_calls],
                            "logprobs": turn.logprobs,
                            "usage": turn.usage,
                        }
                    )
                    pending_calls = turn.tool_calls
                    if pending_calls:
                        state = AgentState.PROCESSING_TOOLS
                    else:
                        state = AgentState.TERMINATED
                    continue
                if state is AgentState.PROCESSING_TOOLS:
                    for call in pending_calls:
                        tool_started = time.monotonic()
                        if call.name != self.tool.name:
                            result = {
                                "ok": False,
                                "error": "unknown tool %r; expected %r"
                                % (call.name, self.tool.name),
                            }
                            reward, tool_metrics = 0.0, {"unknown_tool": call.name}
                        else:
                            result, reward, tool_metrics = await asyncio.to_thread(
                                self.tool.execute,
                                session_id,
                                call.arguments,
                            )
                        elapsed = time.monotonic() - tool_started
                        metrics.tool_seconds += elapsed
                        metrics.tool_calls += 1
                        tool_rewards.append(float(reward))
                        content = json.dumps(result, sort_keys=True, default=str)
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "name": call.name,
                                "content": content,
                            }
                        )
                        events.append(
                            {
                                "type": "tool",
                                "name": call.name,
                                "elapsed_seconds": elapsed,
                                "parameters": call.arguments,
                                "result": result,
                                "reward": reward,
                                "metrics": dict(tool_metrics),
                            }
                        )
                    state = (
                        AgentState.TERMINATED
                        if metrics.assistant_turns >= self.max_assistant_turns
                        else AgentState.GENERATING
                    )
                    continue
                raise RuntimeError("invalid agent state: %s" % state)
        finally:
            if not self.retain_session:
                await asyncio.to_thread(self.tool.release, session_id)

        metrics.total_seconds = time.monotonic() - started
        reward_score = float(self.tool.calc_reward(session_id))
        return AgentLoopOutput(
            messages=history,
            final_text=final_text,
            policy_logprobs=policy_logprobs,
            tool_rewards=tool_rewards,
            reward_score=reward_score,
            metrics=metrics,
            session_id=session_id,
            events=events,
        )


async def gather_limited(
    coroutines: Sequence[Any], limit: int
) -> list[Any]:
    """Gather work concurrently while bounding active sessions."""

    semaphore = asyncio.Semaphore(max(1, int(limit)))

    async def guarded(coroutine: Any) -> Any:
        async with semaphore:
            return await coroutine

    return list(await asyncio.gather(*(guarded(item) for item in coroutines)))

