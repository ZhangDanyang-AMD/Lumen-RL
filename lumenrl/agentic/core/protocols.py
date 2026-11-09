"""Backend-neutral AgenticRL protocol: actions, observations, environments.

Nothing in this module may reference a concrete backend (MICake, GEAK, a code
sandbox, ...). Backend-specific facts travel inside ``observation`` and
``metadata`` mappings or as opaque :class:`ReceiptRef` handles.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

TERMINAL_STATES = ("RUNNING", "SUCCESS", "BUDGET_EXHAUSTED", "AGENT_FINISHED", "UNKNOWN_OUTCOME", "ERROR")


@dataclass(frozen=True)
class ToolCall:
    """One model-issued tool invocation.

    Attributes:
        call_id: Identifier assigned by the rollout backend (echoed in tool results).
        name: Tool name as exposed in the environment's tool schemas.
        arguments: Parsed JSON arguments, or ``None`` when the raw text did not parse.
        raw_arguments: Exact argument text produced by the model.
    """

    call_id: str
    name: str
    arguments: Mapping[str, Any] | None
    raw_arguments: str

    @classmethod
    def from_raw(cls, call_id: str, name: str, raw_arguments: str) -> "ToolCall":
        """Build a ToolCall, keeping unparsable argument text for fail-closed handling."""
        try:
            parsed = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and not isinstance(parsed, dict):
            parsed = None
        return cls(call_id=call_id, name=name, arguments=parsed, raw_arguments=raw_arguments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments) if self.arguments is not None else None,
            "raw_arguments": self.raw_arguments,
        }


@dataclass(frozen=True)
class ReceiptRef:
    """Opaque reference to evidence owned by an environment backend."""

    provider: str
    kind: str
    ref: str
    digest: str | None = None
    terminal_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "kind": self.kind,
            "ref": self.ref,
            "digest": self.digest,
            "terminal_state": self.terminal_state,
        }


@dataclass
class StepResult:
    """Outcome of executing one tool call inside an environment.

    Attributes:
        observation: JSON-serializable payload returned to the model as the tool result.
        reward: Scalar turn reward assigned by the environment's verifiable reward.
        done: Whether the episode must stop after this step.
        terminal_state: One of :data:`TERMINAL_STATES`.
        execution_result: Coarse stage label (backend-defined, e.g. ``"compile_fail"``).
        receipt_refs: Evidence handles produced by this step.
        metadata: Extra backend facts kept out of the model context.
    """

    observation: Mapping[str, Any]
    reward: float
    done: bool = False
    terminal_state: str = "RUNNING"
    execution_result: str = ""
    receipt_refs: list[ReceiptRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terminal_state not in TERMINAL_STATES:
            raise ValueError(f"unknown terminal_state {self.terminal_state!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": dict(self.observation),
            "reward": self.reward,
            "done": self.done,
            "terminal_state": self.terminal_state,
            "execution_result": self.execution_result,
            "receipt_refs": [ref.to_dict() for ref in self.receipt_refs],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ResetResult:
    """Initial state of an episode.

    Attributes:
        system_prompt: Instructions for the agent.
        observation: Task packet shown as the first user message.
        tools: OpenAI function-calling schemas available for this episode.
        metadata: Frozen task/environment identity for the trajectory record.
    """

    system_prompt: str
    observation: Mapping[str, Any]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentCapabilities:
    """Features an environment backend supports, negotiated by the rollout engine."""

    backend: str
    snapshot: bool = False
    async_step: bool = True
    parallel_tool_calls: bool = False
    fidelity_levels: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


class AgentEnvironment(abc.ABC):
    """Stateful multi-turn environment driven by tool calls.

    A single instance serves one episode at a time: ``reset`` starts it,
    ``step`` executes one tool call, ``close`` releases backend resources.
    """

    @abc.abstractmethod
    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        """Start a new episode for ``task`` and return the initial state."""

    @abc.abstractmethod
    async def step(self, action: ToolCall) -> StepResult:
        """Execute one tool call and return its observation and reward."""

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        """Return the terminal reward and summary once the episode has ended."""
        return 0.0, {}

    async def snapshot(self) -> str:
        """Persist the current episode state and return a reference to it."""
        raise NotImplementedError(f"{type(self).__name__} does not support snapshot")

    async def restore(self, state_ref: str) -> None:
        """Restore a state previously returned by :meth:`snapshot`."""
        raise NotImplementedError(f"{type(self).__name__} does not support restore")

    async def close(self) -> None:
        """Release backend resources. Safe to call more than once."""

    @abc.abstractmethod
    def capabilities(self) -> EnvironmentCapabilities:
        """Describe supported features."""
