"""Framework-neutral Agent, Task, Sandbox, Tool, and Environment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .models import RewardBreakdown
from .runtime import AgentLoopOutput, StatefulTool


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    source: str
    objective: str


@runtime_checkable
class Sandbox(Protocol):
    def create(
        self, case_id: str, *, role: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any]]: ...

    def fork(self, session_id: str, *, role: str) -> tuple[str, dict[str, Any]]: ...

    def observe(self, session_id: str) -> dict[str, Any]: ...

    def execute(
        self, session_id: str, parameters: Mapping[str, Any]
    ) -> tuple[dict[str, Any], RewardBreakdown, dict[str, Any]]: ...

    def release(self, session_id: str) -> None: ...


@runtime_checkable
class Agent(Protocol):
    async def run(self, *args: Any, **kwargs: Any) -> AgentLoopOutput: ...


@runtime_checkable
class Environment(Protocol):
    def case_observation(self, case_id: str) -> dict[str, Any]: ...

    def verify(
        self, session_id: str
    ) -> tuple[dict[str, Any], RewardBreakdown, dict[str, Any]]: ...


__all__ = [
    "Agent",
    "AgentLoopOutput",
    "Environment",
    "Sandbox",
    "StatefulTool",
    "Task",
]

