"""Backend-neutral AgenticRL core protocol and trajectory schema."""

from lumenrl.agentic.core.backends import build_environment, register_environment
from lumenrl.agentic.core.protocols import (
    TERMINAL_STATES,
    AgentEnvironment,
    EnvironmentCapabilities,
    ReceiptRef,
    ResetResult,
    StepResult,
    ToolCall,
)
from lumenrl.agentic.core.trajectory import Episode, PolicyTurn, Step, TrajectoryStore

__all__ = [
    "TERMINAL_STATES",
    "AgentEnvironment",
    "EnvironmentCapabilities",
    "Episode",
    "PolicyTurn",
    "ReceiptRef",
    "ResetResult",
    "Step",
    "StepResult",
    "ToolCall",
    "TrajectoryStore",
    "build_environment",
    "register_environment",
]
