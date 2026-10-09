"""Sandbox: pluggable GPU kernel evaluation framework.

Provides abstract interfaces for kernel evaluation backends (compile,
correctness, performance) and multi-turn agent tool interaction.

Usage:
    from sandbox import get_backend, SandboxBackend
    backend = get_backend("geak")
    result = backend.evaluate(task_id, kernel_source, task_dir, gpu_id)
"""

from .base import (
    CommandResult,
    Environment,
    EvalResult,
    RewardBreakdown,
    SandboxBackend,
    StatefulTool,
)
from .registry import get_backend, list_backends, register_backend
from .reward import sandbox_reward_batch

# Auto-register built-in backends
try:
    from .backends import geak as _geak  # noqa: F401
except ImportError:
    pass

__all__ = [
    "CommandResult",
    "Environment",
    "EvalResult",
    "RewardBreakdown",
    "SandboxBackend",
    "StatefulTool",
    "get_backend",
    "list_backends",
    "register_backend",
    "sandbox_reward_batch",
]
