"""Sandbox: pluggable GPU kernel evaluation backends.

Two kinds of backends live here:

* ``SandboxBackend`` (``sandbox.base``): single-shot compile -> correctness ->
  performance evaluation of a kernel source, used by GRPO reward functions.
  Usage::

      from sandbox import get_backend
      backend = get_backend("geak")
      result = backend.evaluate(task_id, kernel_source, task_dir, gpu_id)

* Multi-turn ``lumenrl.agentic.core.AgentEnvironment`` plugins (e.g.
  ``sandbox.micake``), which register into LumenRL's environment registry on
  import and are built with ``lumenrl.agentic.core.build_environment``.
"""

from .base import (
    CommandResult,
    EvalResult,
    RewardBreakdown,
    SandboxBackend,
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
    "EvalResult",
    "RewardBreakdown",
    "SandboxBackend",
    "get_backend",
    "list_backends",
    "register_backend",
    "sandbox_reward_batch",
]
