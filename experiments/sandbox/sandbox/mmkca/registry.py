"""Register the MMKCA environment backend with LumenRL."""

from __future__ import annotations

from lumenrl.agentic.core.backends import register_environment
from sandbox.mmkca.environment import BACKEND_NAME, MmkcaGemmEnvironment


def register() -> None:
    """Register ``mmkca_gemm`` in LumenRL's environment registry."""
    register_environment(BACKEND_NAME, MmkcaGemmEnvironment)
