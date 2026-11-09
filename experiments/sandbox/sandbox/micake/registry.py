"""Register the MICake environment backend with LumenRL."""

from __future__ import annotations

from lumenrl.agentic.core.backends import register_environment
from sandbox.micake.environment import BACKEND_NAME, MicakeGemmEnvironment


def register() -> None:
    """Register ``micake_gemm`` in LumenRL's environment registry."""
    register_environment(BACKEND_NAME, MicakeGemmEnvironment)
