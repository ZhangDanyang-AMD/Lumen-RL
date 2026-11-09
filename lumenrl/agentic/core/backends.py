"""Pluggable environment backend registration.

Backends live outside LumenRL Core (e.g. ``experiments/sandbox/sandbox/micake``)
and register themselves into ``ENV_REGISTRY`` on import. The core only knows
the backend name and, optionally, the module that performs the registration.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from lumenrl.agentic.core.protocols import AgentEnvironment
from lumenrl.core.registry import ENV_REGISTRY

logger = logging.getLogger(__name__)


def register_environment(name: str, factory: Any) -> None:
    """Register an ``AgentEnvironment`` factory (class or callable) under ``name``."""
    ENV_REGISTRY.register(name, factory)


def build_environment(backend: str, module: str | None = None, **kwargs: Any) -> AgentEnvironment:
    """Instantiate a registered environment backend.

    Args:
        backend: Registered backend name.
        module: Optional module imported first so that it can self-register.
        **kwargs: Backend-specific constructor arguments.

    Returns:
        A fresh environment instance.
    """
    if backend not in ENV_REGISTRY and module:
        logger.info("importing environment module %s for backend %s", module, backend)
        importlib.import_module(module)
    env = ENV_REGISTRY.get(backend)(**kwargs)
    if not isinstance(env, AgentEnvironment):
        raise TypeError(f"backend {backend!r} produced {type(env).__name__}, not an AgentEnvironment")
    return env
