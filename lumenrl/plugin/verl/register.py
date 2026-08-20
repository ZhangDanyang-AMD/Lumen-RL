"""verl plugin entry-point.

Called automatically by verl when this package is declared under
[project.entry-points."verl.plugins"] in pyproject.toml.

Registers:
  - ATOM rollout backend ("atom") in verl's _ROLLOUT_REGISTRY
  - LumenRL FSDP2 training engine ("lumenrl_fsdp2") in verl's EngineRegistry
  - LumenRL Megatron training engine ("lumenrl_megatron") in verl's EngineRegistry
"""

from __future__ import annotations

import logging
from typing import Any, Generator

logger = logging.getLogger(__name__)

_registered = False


# ---------------------------------------------------------------------------
# ATOMRolloutAdapter — verl BaseRollout implementation backed by ATOM
# ---------------------------------------------------------------------------

class ATOMRolloutAdapter:
    """Thin verl ``BaseRollout`` adapter around ``ATOMReplicaManager``.

    Weight sync is NOT implemented here — ATOM manages its own weight
    lifecycle via ``sleep`` / ``wake_up`` / ``load_weights``.
    ``update_weights`` is intentionally a no-op.

    The adapter is instantiated by verl's engine worker via the FQDN string
    stored in ``_ROLLOUT_REGISTRY``. verl resolves it with ``importlib`` and
    calls ``cls(config, model_config, device_mesh)``.
    """

    def __init__(self, config: Any, model_config: Any, device_mesh: Any, **kw: Any):
        try:
            from verl.workers.rollout.base import BaseRollout

            BaseRollout.__init__(self, config, model_config, device_mesh)
        except ImportError:
            pass
        self._manager = None

    def set_manager(self, manager: Any) -> None:
        """Inject the ``ATOMReplicaManager`` created by the trainer."""
        self._manager = manager

    async def resume(self, tags: list[str]) -> None:
        if self._manager is not None:
            self._manager.wake_all(tags=tags)

    async def update_weights(
        self,
        weights: Generator[tuple[str, Any], None, None],
        wire_format: str = "named_tensors",
        **kw: Any,
    ) -> None:
        logger.debug("ATOMRolloutAdapter.update_weights: no-op (ATOM-native sync).")

    async def release(self) -> None:
        if self._manager is not None:
            self._manager.sleep_all()

    def generate_sequences(self, prompts: Any) -> Any:
        raise NotImplementedError(
            "ATOMRolloutAdapter.generate_sequences is not supported; "
            "use async mode with verl."
        )


# ---------------------------------------------------------------------------
# LumenRLFSDP2Engine — verl BaseEngine shim for FSDP2 training
# ---------------------------------------------------------------------------

class LumenRLFSDP2Engine:
    """Delegates to ``lumenrl.engine.training.fsdp_engine.FSDP2EngineWithLMHead``.

    Supports Lumen FP8 training via the ``quant_config`` constructor parameter.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        from lumenrl.engine.training.fsdp_engine import FSDP2EngineWithLMHead

        self._inner = FSDP2EngineWithLMHead(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._inner.is_param_offload_enabled

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._inner.is_optimizer_offload_enabled


# ---------------------------------------------------------------------------
# LumenRLMegatronEngine — verl BaseEngine shim delegating to LumenRL
# ---------------------------------------------------------------------------

class LumenRLMegatronEngine:
    """Delegates to ``lumenrl.engine.training.megatron_engine.MegatronEngine``.

    Only the two abstract properties required by verl's ``BaseEngine`` are
    explicitly forwarded; everything else goes through ``__getattr__``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        from lumenrl.engine.training.megatron_engine import MegatronEngineWithLMHead

        self._inner = MegatronEngineWithLMHead(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._inner.is_param_offload_enabled

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._inner.is_optimizer_offload_enabled


# ---------------------------------------------------------------------------
# Entry-point callable
# ---------------------------------------------------------------------------

def register() -> None:
    """Register LumenRL backends in verl's registries.

    Called by verl's plugin auto-loader at ``import verl`` time.
    """
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from verl.workers.rollout.base import _ROLLOUT_REGISTRY

        _ROLLOUT_REGISTRY[("atom", "async")] = (
            "lumenrl.plugin.verl.register.ATOMRolloutAdapter"
        )
        logger.info("lumenrl: registered ATOM rollout in verl._ROLLOUT_REGISTRY")
    except ImportError:
        logger.warning("lumenrl: verl.workers.rollout.base not importable; skipped rollout registration")

    try:
        from lumenrl.engine.training.base_engine import BaseEngine, EngineRegistry

        # Make adapter classes formal BaseEngine subclasses at runtime
        # so that EngineRegistry.register()'s issubclass check passes.
        LumenRLFSDP2Engine.__bases__ = (BaseEngine,)
        LumenRLMegatronEngine.__bases__ = (BaseEngine,)

        EngineRegistry.register(
            model_type="language_model",
            backend=["lumenrl_fsdp", "lumenrl_fsdp2"],
            device=["cuda"],
        )(LumenRLFSDP2Engine)
        logger.info("lumenrl: registered lumenrl_fsdp/lumenrl_fsdp2 in EngineRegistry")

        EngineRegistry.register(
            model_type="language_model",
            backend="lumenrl_megatron",
            device=["cuda"],
        )(LumenRLMegatronEngine)
        logger.info("lumenrl: registered lumenrl_megatron in EngineRegistry")
    except ImportError:
        logger.warning("lumenrl: EngineRegistry not importable; skipped engine registration")
