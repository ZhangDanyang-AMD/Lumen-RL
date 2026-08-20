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

def _make_engine_classes():
    """Lazily create engine adapter classes that inherit from verl's BaseEngine.

    Deferred to avoid importing verl at module-level (it may not be installed).
    """
    from verl.workers.engine.base import BaseEngine as VerlBaseEngine

    # Collect all methods from BaseEngine that raise NotImplementedError
    # and generate forwarding methods to self._inner.
    _delegate_methods = []
    import inspect as _inspect
    for _name, _method in _inspect.getmembers(VerlBaseEngine, predicate=_inspect.isfunction):
        if _name.startswith("_"):
            continue
        try:
            src = _inspect.getsource(_method)
            if "raise NotImplementedError" in src:
                _delegate_methods.append(_name)
        except (OSError, TypeError):
            pass

    class _LumenRLFSDP2Engine(VerlBaseEngine):
        """Delegates to ``lumenrl.engine.training.fsdp_engine.FSDP2EngineWithLMHead``."""

        @staticmethod
        def _convert_verl_config(**kwargs: Any) -> dict:
            """Convert verl config dicts to LumenRL-compatible dicts."""
            from lumenrl.core.config import HFModelConfig, LoRAConfig, FSDPEngineConfig, OptimizerConfig
            import dataclasses

            def _instantiate(src: Any, dc_cls: type) -> Any:
                """Instantiate a LumenRL dataclass from a verl config dict/object."""
                if isinstance(src, dc_cls):
                    return src
                raw = dict(src) if hasattr(src, "items") else {}
                valid = {f.name for f in dataclasses.fields(dc_cls)}
                filtered = {}
                for k, v in raw.items():
                    if k in valid:
                        f_type = {f.name: f.type for f in dataclasses.fields(dc_cls)}.get(k)
                        if dataclasses.is_dataclass(f_type) and hasattr(v, "items"):
                            filtered[k] = _instantiate(v, f_type)
                        else:
                            filtered[k] = v
                return dc_cls(**filtered)

            result: dict = {}

            if "model_config" in kwargs:
                mc = dict(kwargs["model_config"]) if hasattr(kwargs["model_config"], "items") else {}
                if "path" in mc and "local_path" not in mc:
                    mc["local_path"] = mc.pop("path")
                if "lora" in mc and not isinstance(mc["lora"], LoRAConfig):
                    mc["lora"] = _instantiate(mc["lora"], LoRAConfig)
                result["model_config"] = _instantiate(mc, HFModelConfig)

            if "engine_config" in kwargs:
                result["engine_config"] = _instantiate(kwargs["engine_config"], FSDPEngineConfig)

            if "optimizer_config" in kwargs:
                result["optimizer_config"] = _instantiate(kwargs["optimizer_config"], OptimizerConfig)

            for k in ("model_name", "quant_config"):
                if k in kwargs:
                    result[k] = kwargs[k]

            return result

        def __init__(self, *args: Any, **kwargs: Any):
            from lumenrl.engine.training.fsdp_engine import FSDP2EngineWithLMHead
            converted = self._convert_verl_config(**kwargs)
            self._inner = FSDP2EngineWithLMHead(*args, **converted)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def is_param_offload_enabled(self) -> bool:
            return self._inner.is_param_offload_enabled

        @property
        def is_optimizer_offload_enabled(self) -> bool:
            return self._inner.is_optimizer_offload_enabled

    # Dynamically add forwarding methods for all BaseEngine abstract methods
    for _mname in _delegate_methods:
        if _mname not in _LumenRLFSDP2Engine.__dict__:
            def _make_forwarder(name):
                def _forwarder(self, *a, **kw):
                    return getattr(self._inner, name)(*a, **kw)
                _forwarder.__name__ = name
                return _forwarder
            setattr(_LumenRLFSDP2Engine, _mname, _make_forwarder(_mname))

    class _LumenRLMegatronEngine(VerlBaseEngine):
        """Delegates to ``lumenrl.engine.training.megatron_engine.MegatronEngine``."""

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

    for _mname in _delegate_methods:
        if _mname not in _LumenRLMegatronEngine.__dict__:
            def _make_forwarder(name):
                def _forwarder(self, *a, **kw):
                    return getattr(self._inner, name)(*a, **kw)
                _forwarder.__name__ = name
                return _forwarder
            setattr(_LumenRLMegatronEngine, _mname, _make_forwarder(_mname))

    return _LumenRLFSDP2Engine, _LumenRLMegatronEngine


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

    # Also register in RolloutReplicaRegistry (used by v0 legacy trainer)
    try:
        from verl.workers.rollout.replica import RolloutReplicaRegistry

        def _load_atom():
            from lumenrl.plugin.verl.atom_replica import ATOMRolloutReplica
            return ATOMRolloutReplica

        if "atom" not in RolloutReplicaRegistry._registry:
            RolloutReplicaRegistry.register("atom", _load_atom)
            logger.info("lumenrl: registered ATOM in RolloutReplicaRegistry (v0 compat)")
    except ImportError:
        pass

    try:
        from verl.workers.engine.base import EngineRegistry as VerlEngineRegistry

        FSDP2Cls, MegatronCls = _make_engine_classes()

        VerlEngineRegistry.register(
            model_type="language_model",
            backend=["lumenrl_fsdp", "lumenrl_fsdp2"],
            device=["cuda"],
        )(FSDP2Cls)
        logger.info("lumenrl: registered lumenrl_fsdp/lumenrl_fsdp2 in verl EngineRegistry")

        VerlEngineRegistry.register(
            model_type="language_model",
            backend="lumenrl_megatron",
            device=["cuda"],
        )(MegatronCls)
        logger.info("lumenrl: registered lumenrl_megatron in verl EngineRegistry")
    except Exception as e:
        logger.warning("lumenrl: engine registration failed: %s", e)


# Auto-register when this module is imported (verl's plugin loader calls
# _ep.load() which imports this module but doesn't call register()).
register()
