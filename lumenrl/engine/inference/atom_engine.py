"""ATOM-backed inference engine with lazy initialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from lumenrl.core.config import AtomConfig

logger = logging.getLogger(__name__)


@dataclass
class _AtomStubEngine:
    """Minimal placeholder engine when ATOM bindings are unavailable."""

    model_name: str

    def generate(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        return [p + [0] * max(1, max_new_tokens // 2) for p in prompts]


class AtomEngine:
    """Lazy ATOM inference engine: generation, dynamic weights, and shutdown.

    The real ATOM stack is imported only inside :meth:`init` so that installs
    without optional dependencies still import this module successfully.
    """

    def __init__(self, config: AtomConfig, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._backend: Any = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def init(self) -> None:
        """Create the underlying engine on first use (lazy)."""
        if self._initialized:
            return
        try:
            # Optional ATOM / vLLM-style stack — name is illustrative
            import atom  # type: ignore[import-untyped]

            self._backend = atom.Engine(  # type: ignore[attr-defined]
                model=self._model_name,
                tensor_parallel_size=self._config.tensor_parallel_size,
                max_model_len=self._config.max_model_len,
            )
            logger.info("AtomEngine: initialized native ATOM backend for %s.", self._model_name)
        except ImportError as exc:
            logger.warning(
                "AtomEngine: `atom` package not available (%s). "
                "Using a local stub engine; install ATOM for real inference.",
                exc,
            )
            self._backend = _AtomStubEngine(model_name=self._model_name)
        except Exception as exc:  # pragma: no cover - optional stack variability
            logger.warning(
                "AtomEngine: failed to construct ATOM backend (%s). Falling back to stub.",
                exc,
            )
            self._backend = _AtomStubEngine(model_name=self._model_name)
        self._initialized = True

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: Mapping[str, Any] | None = None,
    ) -> list[list[int]]:
        """Generate token sequences for a batch of prompt token lists."""
        self.init()
        sampling_params = dict(sampling_params or {})
        max_new_tokens = int(sampling_params.get("max_new_tokens", 32))
        temperature = float(sampling_params.get("temperature", 1.0))
        _ = temperature  # reserved for real sampler integration

        if hasattr(self._backend, "generate"):
            result = self._backend.generate(prompts, max_new_tokens=max_new_tokens)
            return result
        raise RuntimeError("AtomEngine backend has no generate() method.")

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Apply a new weight snapshot (dynamic rollout refresh)."""
        self.init()
        if hasattr(self._backend, "load_weights"):
            try:
                self._backend.load_weights(state_dict)  # type: ignore[operator]
                logger.info("AtomEngine.update_weights: loaded %d tensors.", len(state_dict))
                return
            except Exception as exc:
                logger.warning("AtomEngine.update_weights via backend failed: %s", exc)
        logger.debug(
            "AtomEngine.update_weights: stub path — received %d tensors (not applied).",
            len(state_dict),
        )

    def shutdown(self) -> None:
        """Release backend resources."""
        if hasattr(self._backend, "shutdown"):
            try:
                self._backend.shutdown()  # type: ignore[operator]
            except Exception as exc:
                logger.warning("AtomEngine.shutdown: backend shutdown failed: %s", exc)
        self._backend = None
        self._initialized = False
        logger.info("AtomEngine.shutdown: complete.")
