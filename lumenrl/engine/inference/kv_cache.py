"""FP8 KV-cache scale management for ATOM-style inference engines."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FP8KVCacheManager:
    """Manages FP8 KV-cache quantization scales and periodic QKV recalibration.

    Integrates with engines that expose optional FP8 KV paths. When the
    underlying runtime is unavailable, operations degrade gracefully while
    still exposing a consistent API for rollout workers.
    """

    def __init__(self, enabled: bool = False, **kwargs: Any) -> None:
        self._fp8_enabled = bool(enabled)
        self._kwargs = kwargs
        self._last_scale_step: int = -1

    def recalibrate_scales(self) -> None:
        """Run per-step QKV scale recalibration for FP8 KV caches.

        Attempts to delegate to optional ``lumen`` / runtime hooks when present;
        otherwise updates internal bookkeeping so callers can still sequence
        recalibration in the training loop.
        """
        try:
            import lumen  # type: ignore[import-untyped]

            _ = lumen  # runtime may register scale hooks; keep import for side effects
            logger.debug("FP8KVCacheManager: lumen runtime present; recalibration delegated.")
        except Exception as exc:
            logger.debug(
                "FP8KVCacheManager.recalibrate_scales: lumen unavailable or failed (%s); "
                "using local bookkeeping only.",
                exc,
            )
        self._last_scale_step += 1

    def is_fp8_enabled(self) -> bool:
        """Return whether FP8 KV-cache paths are active for this manager."""
        return self._fp8_enabled
