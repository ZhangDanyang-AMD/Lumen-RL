"""``EagleMooncakeStore`` for ATOM, backed by LumenRL's implementation.

ATOM constructs the store and immediately starts calling ``put()``:

    self._mooncake_store = EagleMooncakeStore(mc_cfg)

LumenRL's store splits construction from ``setup()`` — the latter opens the
Mooncake connection, allocates the pinned host buffer pool and registers it for
RDMA. Calling ``put()`` without it raises. Rather than patch ATOM to insert a
``setup()`` call, this subclass connects on demand.

Connecting lazily rather than in ``__init__`` also keeps the non-writing ranks
free: ATOM configures extraction on every tensor-parallel rank, but LumenRL's
runner only writes from rank 0, so ranks 1..N never reach ``put()`` and never
pay for a connection or a buffer pool.
"""

from __future__ import annotations

import logging

import torch

from lumenrl.transfer.eagle_mooncake_store import (
    EagleMooncakeStore as _LumenEagleMooncakeStore,
)

logger = logging.getLogger("atom")


class EagleMooncakeStore(_LumenEagleMooncakeStore):
    """LumenRL's store that connects on first use instead of requiring ``setup()``."""

    def _ensure_setup(self) -> None:
        if self._initialized:
            return
        device = None
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.info("EagleMooncakeStore: connecting on first use (device=%s)", device)
        self.setup(device)

    def put(self, *args, **kwargs):
        self._ensure_setup()
        return super().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self._ensure_setup()
        return super().get(*args, **kwargs)

    def remove_eagle3_tensors(self, *args, **kwargs):
        self._ensure_setup()
        return super().remove_eagle3_tensors(*args, **kwargs)


__all__ = ["EagleMooncakeStore"]
