"""ATOM server lifecycle management for verl integration.

Provides ``setup_atom_manager`` and ``teardown_atom_manager`` to
bootstrap and destroy the ``ATOMReplicaManager`` that the
``ATOMRolloutAdapter`` depends on.

Typical usage in a verl trainer script::

    from lumenrl.plugin.verl.lifecycle import setup_atom_manager, teardown_atom_manager

    manager = setup_atom_manager(rollout_adapter, actor_worker_group, model_path, engine_kwargs)
    try:
        # ... training loop ...
    finally:
        teardown_atom_manager(rollout_adapter)

For verl's ``ATOMRolloutReplica`` (async server mode), the ATOM servers
are managed by the replica itself — no separate lifecycle setup is needed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_atom_manager(
    rollout_adapter: Any,
    actor_wg: Any,
    model_path: str,
    engine_kwargs: dict[str, Any],
    *,
    max_concurrency: int = 64,
    base_seed: int | None = None,
) -> Any:
    """Create an ``ATOMReplicaManager``, launch ATOM servers, and inject into adapter.

    Args:
        rollout_adapter: The ``ATOMRolloutAdapter`` instance (verl's BaseRollout).
        actor_wg: The Ray worker group whose GPU placement will be used for
            ATOM server colocation.
        model_path: HF model path for ATOM to load.
        engine_kwargs: ATOM engine configuration kwargs.
        max_concurrency: Max concurrent Ray calls per ATOM server actor.
        base_seed: Random seed for reproducibility.

    Returns:
        The ``ATOMReplicaManager`` instance.
    """
    from lumenrl.engine.inference.atom_ray_server import ATOMReplicaManager

    manager = ATOMReplicaManager(
        actor_wg=actor_wg,
        model_name=model_path,
        engine_kwargs=engine_kwargs,
        max_concurrency=max_concurrency,
        base_seed=base_seed,
    )
    manager.create()

    rollout_adapter.set_manager(manager)
    logger.info(
        "lumenrl: ATOMReplicaManager created with %d replicas, injected into ATOMRolloutAdapter.",
        manager.num_replicas,
    )
    return manager


def teardown_atom_manager(rollout_adapter: Any) -> None:
    """Shut down the ``ATOMReplicaManager`` and detach from the adapter."""
    manager = getattr(rollout_adapter, "_manager", None)
    if manager is None:
        return

    manager.shutdown()
    rollout_adapter._manager = None
    logger.info("lumenrl: ATOMReplicaManager shut down.")
