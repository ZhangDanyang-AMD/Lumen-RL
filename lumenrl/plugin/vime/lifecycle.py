"""ATOM server lifecycle management for vime integration.

Provides ``setup_atom_manager`` and ``teardown_atom_manager`` to
bootstrap and destroy the ``ATOMReplicaManager`` that the hook
functions in ``hooks.py`` depend on.

Typical usage in a vime trainer script::

    from lumenrl.plugin.vime.lifecycle import setup_atom_manager, teardown_atom_manager

    setup_atom_manager(args, actor_worker_group)
    try:
        # ... training loop ...
    finally:
        teardown_atom_manager(args)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_atom_manager(
    args: Any,
    actor_wg: Any,
    *,
    engine_kwargs: dict[str, Any] | None = None,
    max_concurrency: int = 64,
) -> Any:
    """Create an ``ATOMReplicaManager``, launch ATOM servers, and store it on ``args``.

    Args:
        args: The training args namespace. Must have ``hf_checkpoint`` for the
            model path. The manager is stored as ``args._lumenrl_atom_manager``.
        actor_wg: The Ray worker group whose GPU placement will be used for
            ATOM server colocation.
        engine_kwargs: Extra engine kwargs forwarded to ATOM. If None, a
            reasonable default is derived from ``args``.
        max_concurrency: Max concurrent Ray calls per ATOM server actor.

    Returns:
        The ``ATOMReplicaManager`` instance.
    """
    from lumenrl.engine.inference.atom_ray_server import ATOMReplicaManager

    model_path = getattr(args, "hf_checkpoint", None) or getattr(args, "model_name", "")
    if not model_path:
        raise ValueError("args must have 'hf_checkpoint' or 'model_name' for ATOM model path.")

    if engine_kwargs is None:
        engine_kwargs = _default_engine_kwargs(args)

    manager = ATOMReplicaManager(
        actor_wg=actor_wg,
        model_name=model_path,
        engine_kwargs=engine_kwargs,
        max_concurrency=max_concurrency,
        base_seed=getattr(args, "seed", None),
    )
    manager.create()

    args._lumenrl_atom_manager = manager
    logger.info(
        "lumenrl: ATOMReplicaManager created with %d replicas, stored on args._lumenrl_atom_manager",
        manager.num_replicas,
    )
    return manager


def teardown_atom_manager(args: Any) -> None:
    """Shut down the ``ATOMReplicaManager`` and remove it from ``args``."""
    manager = getattr(args, "_lumenrl_atom_manager", None)
    if manager is None:
        return

    manager.shutdown()
    args._lumenrl_atom_manager = None
    logger.info("lumenrl: ATOMReplicaManager shut down.")


def _default_engine_kwargs(args: Any) -> dict[str, Any]:
    """Build ATOM engine kwargs from vime training args."""
    kwargs: dict[str, Any] = {}

    tp = getattr(args, "tensor_model_parallel_size", 1)
    kwargs["tensor_parallel_size"] = int(tp)

    if hasattr(args, "dtype"):
        kwargs["dtype"] = str(args.dtype)

    kwargs["enforce_eager"] = getattr(args, "enforce_eager", True)

    mem_util = getattr(args, "gpu_memory_utilization", None)
    if mem_util is not None:
        kwargs["gpu_memory_utilization"] = float(mem_util)

    max_model_len = getattr(args, "rollout_max_prompt_len", 0) + getattr(args, "rollout_max_response_len", 0)
    if max_model_len > 0:
        kwargs["max_model_len"] = max_model_len

    max_seqs = getattr(args, "max_num_seqs", 256)
    kwargs["max_num_seqs"] = int(max_seqs)

    kwargs["enable_chunked_prefill"] = getattr(args, "enable_chunked_prefill", True)

    return kwargs
