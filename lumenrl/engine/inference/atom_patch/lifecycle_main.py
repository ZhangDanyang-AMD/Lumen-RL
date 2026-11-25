# SPDX-License-Identifier: MIT
"""Explicit, default-off RL lifecycle hooks for the local ATOM main interfaces.

Call ``install_lifecycle_main()`` *inside each EngineCore process*, after ATOM
has imported its classes and before accepting RL utility commands. Call
``install_lifecycle_main_worker()`` independently inside each ModelRunner
process before its worker RPC loop. Neither the decode import hook nor an
EngineCore-side install propagates to spawned workers.

Set ``LUMEN_ATOM_MAIN_LIFECYCLE=1`` to opt in. The installer fails closed on
unknown ATOM interfaces; it never changes the checkout or starts a GPU job.
Only drained, non-PP KV replacement is supported. If weights/actor residency
reduce the available paged pool, wake adopts that smaller verified plan and
rebuilds CPU scheduler block IDs before serving resumes.
"""

from __future__ import annotations

import functools
import logging
import os
from copy import copy

logger = logging.getLogger(__name__)
_TRUE = frozenset(("1", "true", "yes", "on"))
_FLAG = "LUMEN_ATOM_MAIN_LIFECYCLE"


def _enabled() -> bool:
    return os.environ.get(_FLAG, "0").strip().lower() in _TRUE


def _check_scheduler(scheduler) -> None:
    """Never discard a live request, state lease, or offload transfer."""
    if scheduler is None or getattr(scheduler, "block_manager", None) is None:
        raise RuntimeError("KV reset requires a scheduler owning its KV pool")
    if getattr(scheduler.config, "pipeline_parallel_size", 1) != 1:
        raise RuntimeError("KV reset does not support pipeline parallelism")
    from atom.model_engine.scheduler import Scheduler

    if type(scheduler) is not Scheduler:
        raise TypeError(
            "KV reset supports only the plain Scheduler, not P/D subclasses"
        )
    if getattr(scheduler.config, "kv_transfer_config", None) or getattr(
        scheduler.config, "disagg_is_decode", False
    ):
        raise RuntimeError("KV reset does not support configured KV connectors")
    if (
        scheduler.running
        or scheduler.deferred_free_blocks
        or scheduler._pp_inflight_token_block
        or scheduler._partial_prefill_count
        or scheduler._num_parked_remote_kv
        or scheduler.finished_recving_kv_req_ids
        or scheduler.failed_recving_kv_req_ids
        or getattr(scheduler, "prefill_waiting", ())
        or getattr(scheduler, "_rejected", ())
    ):
        raise RuntimeError("KV reset requires drained running/transfer state")
    # Main's get_kvconnector returns None if unconfigured. A configured
    # backend can retain the original BlockManager and in-flight remote IDs;
    # this installer cannot safely rebind every implementation.
    if scheduler.kv_connector is not None:
        raise RuntimeError("KV reset does not support configured KV connectors")
    from atom.model_engine.sequence import SequenceStatus

    for seq in scheduler.waiting:
        if (
            seq.block_table
            or seq.num_cached_tokens
            or seq.status != SequenceStatus.WAITING
            or (seq.has_per_req_cache and seq.state_slot >= 0)
        ):
            raise RuntimeError("KV reset cannot discard waiting request state")
    bm = scheduler.block_manager
    if (
        bm.kv.num_used
        or bm.has_pending_state_store_pins()
        or bm._state_loads
        or bm._orphan_load_slots
        or bm.state.num_free() != bm.num_state_slots
        or bm.state._pinned
        or bm.state._relocations
    ):
        raise RuntimeError("KV reset requires drained block/state leases")


def _reset_scheduler(scheduler, num_blocks: int) -> None:
    """Invalidate CPU KV IDs and prefix indices after a GPU pool is cleared."""
    if type(num_blocks) is not int or num_blocks <= 0:
        raise RuntimeError(f"Invalid real KV cache capacity: {num_blocks!r}")
    _check_scheduler(scheduler)
    from atom.model_engine.block_manager import BlockManager

    current = scheduler.block_manager
    # Main's Scheduler does not retain this constructor argument. The
    # immutable transfer/checkpoint spec is retained by our constructor hook;
    # never replace it silently with DEFAULT_STATE_RUNTIME.
    state_runtime = getattr(scheduler, "_lumen_kv_state_runtime", None)
    if state_runtime is None:
        raise RuntimeError(
            "KV reset needs the original StateRuntime; install before Scheduler construction"
        )
    config = copy(scheduler.config)
    config.num_kvcache_blocks = num_blocks
    # Build the replacement before changing the live scheduler or config.
    replacement = BlockManager(config, state_runtime=state_runtime)
    replacement.draft_lookahead = current.draft_lookahead
    scheduler.config.num_kvcache_blocks = num_blocks
    scheduler.block_manager = replacement
    scheduler.engine_stats._pool_pressure = replacement.pool_pressure
    if scheduler.config.enable_prefix_caching:
        scheduler.engine_stats.block_manager = replacement


def _require_rpc_reply(reply, command: str) -> None:
    if reply.get("error") or reply.get("result") is not True:
        raise RuntimeError(f"{command} failed: {reply!r}")


def _patch_engine_handler(cls) -> None:
    if getattr(cls, "_lumen_lifecycle_main_installed", False):
        return
    original_release = cls._handle_release_memory
    original_resume = cls._handle_resume_memory
    original_clear = cls._handle_clear_kv_cache

    @functools.wraps(original_release)
    def release(self, args):
        tags = args.get("tags", ["weights", "kv_cache"])
        if tags is None or "kv_cache" in tags:
            _check_scheduler(self.scheduler)
        return original_release(self, args)

    @functools.wraps(original_resume)
    def resume(self, args):
        tags = args.get("tags", ["weights", "kv_cache"])
        if tags is None or "kv_cache" not in tags:
            return original_resume(self, args)
        _check_scheduler(self.scheduler)
        # Main's _utility_on_every_rank handles all worker replies and sends
        # no response until success. Replace only the KV branch, retaining the
        # original all-rank RPC mechanism and _ReplyQueue request stamping.
        reply = self._utility_on_every_rank(
            "resume_memory",
            "resume_memory",
            tags,
            timeout=_timeout(args),
        )
        _require_rpc_reply(reply, "resume_memory")
        capacity = self._utility_on_every_rank(
            "get_kv_cache_capacity", "get_kv_cache_capacity", timeout=_timeout(args)
        )
        if capacity.get("error"):
            raise RuntimeError(f"get_kv_cache_capacity failed: {capacity['error']}")
        blocks = capacity.get("result")
        _reset_scheduler(self.scheduler, blocks)
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    @functools.wraps(original_clear)
    def clear(self, args):
        _check_scheduler(self.scheduler)
        reply = self._utility_on_every_rank(
            "clear_kv_cache", "clear_kv_cache", timeout=_timeout(args)
        )
        _require_rpc_reply(reply, "clear_kv_cache")
        _reset_scheduler(self.scheduler, self.scheduler.config.num_kvcache_blocks)
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def get_refit_receipts(self, args):
        reply = self._utility_on_every_rank(
            "get_lumen_refit_receipts", "get_lumen_refit_receipt",
            bool(args.get("expect_committed", True)),
            args.get("audit_name"), timeout=_timeout(args),
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def get_kv_capacity_receipt(self, args):
        reply = self._utility_on_every_rank(
            "get_kv_cache_capacity",
            "get_kv_cache_capacity",
            timeout=_timeout(args),
        )
        if reply.get("error"):
            raise RuntimeError(f"get_kv_cache_capacity failed: {reply['error']}")
        blocks = reply.get("result")
        if type(blocks) is not int or blocks < 0:
            raise RuntimeError(f"Invalid KV capacity receipt: {blocks!r}")
        response = {"cmd": "get_lumen_kv_cache_capacity", "result": blocks}
        self.output_queue.put_nowait(("UTILITY_RESPONSE", response))
        return response

    handlers = dict(cls._UTILITY_HANDLERS)
    handlers["get_lumen_refit_receipts"] = "_handle_get_lumen_refit_receipts"
    handlers["get_lumen_kv_cache_capacity"] = "_handle_get_lumen_kv_cache_capacity"
    cls._UTILITY_HANDLERS = handlers
    cls._handle_get_lumen_refit_receipts = get_refit_receipts
    cls._handle_get_lumen_kv_cache_capacity = get_kv_capacity_receipt
    cls._handle_release_memory = release
    cls._handle_resume_memory = resume
    cls._handle_clear_kv_cache = clear
    cls._lumen_lifecycle_main_installed = True


def _timeout(args) -> float:
    from atom.model_engine.engine_utility import _remaining_timeout

    return _remaining_timeout(args, 300.0)


def install_lifecycle_main() -> bool:
    """Install engine-side hooks; obtain StateRuntime from each EngineCore.

    Returns False with no changes if the opt-in flag is off. Raise before
    patching on a missing or changed main interface. Call once per process.
    """
    if not _enabled():
        return False
    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.scheduler import Scheduler

    required = (
        "_handle_release_memory",
        "_handle_resume_memory",
        "_handle_clear_kv_cache",
        "_utility_on_every_rank",
    )
    for name in required:
        if not callable(getattr(EngineUtilityHandler, name, None)):
            raise TypeError(f"ATOM main EngineUtilityHandler lacks {name}")
    if not callable(getattr(Scheduler, "__init__", None)):
        raise TypeError("ATOM main Scheduler unavailable")
    import inspect

    if "state_runtime" not in inspect.signature(Scheduler.__init__).parameters:
        raise RuntimeError("ATOM main Scheduler does not accept state_runtime")
    from atom.model_engine.block_manager import BlockManager

    if not callable(getattr(BlockManager, "__init__", None)):
        raise TypeError("ATOM main BlockManager unavailable")
    # EngineCore constructs the handler after the scheduler. Capture the exact
    # StateRuntime used by the engine; never substitute the default.
    if getattr(EngineUtilityHandler, "_lumen_lifecycle_main_installed", False):
        return True
    # Main's Scheduler discards state_runtime; intercept its constructor so
    # each scheduler retains exactly its own (EngineCore.state_runtime) value.
    original_scheduler_init = Scheduler.__init__

    @functools.wraps(original_scheduler_init)
    def scheduler_init(self, *args, **kwargs):
        from atom.model_engine.state_runtime import DEFAULT_STATE_RUNTIME

        original_scheduler_init(self, *args, **kwargs)
        self._lumen_kv_state_runtime = kwargs.get(
            "state_runtime", DEFAULT_STATE_RUNTIME
        )

    Scheduler.__init__ = scheduler_init
    _patch_engine_handler(EngineUtilityHandler)
    return True


def _check_weight_sync_recipe(runner, names=None) -> None:
    """Reject unsupported expert storage before a relevant bucket writes bytes.

    Main's updater handles BF16 FusedMoE layout restoration itself. Quantized
    expert weights and EP replicas are unsupported there. Only inspect expert
    storage when a bucket actually carries expert parameters: unrelated updates
    must retain main's behavior on an inference-only quantized MoE.
    """
    if names is not None and not any(".experts." in name for name in names):
        return
    model = runner._sync_target_model()
    for module in model.modules():
        if not any(hasattr(module, name) for name in ("w13_weight", "w2_weight")):
            continue
        if getattr(module, "expert_map", None) is not None or getattr(
            module, "num_redundant_experts", 0
        ):
            raise NotImplementedError(
                "RL weight sync does not support expert parallelism"
            )
        for name in ("w13_weight", "w2_weight"):
            param = getattr(module, name, None)
            if param is not None and not (
                param.dtype.is_floating_point and param.element_size() >= 2
            ):
                raise NotImplementedError(
                    "RL weight sync does not support quantized FusedMoE experts"
                )


def install_weight_sync_guard() -> bool:
    """Opt-in pre-write guard for ATOM main direct, SHM and IPC weight sync.

    Main's WeightUpdaterMixin already contains reader fences, FP8 weight+scale
    writes, packed-shard accumulation, expert remapping and relayout. Do not
    shadow those coupled operations with a stale updater implementation.
    """
    if not _enabled():
        return False
    from atom.rollout.weight_updater import WeightUpdaterMixin

    if getattr(WeightUpdaterMixin, "_lumen_weight_sync_guard_installed", False):
        return True
    method_names = (
        "update_weights",
        "update_weights_from_shm",
        "update_weights_from_ipc",
    )
    for name in method_names:
        if not callable(getattr(WeightUpdaterMixin, name, None)):
            raise TypeError(f"ATOM main WeightUpdaterMixin lacks {name}")
    if not callable(getattr(WeightUpdaterMixin, "_sync_target_model", None)):
        raise TypeError("ATOM main WeightUpdaterMixin lacks _sync_target_model")
    for name in method_names:
        original = getattr(WeightUpdaterMixin, name)

        @functools.wraps(original)
        def guarded(self, *args, _original=original, _method=name, **kwargs):
            if _method == "update_weights":
                named = args[0] if args else kwargs.get("named_tensors", ())
                names = [item[0] for item in named]
            else:
                metadata = args[1] if len(args) > 1 else kwargs.get("bucket_meta", {})
                names = metadata.keys()
            _check_weight_sync_recipe(self, names)
            return _original(self, *args, **kwargs)

        setattr(WeightUpdaterMixin, name, guarded)
    WeightUpdaterMixin._lumen_weight_sync_guard_installed = True
    return True


def _worker_capacity(self) -> int:
    """Report the allocated block count on all TP and PCP workers."""
    import torch

    value = self.config.num_kvcache_blocks
    if type(value) is not int or value <= 0 or getattr(self, "kv_cache", None) is None:
        value = 0
    if torch.distributed.is_initialized():
        from aiter.dist.parallel_state import get_pcp_group, get_tp_group

        count = torch.tensor([value], dtype=torch.int64, device="cpu")
        for group in (get_tp_group(), get_pcp_group()):
            torch.distributed.all_reduce(
                count, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
            )
        value = int(count.item())
    return value


def _worker_resume_kv(self) -> None:
    import torch
    from atom.rollout.memory_manager import sleep_keeps_memory_resident

    if (
        sleep_keeps_memory_resident(self)
        and getattr(self, "kv_cache", None) is not None
    ):
        return
    if getattr(self.config, "pipeline_parallel_size", 1) != 1:
        raise RuntimeError("KV wake does not support pipeline parallelism")
    if getattr(self.config, "kv_transfer_config", None) or getattr(
        self.config, "disagg_is_decode", False
    ):
        raise RuntimeError("KV wake does not support configured KV connectors")
    saved = getattr(self, "_kv_cache_num_blocks", None)
    if type(saved) is not int or saved <= 0:
        if getattr(self, "kv_cache", None) is None:
            raise RuntimeError("KV wake has no saved pool capacity")
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    info = self.get_num_blocks()
    available = info["num_kvcache_blocks"]
    if type(available) is not int or available <= 0:
        raise RuntimeError(f"Invalid KV wake capacity: {available!r}")
    plan = self.pool_plan
    if getattr(plan, "paged_entries", None) != available or not callable(
        getattr(plan, "with_paged_entries", None)
    ):
        raise RuntimeError("KV wake pool plan does not match runner probe")
    previous = getattr(self, "_lumen_sleep_pool_entries", None)
    current_slots = {
        k: v for k, v in info["pool_entries"].items() if k != plan.paged_class
    }
    if previous is None or current_slots != previous:
        raise RuntimeError("KV wake cannot change non-paged state slot capacities")
    original_per_req = getattr(self, "_lumen_sleep_pool_per_req", None)
    if original_per_req is None or info.get("pool_entries_per_req") != original_per_req:
        raise RuntimeError("KV wake cannot change per-request state slot geometry")
    original_runtime = getattr(self, "_lumen_sleep_state_runtime_wire", None)
    if original_runtime is None or info.get("state_runtime") != original_runtime:
        raise RuntimeError("KV wake cannot change StateRuntime transfer geometry")
    # get_num_blocks publishes a fresh plan. Only its PAGE class may change;
    # retain the older count when possible, otherwise adopt the probed smaller
    # pool and let EngineCore rebuild the drained scheduler at that exact size.
    original_plan = getattr(self, "_lumen_sleep_pool_plan", None)
    if (
        original_plan is None
        or plan.paged_class != original_plan.paged_class
        or original_plan.paged_entries != saved
        or plan.entry_bytes != original_plan.entry_bytes
    ):
        raise RuntimeError("KV wake changed the original pool geometry")
    target = min(saved, available)
    selected_plan = original_plan if target == saved else plan
    if target < saved:
        logger.warning("KV wake adopts drained paged capacity %d -> %d blocks", saved, target)
    self.pool_plan = selected_plan
    self.config.pool_entries = dict(selected_plan.entries)
    self.config.pool_entries_per_req = dict(selected_plan.entries_per_req)
    self.config.num_kvcache_blocks = target
    # A failed allocation remains fenced and keeps the saved count for retry.
    self.allocate_kv_cache(target)
    self._kv_cache_num_blocks = None
    self._lumen_sleep_pool_entries = None
    self._lumen_sleep_pool_per_req = None
    self._lumen_sleep_state_runtime_wire = None
    self._lumen_sleep_pool_plan = None


def install_lifecycle_main_worker() -> bool:
    """Install worker-side capacity/restore hooks in each spawned ModelRunner."""
    if not _enabled():
        return False
    from atom.rollout.memory_manager import MemoryManagerMixin

    if getattr(MemoryManagerMixin, "_lumen_lifecycle_main_worker_installed", False):
        return True
    for name in ("_release_kv_cache", "_resume_kv_cache"):
        if not callable(getattr(MemoryManagerMixin, name, None)):
            raise TypeError(f"ATOM main MemoryManagerMixin lacks {name}")
    from atom.model_engine.model_runner import ModelRunner

    for name in ("get_num_blocks", "allocate_kv_cache"):
        if not callable(getattr(ModelRunner, name, None)):
            raise TypeError(f"ATOM main ModelRunner lacks {name}")
    release = MemoryManagerMixin._release_kv_cache

    @functools.wraps(release)
    def release_kv(self):
        from atom.rollout.memory_manager import sleep_keeps_memory_resident

        if (
            not sleep_keeps_memory_resident(self)
            and getattr(self, "kv_cache", None) is not None
        ):
            if getattr(self.config, "pipeline_parallel_size", 1) != 1:
                raise RuntimeError("KV release does not support pipeline parallelism")
            if getattr(self.config, "kv_transfer_config", None) or getattr(
                self.config, "disagg_is_decode", False
            ):
                raise RuntimeError(
                    "KV release does not support configured KV connectors"
                )
            plan = getattr(self, "pool_plan", None)
            if plan is None or getattr(plan, "paged_class", None) is None:
                raise RuntimeError("KV release needs a paged pool plan")
            self._lumen_sleep_pool_entries = {
                k: v
                for k, v in self.config.pool_entries.items()
                if k != plan.paged_class
            }
            self._lumen_sleep_pool_per_req = dict(plan.entries_per_req)
            runtime = getattr(self, "state_runtime", None)
            if runtime is None or not callable(getattr(runtime, "to_wire", None)):
                raise RuntimeError("KV release requires the original StateRuntime")
            self._lumen_sleep_state_runtime_wire = runtime.to_wire()
            self._lumen_sleep_pool_plan = plan
        return release(self)

    install_weight_sync_guard()
    MemoryManagerMixin._release_kv_cache = release_kv
    MemoryManagerMixin._resume_kv_cache = _worker_resume_kv
    MemoryManagerMixin.get_kv_cache_capacity = _worker_capacity
    MemoryManagerMixin._lumen_lifecycle_main_worker_installed = True
    return True
