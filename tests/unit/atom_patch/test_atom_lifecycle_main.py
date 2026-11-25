# SPDX-License-Identifier: MIT
"""CPU-only regression tests for opt-in ATOM-main RL lifecycle hooks."""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path
from queue import Queue
from types import ModuleType, SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "lumenrl"
    / "engine"
    / "inference"
    / "atom_patch"
    / "lifecycle_main.py"
)
spec = importlib.util.spec_from_file_location(
    "_lumen_atom_lifecycle_main_test", MODULE_PATH
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


class FakePool:
    num_used = 0
    num_blocks = 10


class FakeSlots:
    def __init__(self):
        self._pinned = {}
        self._relocations = []

    def num_free(self):
        return 4


class FakeBlockManager:
    def __init__(self, config, *, state_runtime=None):
        self.kv = FakePool()
        self.kv.num_blocks = config.num_kvcache_blocks
        self.num_state_slots = 4
        self.state = FakeSlots()
        self._state_loads = []
        self._orphan_load_slots = {}
        self.draft_lookahead = 0
        self.state_runtime_seen = state_runtime

    def pool_pressure(self):
        return self.kv.num_blocks

    def has_pending_state_store_pins(self):
        return False


class FakeScheduler:
    def __init__(self, config, *, state_runtime=None):
        self.config = config
        self.block_manager = FakeBlockManager(config, state_runtime=state_runtime)
        self.running = deque()
        self.waiting = deque()
        self.deferred_free_blocks = {}
        self._pp_inflight_token_block = set()
        self._partial_prefill_count = 0
        self._num_parked_remote_kv = 0
        self.finished_recving_kv_req_ids = []
        self.failed_recving_kv_req_ids = []
        self.kv_connector = None
        self.engine_stats = SimpleNamespace(
            _pool_pressure=self.block_manager.pool_pressure,
            block_manager=self.block_manager,
        )


class FakeReplyQueue:
    def __init__(self, queue):
        self.queue = queue

    def put_nowait(self, item):
        self.queue.put_nowait(item)


class FakeHandler:
    _UTILITY_HANDLERS = {}

    def __init__(self, runner_mgr, output_queue, label="Engine Core", scheduler=None):
        self.scheduler = scheduler
        self.output_queue = FakeReplyQueue(output_queue)
        self.calls = runner_mgr

    def _utility_on_every_rank(self, command, method, *args, timeout):
        self.calls.append((command, method, args))
        return {
            "cmd": command,
            "result": self.calls.capacity
            if command == "get_kv_cache_capacity"
            else True,
        }

    def _handle_release_memory(self, args):
        self.calls.append(("original-release", args))
        return {"result": True}

    def _handle_resume_memory(self, args):
        self.calls.append(("original-resume", args))
        return {"result": True}

    def _handle_clear_kv_cache(self, args):
        self.calls.append(("original-clear", args))
        return {"result": True}


class Calls(list):
    capacity = 10


@pytest.fixture
def atom_stubs(monkeypatch):
    def module(name, **attrs):
        obj = ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)
        return obj

    module("atom.model_engine.sequence", SequenceStatus=SimpleNamespace(WAITING=1))
    module("atom.model_engine.block_manager", BlockManager=FakeBlockManager)
    module(
        "atom.model_engine.scheduler", Scheduler=type("Scheduler", (FakeScheduler,), {})
    )
    module(
        "atom.model_engine.engine_utility",
        EngineUtilityHandler=type("EngineUtilityHandler", (FakeHandler,), {}),
        _remaining_timeout=lambda args, default: args.get("timeout", default),
    )
    module("atom.model_engine.state_runtime", DEFAULT_STATE_RUNTIME=object())
    module(
        "atom.rollout.memory_manager",
        MemoryManagerMixin=type(
            "MemoryManagerMixin",
            (),
            {
                "_release_kv_cache": lambda self: True,
                "_resume_kv_cache": lambda self: True,
            },
        ),
        sleep_keeps_memory_resident=lambda runner: False,
    )
    module(
        "atom.rollout.weight_updater",
        WeightUpdaterMixin=type(
            "WeightUpdaterMixin",
            (),
            {
                "_sync_target_model": lambda self: self.model,
                "update_weights": lambda self, *a, **k: "direct",
                "update_weights_from_shm": lambda self, *a, **k: "shm",
                "update_weights_from_ipc": lambda self, *a, **k: "ipc",
            },
        ),
    )
    module(
        "atom.model_engine.model_runner",
        ModelRunner=type(
            "ModelRunner",
            (),
            {
                "get_num_blocks": lambda self: {},
                "allocate_kv_cache": lambda self, n: n,
            },
        ),
    )
    monkeypatch.setenv("LUMEN_ATOM_MAIN_LIFECYCLE", "1")
    return sys.modules


def make_scheduler(atom_stubs, blocks=10, runtime=None):
    config = SimpleNamespace(
        num_kvcache_blocks=blocks, pipeline_parallel_size=1, enable_prefix_caching=True
    )
    return atom_stubs["atom.model_engine.scheduler"].Scheduler(
        config, state_runtime=runtime
    )


def test_worker_refit_receipt_utility_requires_all_rank_reply(atom_stubs):
    patch.install_lifecycle_main()
    cls = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler
    assert cls._UTILITY_HANDLERS["get_lumen_refit_receipts"] == (
        "_handle_get_lumen_refit_receipts"
    )
    calls = Calls()
    out = Queue()
    handler = cls(calls, out, scheduler=make_scheduler(atom_stubs))
    reply = handler._handle_get_lumen_refit_receipts({"timeout": 2.0})
    assert reply["cmd"] == "get_lumen_refit_receipts"
    assert calls[-1][1] == "get_lumen_refit_receipt"
    assert calls[-1][2] == (True, None)
    assert out.get_nowait()[1] == reply


def test_kv_capacity_receipt_uses_the_worker_utility_rpc(atom_stubs):
    patch.install_lifecycle_main()
    cls = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler
    calls = Calls()
    calls.capacity = 6
    out = Queue()
    handler = cls(calls, out, scheduler=make_scheduler(atom_stubs))

    method = cls._UTILITY_HANDLERS["get_lumen_kv_cache_capacity"]
    assert method == "_handle_get_lumen_kv_cache_capacity"
    reply = getattr(handler, method)({"timeout": 2.0})

    assert reply == {"cmd": "get_lumen_kv_cache_capacity", "result": 6}
    assert calls[-1] == (
        "get_kv_cache_capacity",
        "get_kv_cache_capacity",
        (),
    )
    assert out.get_nowait() == ("UTILITY_RESPONSE", reply)


def test_default_off_is_noop(monkeypatch, atom_stubs):
    monkeypatch.delenv("LUMEN_ATOM_MAIN_LIFECYCLE")
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler
    original = handler._handle_resume_memory
    assert patch.install_lifecycle_main() is False
    assert handler._handle_resume_memory is original


@pytest.mark.parametrize("blocks", [6, 10, 14])
def test_resume_rebuilds_cpu_ids_only_after_worker_ack(atom_stubs, blocks):
    assert patch.install_lifecycle_main() is True
    runtime = object()
    scheduler = make_scheduler(atom_stubs, runtime=runtime)
    previous = scheduler.block_manager
    calls = Calls()
    calls.capacity = blocks
    out = Queue()
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler(
        calls, out, scheduler=scheduler
    )
    assert handler._handle_resume_memory({"tags": ["kv_cache"]})["result"] is True
    assert scheduler.block_manager is not previous
    assert scheduler.block_manager.kv.num_blocks == blocks
    assert scheduler.block_manager.state_runtime_seen is runtime
    assert scheduler.engine_stats.block_manager is scheduler.block_manager
    assert scheduler.engine_stats._pool_pressure.__self__ is scheduler.block_manager
    assert out.get_nowait()[1]["result"] is True
    assert [item[0] for item in calls] == ["resume_memory", "get_kv_cache_capacity"]


@pytest.mark.parametrize("capacity", [None, False, 0, -1, 2.5, {}])
def test_invalid_capacity_no_ack_and_no_cpu_swap(atom_stubs, capacity):
    patch.install_lifecycle_main()
    scheduler = make_scheduler(atom_stubs)
    before = scheduler.block_manager
    calls = Calls()
    calls.capacity = capacity
    out = Queue()
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler(
        calls, out, scheduler=scheduler
    )
    with pytest.raises(RuntimeError, match="Invalid real"):
        handler._handle_resume_memory({"tags": ["kv_cache"]})
    assert scheduler.block_manager is before
    assert out.empty()


@pytest.mark.parametrize(
    "field,value",
    [
        ("running", deque([object()])),
        ("deferred_free_blocks", {1: object()}),
        ("_pp_inflight_token_block", {1}),
        ("_partial_prefill_count", 1),
        ("_num_parked_remote_kv", 1),
    ],
)
def test_active_state_prevents_worker_rpc(atom_stubs, field, value):
    patch.install_lifecycle_main()
    scheduler = make_scheduler(atom_stubs)
    setattr(scheduler, field, value)
    calls = Calls()
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler(
        calls, Queue(), scheduler=scheduler
    )
    with pytest.raises(RuntimeError, match="drained"):
        handler._handle_resume_memory({"tags": ["kv_cache"]})
    assert not calls


def test_weights_only_uses_unmodified_main_handler(atom_stubs):
    patch.install_lifecycle_main()
    calls = Calls()
    scheduler = make_scheduler(atom_stubs)
    before = scheduler.block_manager
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler(
        calls, Queue(), scheduler=scheduler
    )
    assert handler._handle_resume_memory({"tags": ["weights"]}) == {"result": True}
    assert scheduler.block_manager is before
    assert calls[0][0] == "original-resume"


def test_clear_invalidates_equal_size_prefix(atom_stubs):
    patch.install_lifecycle_main()
    scheduler = make_scheduler(atom_stubs, runtime=object())
    before = scheduler.block_manager
    calls = Calls()
    out = Queue()
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler(
        calls, out, scheduler=scheduler
    )
    handler._handle_clear_kv_cache({})
    assert scheduler.block_manager is not before
    assert scheduler.config.num_kvcache_blocks == 10
    assert out.get_nowait()[1]["result"] is True


def test_unsupported_connector_and_pp_fail_before_rpc(atom_stubs):
    patch.install_lifecycle_main()
    scheduler = make_scheduler(atom_stubs)
    calls = Calls()
    handler = atom_stubs["atom.model_engine.engine_utility"].EngineUtilityHandler(
        calls, Queue(), scheduler=scheduler
    )
    scheduler.kv_connector = object()
    with pytest.raises(RuntimeError, match="configured KV connectors"):
        handler._handle_release_memory({"tags": ["kv_cache"]})
    scheduler.kv_connector = None
    scheduler.config.pipeline_parallel_size = 2
    with pytest.raises(RuntimeError, match="pipeline parallelism"):
        handler._handle_resume_memory({"tags": ["kv_cache"]})
    assert not calls


def test_worker_wake_rejects_invalid_capacity_and_preserves_saved_count(
    monkeypatch, atom_stubs
):
    patch.install_lifecycle_main_worker()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                empty_cache=lambda: None, reset_peak_memory_stats=lambda: None
            )
        ),
    )
    worker = atom_stubs["atom.rollout.memory_manager"].MemoryManagerMixin()
    worker.config = SimpleNamespace(
        num_kvcache_blocks=10, pool_entries={"kv": 10, "state": 4}
    )
    worker.kv_cache = None
    worker._kv_cache_num_blocks = 10
    worker._lumen_sleep_pool_entries = {"state": 4}
    worker._lumen_sleep_pool_per_req = {}
    worker._lumen_sleep_state_runtime_wire = {"transfer": {}}
    worker._lumen_sleep_pool_plan = SimpleNamespace(paged_class="kv")
    worker.get_num_blocks = lambda: {
        "num_kvcache_blocks": 0,
        "pool_entries": {"kv": 0, "state": 4},
    }
    with pytest.raises(RuntimeError, match="Invalid KV wake capacity"):
        worker._resume_kv_cache()
    assert worker._kv_cache_num_blocks == 10


def test_worker_wake_accepts_drained_paged_shrink_and_rebinds_plan(monkeypatch, atom_stubs):
    patch.install_lifecycle_main_worker()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: None,
                             reset_peak_memory_stats=lambda: None)
    ))

    class PoolPlan:
        paged_class = "kv"
        entries_per_req = {}
        entry_bytes = {"kv": 42, "state": 16}

        def __init__(self, count):
            self.paged_entries = count
            self.entries = {"kv": count, "state": 4}

        def with_paged_entries(self, count):
            return PoolPlan(count)

    worker = atom_stubs["atom.rollout.memory_manager"].MemoryManagerMixin()
    original = PoolPlan(10)
    probed = PoolPlan(6)
    worker.config = SimpleNamespace(
        num_kvcache_blocks=10, pool_entries=dict(original.entries),
        pool_entries_per_req={}, pipeline_parallel_size=1,
        kv_transfer_config=None, disagg_is_decode=False,
    )
    worker.kv_cache = None
    worker.pool_plan = original
    worker._kv_cache_num_blocks = 10
    worker._lumen_sleep_pool_entries = {"state": 4}
    worker._lumen_sleep_pool_per_req = {}
    worker._lumen_sleep_state_runtime_wire = {"transfer": {}}
    worker._lumen_sleep_pool_plan = original
    allocated = []

    def probe():
        worker.pool_plan = probed
        worker.config.pool_entries = dict(probed.entries)
        return {"num_kvcache_blocks": 6, "pool_entries": dict(probed.entries),
                "pool_entries_per_req": {}, "state_runtime": {"transfer": {}}}

    def allocate(count):
        allocated.append(count)
        worker.kv_cache = object()

    worker.get_num_blocks = probe
    worker.allocate_kv_cache = allocate
    worker._resume_kv_cache()
    assert allocated == [6]
    assert worker.pool_plan is probed
    assert worker.config.num_kvcache_blocks == 6
    assert worker.config.pool_entries == {"kv": 6, "state": 4}
    assert worker._kv_cache_num_blocks is None


@pytest.mark.parametrize(
    "geometry,expected_error",
    [
        ("state_pool", "non-paged state slot capacities"),
        ("request", "per-request state slot geometry"),
        ("state_runtime", "StateRuntime transfer geometry"),
    ],
)
def test_worker_wake_refuses_non_paged_state_geometry_changes(
    monkeypatch, atom_stubs, geometry, expected_error
):
    patch.install_lifecycle_main_worker()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                empty_cache=lambda: None, reset_peak_memory_stats=lambda: None
            )
        ),
    )

    class PoolPlan:
        paged_class = "kv"
        entry_bytes = {"kv": 42, "state": 16}

        def __init__(self, count, state_slots=4, entries_per_req=None):
            self.paged_entries = count
            self.entries = {"kv": count, "state": state_slots}
            self.entries_per_req = dict(entries_per_req or {})

        def with_paged_entries(self, count):
            return PoolPlan(count)

    original = PoolPlan(10, entries_per_req={"state": 1})
    probed = PoolPlan(6, entries_per_req={"state": 1})
    probed_runtime = {"transfer": {"mode": "original"}}
    if geometry == "state_pool":
        probed = PoolPlan(6, state_slots=5, entries_per_req={"state": 1})
    elif geometry == "request":
        probed = PoolPlan(6, entries_per_req={"state": 2})
    else:
        probed_runtime = {"transfer": {"mode": "changed"}}

    worker = atom_stubs["atom.rollout.memory_manager"].MemoryManagerMixin()
    worker.config = SimpleNamespace(
        num_kvcache_blocks=10,
        pool_entries=dict(original.entries),
        pool_entries_per_req=dict(original.entries_per_req),
        pipeline_parallel_size=1,
        kv_transfer_config=None,
        disagg_is_decode=False,
    )
    worker.kv_cache = None
    worker.pool_plan = original
    worker._kv_cache_num_blocks = 10
    worker._lumen_sleep_pool_entries = {"state": 4}
    worker._lumen_sleep_pool_per_req = dict(original.entries_per_req)
    worker._lumen_sleep_state_runtime_wire = {
        "transfer": {"mode": "original"}
    }
    worker._lumen_sleep_pool_plan = original
    allocated = []

    def probe():
        worker.pool_plan = probed
        worker.config.pool_entries = dict(probed.entries)
        worker.config.pool_entries_per_req = dict(probed.entries_per_req)
        return {
            "num_kvcache_blocks": 6,
            "pool_entries": dict(probed.entries),
            "pool_entries_per_req": dict(probed.entries_per_req),
            "state_runtime": probed_runtime,
        }

    worker.get_num_blocks = probe
    worker.allocate_kv_cache = lambda count: allocated.append(count)
    with pytest.raises(RuntimeError, match=expected_error):
        worker._resume_kv_cache()

    assert allocated == []
    assert worker.kv_cache is None
    assert worker._kv_cache_num_blocks == 10
    assert worker._lumen_sleep_pool_plan is original


def test_worker_wake_allocation_failure_keeps_sleep_fence(monkeypatch, atom_stubs):
    patch.install_lifecycle_main_worker()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                empty_cache=lambda: None, reset_peak_memory_stats=lambda: None
            )
        ),
    )

    class PoolPlan:
        paged_class = "kv"
        entries_per_req = {}
        entry_bytes = {"kv": 42, "state": 16}

        def __init__(self, count):
            self.paged_entries = count
            self.entries = {"kv": count, "state": 4}

        def with_paged_entries(self, count):
            return PoolPlan(count)

    original = PoolPlan(10)
    probed = PoolPlan(6)
    worker = atom_stubs["atom.rollout.memory_manager"].MemoryManagerMixin()
    worker.config = SimpleNamespace(
        num_kvcache_blocks=10,
        pool_entries=dict(original.entries),
        pool_entries_per_req={},
        pipeline_parallel_size=1,
        kv_transfer_config=None,
        disagg_is_decode=False,
    )
    worker.kv_cache = None
    worker.pool_plan = original
    worker._kv_cache_num_blocks = 10
    worker._lumen_sleep_pool_entries = {"state": 4}
    worker._lumen_sleep_pool_per_req = {}
    worker._lumen_sleep_state_runtime_wire = {"transfer": {"mode": "original"}}
    worker._lumen_sleep_pool_plan = original
    attempted = []

    def probe():
        worker.pool_plan = probed
        worker.config.pool_entries = dict(probed.entries)
        return {
            "num_kvcache_blocks": 6,
            "pool_entries": dict(probed.entries),
            "pool_entries_per_req": {},
            "state_runtime": {"transfer": {"mode": "original"}},
        }

    def fail_allocation(count):
        attempted.append(count)
        raise RuntimeError("simulated KV allocation failure")

    worker.get_num_blocks = probe
    worker.allocate_kv_cache = fail_allocation
    with pytest.raises(RuntimeError, match="simulated KV allocation failure"):
        worker._resume_kv_cache()

    assert attempted == [6]
    assert worker.kv_cache is None
    assert worker._kv_cache_num_blocks == 10
    assert worker._lumen_sleep_pool_plan is original
    assert worker._lumen_sleep_pool_entries == {"state": 4}
    assert worker._lumen_sleep_pool_per_req == {}
    assert worker._lumen_sleep_state_runtime_wire == {
        "transfer": {"mode": "original"}
    }


def test_weight_guard_refuses_quantized_experts_before_update(atom_stubs):
    assert patch.install_weight_sync_guard() is True
    updater = atom_stubs["atom.rollout.weight_updater"].WeightUpdaterMixin()
    param = SimpleNamespace(
        dtype=SimpleNamespace(is_floating_point=False), element_size=lambda: 1
    )
    expert = SimpleNamespace(
        w13_weight=param, w2_weight=None, expert_map=None, num_redundant_experts=0
    )
    updater.model = SimpleNamespace(modules=lambda: [expert])
    with pytest.raises(NotImplementedError, match="quantized FusedMoE"):
        updater.update_weights([("model.experts.w13_weight", object())])
    assert updater.update_weights([]) == "direct"
    expert.w13_weight = SimpleNamespace(
        dtype=SimpleNamespace(is_floating_point=True), element_size=lambda: 2
    )
    assert updater.update_weights([]) == "direct"
