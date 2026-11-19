"""The runner Lumen-RL hands every ATOM replica (``NoCustomARModelRunner``).

ATOM is not importable here, so its runner is a stand-in with the behaviour that
matters: ``MemoryManagerMixin.clear_kv_cache`` reads ``self.kv_cache`` unguarded,
which ATOM only sets when its shared paged pool is not empty -- never for
DeepSeek-V4.1, whose attention builders own every pool -- and
``_requantize_fp8_weight`` knows only ATOM's own layouts.
"""

import sys
import types

import pytest
import torch

from lumenrl.engine.inference import model_runner_nocustomar as m


class _AtomRunner:
    def __init__(self, rank, config, with_pool=False, world_size=1):
        self.rank, self.world_size, self.device = rank, world_size, torch.device("cpu")
        self.cleared, self.requantized, self.post_processed = False, [], []
        if config == "model":
            self.model = torch.nn.Linear(4, 3)
        if with_pool:
            self.kv_cache = types.SimpleNamespace(zero_=lambda: setattr(self, "cleared", True))

    def update_weights(self, named_tensors):
        return len(named_tensors)

    def clear_kv_cache(self):
        kv = self.kv_cache
        if kv is None:
            return True
        kv.zero_()
        return True

    def _requantize_fp8_weight(self, module, param_name, param, tensor):
        self.requantized.append(param_name)

    def _copy_into_param(self, param, tensor):
        param.data.copy_(tensor)

    def _post_process_fp8_weight(self, module, param):
        self.post_processed.append(module)

    def _release_weights(self):
        self.discarded = True
        for p in self.model.parameters():
            p.data = torch.empty(0, dtype=p.dtype)

    def _resume_weights(self):
        self.resumed_by_atom = True

    def _weight_memory_modules(self):
        return []


@pytest.fixture
def runner_cls(monkeypatch):
    ext = types.ModuleType("atom.rollout.model_runner_ext")
    ext.RLHFModelRunner = _AtomRunner
    memory = types.ModuleType("atom.rollout.memory_manager")
    memory.release_cudagraphs = lambda runner: setattr(runner, "graphs_released", True)
    monkeypatch.setitem(sys.modules, "atom.rollout.model_runner_ext", ext)
    monkeypatch.setitem(sys.modules, "atom.rollout.memory_manager", memory)
    monkeypatch.setattr(m.NoCustomARModelRunner, "_real_cls", None)
    return m.NoCustomARModelRunner


def _with_model(runner):
    runner.model = torch.nn.Linear(4, 3)
    return {n: p.detach().clone() for n, p in runner.model.named_parameters()}


def test_the_host_copy_is_made_when_the_runner_starts(runner_cls, monkeypatch):
    monkeypatch.setenv("LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS", "1")
    runner = runner_cls(0, "model")

    host = runner._host_weights
    assert set(host) == {n for n, _ in runner.model.named_parameters()}
    runner._release_weights()
    assert runner._host_weights is host


def test_the_host_copy_is_packed_into_chunks_no_larger_than_a_power_of_two(runner_cls, monkeypatch):
    """The pinned allocator rounds each allocation up to a power of two and pins all of
    it, so one allocation of the whole copy pinned up to twice its size (137 GB for the
    84.8 GB of a V4.1 TP rank)."""
    monkeypatch.setattr(m, "HOST_CHUNK_BYTES", 1024, raising=False)
    runner = runner_cls(0, None)
    runner.model = torch.nn.Sequential(*(torch.nn.Linear(n, 4) for n in (40, 24, 8, 30, 16)))
    sizes = {n: p.numel() * p.element_size() for n, p in runner.model.named_parameters()}

    host = m.host_weight_buffers(runner)

    chunks = {}
    for name, t in host.items():
        assert t.shape == dict(runner.model.named_parameters())[name].shape
        chunks.setdefault(t.untyped_storage().data_ptr(), []).append((t.storage_offset() * t.element_size(), sizes[name]))
    nbytes = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in host.values()}
    assert len(chunks) > 1 and all(n <= 1024 for n in nbytes.values())
    for spans in chunks.values():
        spans.sort()
        assert all(a + n <= b for (a, n), (b, _) in zip(spans, spans[1:])), spans
    assert sum(nbytes.values()) < sum(-(-s // 256) * 256 for s in sizes.values()) + 1024


def test_sleep_can_keep_the_weights_for_a_partial_sync(runner_cls, monkeypatch):
    """A trainer syncing only what it trains (V4.1's frozen experts never go) must
    wake to the weights it slept with, not to ATOM's discarded placeholders."""
    runner = runner_cls(0, None)
    want = _with_model(runner)
    monkeypatch.setenv("LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS", "1")

    for _ in range(2):  # the host copy is reused
        runner._release_weights()
        assert all(p.numel() == 0 for p in runner.model.parameters()) and runner.graphs_released
        runner._resume_weights()
        for n, p in runner.model.named_parameters():
            assert torch.equal(p, want[n]), n
    assert not getattr(runner, "discarded", False) and not getattr(runner, "resumed_by_atom", False)


def test_by_default_sleep_is_atoms(runner_cls, monkeypatch):
    monkeypatch.delenv("LUMENRL_ATOM_SLEEP_KEEPS_WEIGHTS", raising=False)
    runner = runner_cls(0, None)
    _with_model(runner)

    runner._release_weights()
    runner._resume_weights()

    assert runner.discarded and runner.resumed_by_atom


def test_a_runner_without_a_kv_pool_clears_without_raising(runner_cls):
    runner = runner_cls(0, None, False)

    assert runner.clear_kv_cache() is True
    assert runner.kv_cache is None


def test_a_runner_with_a_pool_still_zeroes_it(runner_cls):
    runner = runner_cls(0, None, True)

    assert runner.clear_kv_cache() is True and runner.cleared


def test_the_factory_answers_for_the_runner_it_builds(runner_cls):
    """ATOM keeps V4.1 off its "mono" decode only for a runner class that takes
    weight updates, and asks the class, not an instance."""
    assert hasattr(runner_cls, "update_weights")
    assert not hasattr(runner_cls, "no_such_capability")


def _release_like(n, k, rows, generator):
    codes = (torch.randn(n, k, generator=generator) * 60).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    exps = torch.randint(-12, 2, (n // rows, k // 32), generator=generator).float()
    return codes * torch.exp2(exps).repeat_interleave(rows, 0).repeat_interleave(32, 1)


@pytest.mark.parametrize("rows", [32, 1])
def test_a_dequantized_release_weight_requantizes_to_the_same_values(rows):
    w = _release_like(64, 96, rows, torch.Generator().manual_seed(0))

    q, scale = m.quantize_native_group(w, rows, torch.float8_e4m3fn)

    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (64 // rows, 96 // 32)
    back = q.float() * scale.float().repeat_interleave(rows, 0).repeat_interleave(32, 1)
    assert torch.equal(back, w)


def test_scales_round_up_to_a_power_of_two():
    w = torch.full((32, 32), 448.0 * 3)  # amax / 448 = 3 -> 4

    _, scale = m.quantize_native_group(w, 32, torch.float8_e4m3fn)

    assert scale.float().item() == 4.0


def _native_linear(n, k, rows):
    module = torch.nn.Module()
    module.native_a8_group_rows = rows
    module.weight = torch.nn.Parameter(torch.zeros(n, k, dtype=torch.float8_e4m3fn), requires_grad=False)
    module.weight_scale = torch.nn.Parameter(torch.zeros(n // rows, k // 32, dtype=torch.float8_e8m0fnu),
                                             requires_grad=False)
    return module


def test_a_native_group_weight_is_written_with_its_scale_from_this_ranks_shard(runner_cls):
    full = _release_like(128, 64, 32, torch.Generator().manual_seed(1))
    runner = runner_cls(1, None, world_size=2)
    module = _native_linear(64, 64, 32)

    runner._requantize_fp8_weight(module, "weight", module.weight, full)

    back = module.weight.float() * module.weight_scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert torch.equal(back, full[64:])
    assert runner.post_processed == [module] and runner.requantized == []


def test_other_layouts_stay_with_atom(runner_cls):
    runner = runner_cls(0, None)
    module = torch.nn.Module()

    runner._requantize_fp8_weight(module, "weight", None, None)

    assert runner.requantized == ["weight"]


def test_a_weight_that_does_not_shard_is_an_error():
    with pytest.raises(ValueError, match="does not shard"):
        m.rank_shard(torch.zeros(10, 4), (3, 4), 0, 2)
