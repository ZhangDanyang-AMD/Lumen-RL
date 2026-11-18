"""A colocated rollout replica reads its own host's /dev/shm, so a multi-host
export writes one copy per host."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lumenrl.workers.actor_worker import LumenActorWorker, first_rank_per_host

_WORLD = 2
_TENSORS = {"embed.weight": (6, 4), "norm.weight": (8,)}


def test_the_lowest_rank_on_each_host_writes():
    assert first_rank_per_host(["a", "b", "a", "b"]) == {0, 1}
    assert first_rank_per_host(["a", "a", "b", "b"]) == {0, 2}
    assert first_rank_per_host(["a", "a"]) == {0}


class _Engine:
    def get_per_tensor_param(self):
        return ((n, torch.full(s, float(len(n)))) for n, s in _TENSORS.items()), None


def _export_rank(rank: int, store: str, root: str, per_host: bool) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=_WORLD)
    try:
        socket.gethostname = lambda: f"host{rank}"  # each rank on a host of its own
        worker = SimpleNamespace(_engine=_Engine(), rank=rank, _ensure_weight_http_server=lambda _d: "")
        worker._first_rank_per_host = lambda: LumenActorWorker._first_rank_per_host(worker)
        LumenActorWorker.export_state_dict_safetensors(worker, f"{root}/host{rank}", per_host=per_host)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("per_host", [True, False])
def test_every_host_gets_its_own_copy_only_when_asked(tmp_path, per_host):
    mp.spawn(_export_rank, args=(str(tmp_path / "store"), str(tmp_path), per_host), nprocs=_WORLD, join=True)

    from safetensors.torch import load_file

    for rank in range(_WORLD):
        files = sorted((tmp_path / f"host{rank}").glob("model-*.safetensors"))
        loaded = {k: v for f in files for k, v in load_file(str(f)).items()}
        assert set(loaded) == (set(_TENSORS) if per_host or rank == 0 else set()), rank
