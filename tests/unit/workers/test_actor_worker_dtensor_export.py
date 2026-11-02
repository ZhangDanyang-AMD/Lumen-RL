"""FSDP2 yields DTensor shards to weight sync; the export must write whole tensors."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lumenrl.workers.actor_worker import LumenActorWorker

_WORLD = 2
_SHAPES = {
    "model.embed_tokens.weight": (6, 4),
    "model.norm.weight": (8,),
}


def _full(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(len(name))
    return torch.randn(shape, generator=generator)


class _ShardedEngine:
    def __init__(self, mesh) -> None:
        self._mesh = mesh

    def get_per_tensor_param(self):
        from torch.distributed.tensor import Shard, distribute_tensor

        def params():
            for name, shape in _SHAPES.items():
                yield name, distribute_tensor(_full(name, shape), self._mesh, [Shard(0)])

        return params(), None


def _export_rank(rank: int, port: int, sync_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD,
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh

        worker = SimpleNamespace(
            _engine=_ShardedEngine(init_device_mesh("cpu", (_WORLD,))),
            rank=rank,
            _ensure_weight_http_server=lambda _output_dir: "",
        )
        LumenActorWorker.export_state_dict_safetensors(worker, sync_dir)
    finally:
        dist.destroy_process_group()


def test_export_writes_whole_tensors_from_fsdp2_shards(tmp_path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_export_rank, args=(port, str(tmp_path)), nprocs=_WORLD, join=True)

    from safetensors.torch import load_file

    loaded: dict[str, torch.Tensor] = {}
    for path in sorted(tmp_path.glob("model-*.safetensors")):
        loaded.update(load_file(str(path)))
    assert set(loaded) == set(_SHAPES)
    for name, shape in _SHAPES.items():
        torch.testing.assert_close(loaded[name], _full(name, shape))
