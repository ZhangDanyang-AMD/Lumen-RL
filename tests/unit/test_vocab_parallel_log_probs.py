"""Vocab-parallel selected log-probs (P7): TP-sharded logits, gloo on CPU."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lumenrl.engine.training.packing import selected_log_probs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, v, temperature, chunk_rows, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from lumenrl.engine.training.vocab_parallel import vocab_parallel_selected_log_probs

        g = torch.Generator().manual_seed(0)
        full = torch.randn(7, v, generator=g, dtype=torch.float64) * 3
        ids = torch.randint(0, v, (7, 5), generator=g)
        ids[:, 1] = ids[:, 0]  # duplicate id in a row
        ids[0, 2] = v - 1      # last vocab id (last shard)
        g_out = torch.randn(7, 5, generator=g, dtype=torch.float64)
        shard = v // world
        start = rank * shard
        local = full[:, start:start + shard].clone().requires_grad_(True)
        out = vocab_parallel_selected_log_probs(
            local, ids, vocab_start=start, temperature=temperature,
            chunk_rows=chunk_rows, group=dist.group.WORLD,
        )
        (out * g_out).sum().backward()
        # numpy, not tensors: torch's Queue shares tensors by fd, which races
        # with this process exiting.
        out_q.put((rank, out.detach().numpy(), local.grad.detach().numpy(), start))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("chunk_rows", [3, 4096])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_vocab_parallel_matches_full_vocab_op(chunk_rows, temperature) -> None:
    world, v = 2, 12
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, v, temperature, chunk_rows, q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = sorted(q.get(timeout=120) for _ in range(world))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    g = torch.Generator().manual_seed(0)
    full = (torch.randn(7, v, generator=g, dtype=torch.float64) * 3).requires_grad_(True)
    ids = torch.randint(0, v, (7, 5), generator=g)
    ids[:, 1] = ids[:, 0]
    ids[0, 2] = v - 1
    g_out = torch.randn(7, 5, generator=g, dtype=torch.float64)
    ref = selected_log_probs(full, ids, temperature=temperature)
    (ref * g_out).sum().backward()

    for rank, out, grad, start in results:
        out, grad = torch.from_numpy(out), torch.from_numpy(grad)
        torch.testing.assert_close(out, ref.detach())  # identical on every TP rank
        torch.testing.assert_close(grad, full.grad[:, start:start + v // world])
