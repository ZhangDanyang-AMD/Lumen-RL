"""Routing replay through activation recomputation, with Megatron's RouterReplay."""

from __future__ import annotations

import pytest
import torch

router_replay = pytest.importorskip("megatron.core.transformer.moe.router_replay")

from lumenrl.engine.training.megatron_r3_replay import (  # noqa: E402
    clear_replay,
    clear_stale_instances,
    install_replay,
    queued_recompute_routes,
    replay_on_recompute,
)

SCORES = torch.zeros(3, 4)


class Router(torch.nn.Module):
    """Megatron's router as far as replay goes; ``hashed`` routes by token id."""

    def __init__(self, hashed: bool = False) -> None:
        super().__init__()
        self.router_replay = router_replay.RouterReplay()
        self.tid2eid = torch.zeros(8, 2, dtype=torch.int32) if hashed else None


def native_topk(scores, topk, num_groups=None, group_topk=None):
    return torch.topk(scores, topk)


def routes(model):
    """What each learned router routes; a hashed router never consults replay."""
    return [r.router_replay.get_replay_topk(SCORES, 2, default_compute_topk=native_topk)[1]
            for r in model if r.tid2eid is None]


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.fixture
def model():
    clear_stale_instances()
    routers = torch.nn.ModuleList([Router(hashed=True), Router(), Router()])
    clear_replay()
    yield routers
    clear_stale_instances()


def microbatch(m):
    """Microbatch ``m``'s replayed experts, one tensor per router, distinct for every
    (microbatch, router) and below the 4 experts SCORES has."""
    return [torch.full((3, 2), (2 * m + layer) % 4) for layer in range(3)]


def test_a_recomputed_layer_routes_like_the_forward_it_repeats(model):
    # 1F1B with two microbatches in flight: F0 F1 B0 B1.
    for m in (0, 1):
        install_replay(microbatch(m), append=True)
        assert same(routes(model), microbatch(m)[1:])
        replay_on_recompute()

    assert same(routes(model), microbatch(0)[1:])
    assert same(routes(model), microbatch(1)[1:])
    assert queued_recompute_routes(model) == 0  # the hashed router's queue is not counted


def test_without_the_switch_a_recompute_replays_the_latest_forward(model):
    for m in (0, 1):
        install_replay(microbatch(m), append=True)

    assert same(routes(model), microbatch(1)[1:])  # what B0 would have routed
    assert queued_recompute_routes(model) == 4  # 2 learned routers x 2 microbatches
