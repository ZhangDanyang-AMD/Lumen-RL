"""Megatron engines free their DDP gradient buffers between updates and get them back."""

from __future__ import annotations

import torch

from lumenrl.engine.training.megatron_base_engine import MegatronBaseEngine


class Buffer:
    def __init__(self, numel: int) -> None:
        self.grad_data = torch.zeros(numel)


class DDP:
    """The two buffer lists of Megatron's DistributedDataParallel."""

    def __init__(self) -> None:
        self.buffers = [Buffer(8)]
        self.expert_parallel_buffers = [Buffer(4)]

    def zero_grad_buffer(self) -> None:
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.grad_data.zero_()


def engine() -> MegatronBaseEngine:
    eng = object.__new__(MegatronBaseEngine)
    eng._ddp = DDP()
    return eng


def test_released_buffers_come_back_zeroed_and_their_views_still_alias_them():
    eng = engine()
    dense = eng._ddp.buffers[0]
    main_grad = dense.grad_data[2:6]  # a parameter's main_grad is a view like this
    main_grad.fill_(3.0)

    freed = eng.release_grad_buffers()

    assert freed == (8 + 4) * 4
    assert dense.grad_data.untyped_storage().nbytes() == 0
    eng._zero_grad_buffer()
    assert dense.grad_data.untyped_storage().nbytes() == 8 * 4
    assert torch.equal(dense.grad_data, torch.zeros(8))
    main_grad.fill_(5.0)
    assert dense.grad_data[2:6].tolist() == [5.0] * 4


def test_releasing_twice_frees_nothing_more_and_an_unreleased_buffer_is_only_zeroed():
    eng = engine()
    eng._ddp.buffers[0].grad_data.fill_(1.0)

    eng._zero_grad_buffer()
    assert torch.equal(eng._ddp.buffers[0].grad_data, torch.zeros(8))
    assert eng.release_grad_buffers() == (8 + 4) * 4
    assert eng.release_grad_buffers() == 0.0
