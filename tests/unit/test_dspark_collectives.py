import torch

from lumenrl.trainer.spec_distill_trainer import (
    SpecDistillTrainer,
    _balanced_disagg_rows,
)


def test_balanced_rows_preserve_denominator_groups_and_samples():
    lengths = [
        100, 90, 20, 10,
        80, 70, 60, 5,
        95, 50, 40, 30,
    ]
    assignments = [
        _balanced_disagg_rows(lengths, world_size=4, rank=rank)
        for rank in range(4)
    ]

    assert all(len(rows) == 3 for rows in assignments)
    assert sorted(row for rows in assignments for row in rows) == list(range(12))
    for group_index in range(3):
        assert {
            assignments[rank][group_index] for rank in range(4)
        } == set(range(group_index * 4, (group_index + 1) * 4))

    balanced_loads = [
        sum(lengths[row] for row in rows) for rows in assignments
    ]
    strided_loads = [
        sum(lengths[row] for row in range(rank, len(lengths), 4))
        for rank in range(4)
    ]
    assert max(balanced_loads) - min(balanced_loads) < (
        max(strided_loads) - min(strided_loads)
    )


def test_loss_denominators_are_summed_in_one_vector_collective(monkeypatch):
    trainer = object.__new__(SpecDistillTrainer)
    trainer._is_distributed = True
    local = [torch.tensor(float(index)) for index in range(16)]
    peer = torch.arange(100.0, 116.0)
    calls = []

    def fake_all_reduce(values, op):
        calls.append((values.clone(), op))
        values.add_(peer)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    result = trainer._sum_loss_denominators(local)

    assert len(calls) == 1
    assert calls[0][1] == torch.distributed.ReduceOp.SUM
    torch.testing.assert_close(calls[0][0], torch.arange(16.0))
    torch.testing.assert_close(result, torch.arange(16.0) + peer)
    torch.testing.assert_close(torch.stack(local), torch.arange(16.0))


def test_batched_metric_average_preserves_key_mapping(monkeypatch):
    trainer = object.__new__(SpecDistillTrainer)
    trainer._is_distributed = True
    trainer._device = torch.device("cpu")
    metrics = {"loss": 1.0, "lr": 2.0, "timing/train_s": 3.0}
    peer = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
    calls = []

    def fake_all_reduce(values, op):
        calls.append((values.clone(), op))
        values.copy_((values + peer) / 2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    trainer._average_metrics_across_ranks(metrics)

    assert len(calls) == 1
    assert calls[0][1] == torch.distributed.ReduceOp.AVG
    assert metrics == {
        "loss": 2.0,
        "lr": 3.0,
        "timing/train_s": 4.0,
    }
