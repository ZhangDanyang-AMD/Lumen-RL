"""Unit tests for the shared Megatron-Native engine helpers."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

import lumenrl.engine.training  # noqa: F401 - populate EngineRegistry
from lumenrl.algorithms.loss_functions import asymmetric_clip_loss
from lumenrl.core.protocol import DataProto
from lumenrl.engine.training import megatron_engine, megatron_lumen_dsv4_engine
from lumenrl.engine.training.base_engine import EngineRegistry
from lumenrl.engine.training.megatron_base_engine import (
    MegatronBaseEngine,
    _FusedTokenLogProb,
)
from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine
from lumenrl.workers.actor_worker import LumenActorWorker


def test_megatron_backends_are_registered_for_existing_configs() -> None:
    language_backends = EngineRegistry._engines["language_model"]
    value_backends = EngineRegistry._engines["value_model"]

    assert "cuda" in language_backends["megatron"]
    assert "cuda" in value_backends["megatron"]
    assert "megatron_native" in language_backends
    assert "megatron_native" in value_backends


def test_native_engine_uses_shared_base() -> None:
    assert issubclass(MegatronNativeEngine, MegatronBaseEngine)


def test_native_pp_grpo_passes_sequence_normalization_under_dp_cp(
    monkeypatch,
) -> None:
    log_probs = torch.zeros(2, 2, requires_grad=True)
    scheduled_losses = []

    def get_forward_backward_func():
        def forward_backward(
            *,
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            **_,
        ):
            output, loss_func = forward_step_func(data_iterator, model)
            scaled_loss, metrics = loss_func(output)
            scheduled_loss = scaled_loss * 2 / num_microbatches
            scheduled_losses.append(float(scheduled_loss.detach()))
            scheduled_loss.backward()
            return [metrics]

        return forward_backward

    megatron = ModuleType("megatron")
    megatron_core = ModuleType("megatron.core")
    pipeline_parallel = ModuleType("megatron.core.pipeline_parallel")
    pipeline_parallel.get_forward_backward_func = get_forward_backward_func
    megatron.core = megatron_core
    megatron_core.pipeline_parallel = pipeline_parallel
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setitem(
        sys.modules, "megatron.core.pipeline_parallel", pipeline_parallel
    )

    class FakeModule:
        @staticmethod
        def train():
            return None

    class FakeDDP:
        @staticmethod
        def zero_grad_buffer():
            return None

    class FakeOptimizer:
        @staticmethod
        def zero_grad():
            return None

        @staticmethod
        def step():
            return True, torch.tensor(1.0), None

    engine = object.__new__(MegatronNativeEngine)
    engine._pp = 2
    engine._cp = 2
    engine._ep = 1
    engine._is_moe = False
    engine._is_last_stage = True
    engine._r3_enabled = False
    engine._r3_store = None
    engine.module = FakeModule()
    engine._ddp = FakeDDP()
    engine.optimizer = FakeOptimizer()
    engine._pp_setup_config = lambda: None
    engine.get_data_parallel_size = lambda: 2
    engine._collect_rows = lambda seqs, am: [(0, 0, 3), (1, 0, 3)]
    engine._build_microbatches = lambda seqs, rows: [
        {
            "rows": rows,
            "ids_list": [seqs[0], seqs[1]],
        }
    ]
    engine._pad_mbs_for_ep = lambda mbs: mbs
    engine._pp_forward_model = lambda model, ids: (
        torch.zeros(1, 1, requires_grad=True),
        [0, 1],
    )

    def cp_log_probs(logits, ids, row, **_):
        # Preserve the forward value while emulating CP all-reduce's backward SUM.
        return log_probs[row] * engine._cp, None

    engine._cp_row_logprob_entropy = cp_log_probs
    engine._sched_step = lambda: 1.0e-6
    engine._cur_lr = lambda: 1.0e-6
    batch = DataProto(
        tensors={
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.ones(2),
            "response_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        },
        meta={
            "algorithm": "grpo",
            "dp_size": 2,
            "algo_config": {
                "loss_agg_mode": "seq-mean-token-mean",
                "clip_ratio_high": 0.28,
                "grpo": {"clip_ratio": 0.2},
            },
        },
    )

    engine._pp_update_policy(batch)

    # Fallback global B is local_B * DP = 4. Two local sequence means each
    # contribute -1 * DP/global_B = -0.5; PP scheduling preserves their sum.
    assert scheduled_losses == pytest.approx([-1.0])
    torch.testing.assert_close(log_probs.grad, torch.full_like(log_probs, -0.5))


def test_actor_optimizer_config_propagates_optimizer_type() -> None:
    actor = LumenActorWorker(rank=0, world_size=1, config={})

    config = actor._build_optimizer_config(
        {"optimizer_type": "sgd", "sgd_momentum": 0.0}
    )

    assert config["optimizer"] == "sgd"
    assert config["sgd_momentum"] == 0.0


def test_rdma_weight_sync_releases_cuda_cache_before_gather(monkeypatch) -> None:
    events = []

    class FakeEngine:
        @staticmethod
        def get_per_tensor_param():
            assert events == ["synchronize", "empty_cache"]
            return iter(()), None

    actor = LumenActorWorker(rank=1, world_size=2, config={})
    actor._engine = FakeEngine()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    result = actor.send_weights_rdma(version=1, bucket_size_mb=1)

    assert result == {"writer": False}


def test_fused_token_log_prob_matches_reference_value_and_gradient() -> None:
    torch.manual_seed(7)
    target = torch.tensor([0, 3, 1, 4, 2], dtype=torch.long)
    weight = torch.randn(target.numel())

    fused_logits = torch.randn(target.numel(), 7, requires_grad=True)
    reference_logits = fused_logits.detach().clone().requires_grad_(True)

    fused = _FusedTokenLogProb.apply(fused_logits, target)
    reference = torch.log_softmax(reference_logits, dim=-1).gather(
        -1, target.unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(fused, reference)

    (fused * weight).sum().backward()
    (reference * weight).sum().backward()
    torch.testing.assert_close(fused_logits.grad, reference_logits.grad)


def test_pipeline_tokens_are_padded_to_tensor_parallel_alignment() -> None:
    token_ids = torch.tensor([11, 12, 13, 14, 15, 16])

    assert hasattr(megatron_engine, "_pad_token_ids_for_sequence_parallel")
    padded = megatron_engine._pad_token_ids_for_sequence_parallel(
        token_ids, tensor_parallel_size=4
    )

    assert padded.tolist() == [11, 12, 13, 14, 15, 16, 0, 0]
    assert padded.numel() % 4 == 0


def test_pipeline_logits_discard_sequence_parallel_padding() -> None:
    logits = torch.arange(8 * 3).reshape(8, 1, 3)

    assert hasattr(megatron_engine, "_flatten_pipeline_logits")
    flattened = megatron_engine._flatten_pipeline_logits(
        logits, unpadded_length=6
    )

    assert flattened.shape == (6, 3)
    torch.testing.assert_close(flattened, logits[:6, 0])


def test_grpo_rowwise_packed_and_pipeline_scaling_match() -> None:
    advantages = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    packed_logp = torch.zeros(2, 3, requires_grad=True)
    packed_loss = asymmetric_clip_loss(
        packed_logp,
        torch.zeros_like(packed_logp),
        advantages,
        0.2,
        0.28,
        mask=mask,
        loss_agg_mode="seq-mean-token-mean",
        global_batch_size=2,
    )
    packed_loss.backward()

    rowwise_logp = torch.zeros(2, 3, requires_grad=True)
    row_losses = [
        asymmetric_clip_loss(
            rowwise_logp[row : row + 1],
            torch.zeros_like(rowwise_logp[row : row + 1]),
            advantages[row : row + 1],
            0.2,
            0.28,
            mask=mask[row : row + 1],
            loss_agg_mode="seq-mean-token-mean",
            global_batch_size=2,
        )
        for row in range(2)
    ]
    rowwise_loss = sum(row_losses)
    rowwise_loss.backward()

    pipeline_logp = torch.zeros(2, 3, requires_grad=True)
    pipeline_rows = [
        asymmetric_clip_loss(
            pipeline_logp[row : row + 1],
            torch.zeros_like(pipeline_logp[row : row + 1]),
            advantages[row : row + 1],
            0.2,
            0.28,
            mask=mask[row : row + 1],
            loss_agg_mode="seq-mean-token-mean",
            global_batch_size=2,
        )
        for row in range(2)
    ]
    # Megatron divides each returned microbatch loss by num_microbatches.
    pipeline_loss = sum(
        megatron_engine._pipeline_schedule_loss(loss, 2)
        for loss in pipeline_rows
    ) / 2
    pipeline_loss.backward()

    torch.testing.assert_close(rowwise_loss, packed_loss)
    torch.testing.assert_close(pipeline_loss, packed_loss)
    torch.testing.assert_close(rowwise_logp.grad, packed_logp.grad)
    torch.testing.assert_close(pipeline_logp.grad, packed_logp.grad)


def test_dsv4_indexer_tuning_is_exported_to_tilelang(monkeypatch) -> None:
    monkeypatch.delenv("V4_INDEXER_BLOCK_N", raising=False)
    monkeypatch.delenv("V4_INDEXER_NUM_STAGES", raising=False)

    assert hasattr(
        megatron_lumen_dsv4_engine, "_configure_dsv4_indexer_environment"
    )
    megatron_lumen_dsv4_engine._configure_dsv4_indexer_environment(
        {"v4_indexer_block_n": 64, "v4_indexer_num_stages": 1}
    )

    assert os.environ["V4_INDEXER_BLOCK_N"] == "64"
    assert os.environ["V4_INDEXER_NUM_STAGES"] == "1"


def test_dsv4_sequence_alignment_includes_compressor_ratios() -> None:
    assert hasattr(megatron_lumen_dsv4_engine, "_dsv4_sequence_alignment")
    assert (
        megatron_lumen_dsv4_engine._dsv4_sequence_alignment(
            tensor_parallel_size=4, compress_ratios=[0, 4, 128]
        )
        == 128
    )


def test_shared_packing_helpers() -> None:
    engine = MegatronBaseEngine({}, {}, {}, "")

    bins = engine._build_bins([7, 5, 4, 2], budget=9)
    assert sorted(index for group in bins for index in group) == [0, 1, 2, 3]
    assert all(sum([7, 5, 4, 2][index] for index in group) <= 9 for group in bins)

    start, length = engine._real_block(torch.tensor([0, 0, 1, 1, 1, 0]))
    assert (start, length) == (2, 3)
    assert engine._real_block(torch.zeros(4, dtype=torch.long)) == (0, 0)
