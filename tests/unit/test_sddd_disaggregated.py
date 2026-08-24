from pathlib import Path

import torch

from lumenrl.core.config import LumenRLConfig
from lumenrl.engine.inference.atom_teacher_ray import AtomTeacherRayActor


def test_generated_masks_only_score_teacher_continuation():
    attention, loss = AtomTeacherRayActor._build_generated_masks(
        seq_lens=torch.tensor([5, 3]),
        prompt_lens=torch.tensor([2, 1]),
        total_len=5,
    )

    assert attention.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ]
    assert loss.tolist() == [
        [0, 0, 1, 1, 1],
        [0, 1, 1, 0, 0],
    ]


def test_prefill_batch_preserves_dataset_loss_mask():
    class EngineStub:
        def extract_hidden_state_manifest(self, input_ids, attention_mask):
            assert input_ids.tolist() == [[1, 2, 3]]
            assert attention_mask.tolist() == [[1, 1, 1]]
            return {"mooncake_keys": ["row-0"]}

    actor = object.__new__(AtomTeacherRayActor)
    actor._generate_mode = "prefill"
    actor.replica_index = 2
    actor._engine = EngineStub()
    loss_mask = torch.tensor([[0, 1, 1]])

    manifest = actor.process_prefill_batch(
        7,
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[1, 1, 1]]),
        loss_mask,
    )

    assert manifest["batch_id"] == 7
    assert manifest["replica_index"] == 2
    assert torch.equal(manifest["loss_mask"], loss_mask)


def test_k3_config_enables_five_node_streaming_topology():
    root = Path(__file__).resolve().parents[2]
    config = LumenRLConfig.from_yaml(
        root / "examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml"
    )

    spec = config.algorithm.spec_distill
    assert spec.sequential_mode == "streaming_disaggregated"
    assert spec.teacher_replicas == 4
    assert spec.stream_prefetch_batches >= spec.teacher_replicas
    assert config.policy.train_global_batch_size == 128
    assert config.policy.max_total_sequence_length == 8192
    assert config.policy.learning_rate == 5.0e-5
    assert config.mooncake.protocol == "rdma"
    assert config.mooncake.enable_gpu_direct is False
    assert config.mooncake.enable_hard_pin is True
    assert config.algorithm.teacher.generate_mode == "prefill"
    assert config.algorithm.teacher.atom["max_model_len"] == 8192
    assert config.algorithm.teacher.atom["kv_cache_dtype"] == "fp8"
    assert config.dataset.max_prompt_tokens == 0
    assert config.dataset.last_turn_loss_only == "true"
    assert config.dataset.min_loss_tokens == 14
    assert "ATOM_regen_seeklight_kimi_mtp" in config.reward.dataset
