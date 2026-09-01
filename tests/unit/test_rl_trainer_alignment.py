from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

from lumenrl.trainer import rl_trainer
from lumenrl.trainer.rl_trainer import RLTrainer


def test_validation_uses_configured_miles_sampling(monkeypatch) -> None:
    from lumenrl.rewards import math_reward

    rollout_calls = []

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return r"Answer: \boxed{2}"

    class RolloutEngine:
        @staticmethod
        def sleep():
            return None

    def rollout(prompts, *, num_generations, sampling_params):
        rollout_calls.append((prompts, num_generations, sampling_params))
        batch = len(prompts) * num_generations
        return (
            torch.tensor([[11, 22]] * batch),
            torch.ones(batch, 2, dtype=torch.long),
            [1] * batch,
            None,
            None,
        )

    monkeypatch.setattr(
        math_reward,
        "compute_math_reward",
        lambda responses, ground_truths: (
            torch.ones(len(responses)),
            [{"acc": True}] * len(responses),
        ),
    )

    trainer = object.__new__(RLTrainer)
    trainer._val_dataset = [{"prompt": "1+1", "label": "2"}]
    trainer._rank = 0
    trainer._tokenizer = Tokenizer()
    trainer._ray_vllm_engine = RolloutEngine()
    trainer._actor_wg = SimpleNamespace(num_workers=1)
    trainer._use_vllm = False
    trainer._use_atom = False
    trainer._is_distributed = False
    trainer.global_step = 4
    trainer.config = SimpleNamespace(
        eval=SimpleNamespace(
            num_samples=0,
            num_generations=8,
            temperature=0.8,
            top_p=0.7,
            top_k=-1,
        ),
        val_batch_size=0,
        policy=SimpleNamespace(max_response_length=4, max_total_sequence_length=8),
        logger=SimpleNamespace(num_val_samples_to_print=0),
    )
    trainer._extract_prompt_gt = lambda sample, **_: (sample["prompt"], sample["label"])
    trainer._rollout_with_ray_vllm = rollout

    metrics = trainer.run_validation()

    assert rollout_calls == [
        (
            ["1+1"],
            8,
            {
                "max_tokens": 4,
                "temperature": 0.8,
                "top_p": 0.7,
                "top_k": -1,
            },
        )
    ]
    assert metrics["val-core/acc/mean@8"] == 1.0
    assert metrics["val/num_samples"] == 8.0


def test_dsv4_actor_model_parallel_size_uses_megatron_topology() -> None:
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            training_backend="megatron_lumen_dsv4",
            training=SimpleNamespace(
                megatron_cfg=SimpleNamespace(
                    tensor_model_parallel_size=4,
                    pipeline_model_parallel_size=4,
                    context_parallel_size=1,
                )
            ),
        )
    )

    assert trainer._compute_actor_mp() == 16


def test_response_token_ids_respect_left_padding_and_prompt_length() -> None:
    helper = getattr(rl_trainer, "_response_token_ids", None)
    assert helper is not None

    sequence = torch.tensor([0, 0, 11, 12, 21, 22])
    attention_mask = torch.tensor([0, 0, 1, 1, 1, 1])

    response = helper(sequence, attention_mask, prompt_length=2)

    assert response.tolist() == [21, 22]


def test_response_token_ids_exclude_right_padding() -> None:
    helper = getattr(rl_trainer, "_response_token_ids", None)
    assert helper is not None

    sequence = torch.tensor([11, 12, 21, 22, 0, 0])
    attention_mask = torch.tensor([1, 1, 1, 1, 0, 0])

    response = helper(sequence, attention_mask, prompt_length=2)

    assert response.tolist() == [21, 22]


def test_compute_rewards_full_logs_response_diagnostics(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lumenrl.trainer.rl_trainer")
    decoded_ids: list[list[int]] = []

    class Tokenizer:
        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            decoded_ids.append(token_ids.tolist())
            return "Answer: " + r"\boxed{437}"

    trainer = object.__new__(RLTrainer)
    trainer._rank = 0
    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer._is_distributed = False
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(max_response_length=2)
    )

    _, responses, _ = trainer._compute_rewards_full(
        sequences=torch.tensor([[0, 0, 11, 12, 21, 22]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
        prompt_lengths=[2],
        gts_expanded=["437"],
    )

    assert decoded_ids == [[21, 22]]
    assert responses == ["Answer: " + r"\boxed{437}"]
    assert "cap_hits=1/1" in caplog.text
    assert "invalid_format=0/1" in caplog.text


def test_extract_prompt_gt_uses_dsv4_encoder_and_label(monkeypatch) -> None:
    calls = []

    encoding = ModuleType("vllm.tokenizers.deepseek_v4_encoding")

    def encode_messages(messages, *, thinking_mode):
        calls.append((messages, thinking_mode))
        return (
            "<｜begin▁of▁sentence｜><｜User｜>"
            + messages[0]["content"]
            + "<｜Assistant｜><think>"
        )

    encoding.encode_messages = encode_messages
    tokenizers = ModuleType("vllm.tokenizers")
    tokenizers.deepseek_v4_encoding = encoding
    vllm = ModuleType("vllm")
    vllm.tokenizers = tokenizers
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers)
    monkeypatch.setitem(
        sys.modules, "vllm.tokenizers.deepseek_v4_encoding", encoding
    )

    class DeepSeekV4Tokenizer:
        chat_template = None

        @staticmethod
        def convert_tokens_to_ids(token):
            return {
                "<｜User｜>": 128803,
                "<｜Assistant｜>": 128804,
            }.get(token)

        @staticmethod
        def apply_chat_template(*args, **kwargs):
            raise ValueError("DeepSeek-V4 does not ship a Jinja chat template")

    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = DeepSeekV4Tokenizer()
    messages = [{"role": "user", "content": "Solve 1+1."}]

    prompt, ground_truth = trainer._extract_prompt_gt(
        {"prompt": messages, "label": "2"}
    )

    assert calls == [(messages, "thinking")]
    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>"
        "Solve 1+1.<｜Assistant｜><think>"
    )
    assert ground_truth == "2"


def test_extract_prompt_gt_adds_validation_answer_format() -> None:
    class Tokenizer:
        chat_template = "configured"

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return messages[0]["content"]

    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = Tokenizer()

    prompt, ground_truth = trainer._extract_prompt_gt(
        {"prompt": [{"role": "user", "content": "Solve 1+1."}], "label": "2"},
        ensure_answer_format=True,
    )

    assert prompt == (
        "Solve 1+1.\n\n"
        r"Put the final answer on its own line as: Answer: \boxed{answer}."
    )
    assert ground_truth == "2"


@pytest.mark.parametrize("old_seq_len", [6, 7])
def test_response_position_metrics_isolate_late_token_drift(old_seq_len: int) -> None:
    """Rollout/mask are [B, S-1]; Megatron old_log_probs may be width S (7 here)."""
    old_logp = torch.zeros(2, old_seq_len)
    rollout_logp = torch.tensor([
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.0],
        [0.0, 0.0, 0.5, 0.6, 0.0, 0.0],
    ])
    response_mask = torch.tensor([
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
    ])

    got = RLTrainer._mismatch_by_response_position(
        old_logp,
        rollout_logp,
        response_mask,
        bucket_edges=(2, 4),
    )

    assert got["rollout_corr/pos_0_1/tokens"] == 4
    assert got["rollout_corr/pos_0_1/kl"] == pytest.approx(0.35)
    assert got["rollout_corr/pos_2_3/tokens"] == 2
    assert got["rollout_corr/pos_2_3/kl"] == pytest.approx(0.35)
