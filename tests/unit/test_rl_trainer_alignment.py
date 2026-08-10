from __future__ import annotations

import sys
from types import SimpleNamespace

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


def test_compute_rewards_full_decodes_only_left_padded_response() -> None:
    decoded_ids: list[list[int]] = []

    class Tokenizer:
        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            decoded_ids.append(token_ids.tolist())
            return r"The answer is \boxed{437}."

    trainer = object.__new__(RLTrainer)
    trainer._rank = 0
    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer._is_distributed = False

    _, responses, _ = trainer._compute_rewards_full(
        sequences=torch.tensor([[0, 0, 11, 12, 21, 22]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
        prompt_lengths=[2],
        gts_expanded=["437"],
    )

    assert decoded_ids == [[21, 22]]
    assert responses == [r"The answer is \boxed{437}."]
