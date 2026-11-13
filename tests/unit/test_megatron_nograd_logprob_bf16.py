"""No-grad log-probs (old_log_probs pass) take the model's bf16 logits as-is.

engine_compute_log_probs upcast the whole packed [T, V] logits to fp32 and
divided by temperature before the chunked helper, a 12 GiB block at T=21.5K,
V=152K, which OOMed under the actor allocator cap. The helper upcasts and
applies temperature one chunk at a time, so results must be bit-identical.
"""

from __future__ import annotations

import pytest
import torch

from lumenrl.engine.training.megatron_base_engine import MegatronBaseEngine


def _engine(chunk):
    e = object.__new__(MegatronBaseEngine)
    e._logprob_chunk_size = chunk
    return e


@pytest.mark.parametrize("chunk", [3, 0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_nograd_logprob_entropy_on_bf16_matches_upcast_then_divide(chunk, temperature) -> None:
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(11, 40, generator=g) * 4).to(torch.bfloat16)
    targets = torch.randint(0, 40, (11,), generator=g)
    e = _engine(chunk)

    old_lp, old_ent = e._logprob_entropy_nograd(logits.float() / temperature, targets, True)
    new_lp, new_ent = e._logprob_entropy_nograd(logits, targets, True, temperature=temperature)

    assert new_lp.dtype == torch.float32 and new_ent.dtype == torch.float32
    assert torch.equal(new_lp, old_lp)
    assert torch.equal(new_ent, old_ent)


def test_nograd_logprob_without_entropy() -> None:
    logits = torch.randn(5, 9).to(torch.bfloat16)
    targets = torch.randint(0, 9, (5,))
    lp, ent = _engine(2)._logprob_entropy_nograd(logits, targets, False, temperature=0.9)
    assert ent is None
    torch.testing.assert_close(
        lp, torch.log_softmax(logits.float() / 0.9, -1).gather(-1, targets[:, None])[:, 0],
    )
