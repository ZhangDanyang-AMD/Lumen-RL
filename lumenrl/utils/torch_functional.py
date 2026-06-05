# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 LumenRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/utils/torch_functional.py

"""Small torch utilities for RL training: log-probs, entropy, sum-pi-squared."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::logprobs_from_logits_naive
def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities via log_softmax + gather.

    Args:
        logits: (..., vocab_size).
        labels: (...,) indices.

    Returns:
        Log-probabilities with shape (...,).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::entropy_from_logits
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy H(p) = -sum(p * log p) from unnormalised logits.

    Uses the numerically stable formula:
        H = logsumexp(logits) - sum(softmax(logits) * logits)

    Args:
        logits: (..., vocab_size).

    Returns:
        Entropy values with shape (...,).
    """
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::calculate_sum_pi_squared_from_logits
def calculate_sum_pi_squared_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute sum of squared probabilities from logits.

    Formula: sum(pi^2) = exp(logsumexp(2*logits) - 2*logsumexp(logits))

    Used for optimal-baseline advantage estimators.

    Args:
        logits: (..., vocab_size).

    Returns:
        Sum-pi-squared with shape (...,).
    """
    return torch.exp(
        torch.logsumexp(2.0 * logits, dim=-1) - 2.0 * torch.logsumexp(logits, dim=-1)
    )
