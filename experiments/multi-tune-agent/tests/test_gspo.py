"""Unit tests for GSPO algorithm: loss function, advantage estimator, and algorithm class."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure lumenrl is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lumenrl.algorithms.loss_functions import gspo_loss


class TestGSPOLoss:
    """Tests for ``gspo_loss``."""

    def test_zero_advantage_zero_loss(self):
        logp = torch.zeros(4, 10)
        old_logp = torch.zeros(4, 10)
        adv = torch.zeros(4, 10)
        loss = gspo_loss(logp, old_logp, adv, clip_ratio=0.2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_advantage_negative_loss(self):
        logp = torch.zeros(4, 10)
        old_logp = torch.zeros(4, 10)
        adv = torch.ones(4, 10)
        loss = gspo_loss(logp, old_logp, adv, clip_ratio=0.2)
        # ratio=1.0, clipped_ratio=1.0, so loss = max(-1*1, -1*1) = -1
        assert loss.item() == pytest.approx(-1.0, abs=1e-6)

    def test_clipping_limits_ratio(self):
        logp = torch.full((4, 10), 0.5)
        old_logp = torch.zeros(4, 10)
        adv = torch.ones(4, 10)
        mask = torch.ones(4, 10)
        loss = gspo_loss(logp, old_logp, adv, clip_ratio=0.2, mask=mask)
        # seq_log_ratio = 0.5, ratio = exp(0.5) ≈ 1.648
        # clipped_ratio = 1.2 (clamped)
        # For positive advantage: max(-adv*ratio, -adv*clipped) = max(-1.648, -1.2) = -1.2
        assert loss.item() == pytest.approx(-1.2, abs=1e-2)

    def test_mask_excludes_padding(self):
        logp = torch.zeros(2, 10)
        old_logp = torch.zeros(2, 10)
        adv = torch.ones(2, 10)
        mask = torch.zeros(2, 10)
        mask[:, :5] = 1.0
        loss = gspo_loss(logp, old_logp, adv, clip_ratio=0.2, mask=mask)
        assert loss.item() == pytest.approx(-1.0, abs=1e-6)

    def test_sequence_level_not_token_level(self):
        """GSPO should aggregate to sequence-level ratio, not per-token."""
        B, T = 2, 8
        logp = torch.randn(B, T) * 0.1
        old_logp = torch.randn(B, T) * 0.1
        adv = torch.randn(B, T)
        mask = torch.ones(B, T)

        loss = gspo_loss(logp, old_logp, adv, clip_ratio=0.2, mask=mask)
        assert loss.dim() == 0  # scalar


class TestGSPOAdvantageEstimator:
    """Tests for the ``gspo`` advantage estimator."""

    def test_group_normalization(self):
        from lumenrl.algorithms.advantage_estimators import ADV_ESTIMATOR_REGISTRY

        assert "gspo" in ADV_ESTIMATOR_REGISTRY

        config = MagicMock()
        config.algorithm.gspo.num_generations = 4

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        batch = MagicMock()
        batch.tensors = {"rewards": rewards}

        result = ADV_ESTIMATOR_REGISTRY["gspo"](batch, config)
        adv = result.tensors["advantages"]

        assert adv.shape == (8,)
        # Group 1: [1,2,3,4] → mean=2.5, std=~1.118
        # Group 2: [5,6,7,8] → mean=6.5, std=~1.118
        # Within each group, advantages should sum to ~0
        group1 = adv[:4]
        group2 = adv[4:]
        assert group1.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert group2.mean().item() == pytest.approx(0.0, abs=1e-5)


class TestGSPOAlgorithm:
    """Tests for ``GSPOAlgorithm``."""

    def test_registry_registration(self):
        from lumenrl.core.registry import ALGORITHM_REGISTRY

        assert "gspo" in ALGORITHM_REGISTRY._registry

    def test_compute_loss_runs(self):
        from lumenrl.algorithms.gspo import GSPOAlgorithm

        config = MagicMock()
        config.algorithm.gspo.clip_ratio = 0.2
        config.algorithm.gspo.kl_coeff = 0.0

        algo = GSPOAlgorithm(config)

        batch = MagicMock()
        batch.tensors = {
            "log_probs": torch.randn(4, 16),
            "old_log_probs": torch.randn(4, 16),
            "advantages": torch.randn(4, 16),
            "response_mask": torch.ones(4, 16),
        }

        loss, metrics = algo.compute_loss(batch)
        assert loss.dim() == 0
        assert "loss_pg" in metrics
        assert "loss_total" in metrics
