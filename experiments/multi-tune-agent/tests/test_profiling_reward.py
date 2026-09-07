"""Unit tests for profiling-based reward."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lumenrl.rewards.profiling_reward import (
    ProfilingMetrics,
    ProfilingRewardConfig,
    compute_profiling_reward,
    parse_rocprof_output,
    profiling_reward_batch,
)


class TestParseRocprofOutput:
    def test_parse_duration(self):
        text = "DurationNs: 12345\nSQ_INSTS_VALU: 100\nSQ_WAVES: 50"
        m = parse_rocprof_output(text)
        assert m.kernel_duration_ns == 12345
        assert m.valu_instructions == 100
        assert m.total_wavefronts == 50
        assert m.has_data

    def test_parse_bandwidth(self):
        text = "MemoryBW: 450.5\nOccupancy: 75.0"
        m = parse_rocprof_output(text)
        assert m.memory_bandwidth_gb_s == pytest.approx(450.5)
        assert m.occupancy_pct == pytest.approx(75.0)

    def test_parse_empty(self):
        m = parse_rocprof_output("")
        assert not m.has_data

    def test_parse_csv_style(self):
        text = "duration_ns=5000,bandwidth=200.0,occupancy_pct=60.0"
        m = parse_rocprof_output(text)
        assert m.kernel_duration_ns == 5000


class TestComputeProfilingReward:
    def test_improved_bandwidth_positive_reward(self):
        baseline = ProfilingMetrics(
            kernel_duration_ns=10000,
            memory_bandwidth_gb_s=100.0,
            occupancy_pct=50.0,
            valu_instructions=1000,
            vmem_instructions=500,
        )
        candidate = ProfilingMetrics(
            kernel_duration_ns=8000,
            memory_bandwidth_gb_s=150.0,
            occupancy_pct=60.0,
            valu_instructions=1000,
            vmem_instructions=500,
        )
        result = compute_profiling_reward(baseline, candidate)
        assert result["bandwidth"] > 0
        assert result["occupancy"] > 0
        assert result["total"] > 0

    def test_removed_computation_penalty(self):
        baseline = ProfilingMetrics(
            kernel_duration_ns=10000,
            memory_bandwidth_gb_s=100.0,
            valu_instructions=1000,
            vmem_instructions=500,
        )
        candidate = ProfilingMetrics(
            kernel_duration_ns=2000,
            memory_bandwidth_gb_s=20.0,
            valu_instructions=100,  # 90% reduction
            vmem_instructions=50,
        )
        result = compute_profiling_reward(baseline, candidate)
        assert result["instruction_efficiency"] < 0

    def test_no_data_returns_zero(self):
        result = compute_profiling_reward(ProfilingMetrics(), ProfilingMetrics())
        assert result["total"] == 0.0

    def test_same_metrics_near_zero(self):
        m = ProfilingMetrics(
            kernel_duration_ns=10000,
            memory_bandwidth_gb_s=100.0,
            occupancy_pct=50.0,
            valu_instructions=1000,
        )
        result = compute_profiling_reward(m, m)
        assert result["bandwidth"] == pytest.approx(0.0)
        assert result["occupancy"] == pytest.approx(0.0)


class TestProfilingRewardBatch:
    def test_batch_returns_correct_shape(self):
        base = ["DurationNs: 10000\nMemoryBW: 100"] * 3
        cand = ["DurationNs: 8000\nMemoryBW: 150"] * 3
        rewards, details = profiling_reward_batch(base, cand)
        assert rewards.shape == (3,)
        assert len(details) == 3
