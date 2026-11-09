"""Unit tests for reward hacking detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hacking_detection import (
    HackingDetector,
    compute_hacking_penalty,
    detect_copy_paste,
    detect_hardcoded_output,
    detect_lazy_deletion,
    detect_test_gaming,
    detect_timeout,
)


class TestLazyDeletion:
    def test_detects_large_deletion(self):
        baseline = "\n".join(f"line {i}" for i in range(100))
        candidate = "\n".join(f"line {i}" for i in range(30))
        result = detect_lazy_deletion(baseline, candidate, threshold=0.5)
        assert result.detected
        assert result.severity < 0

    def test_no_detection_for_minor_change(self):
        baseline = "\n".join(f"line {i}" for i in range(100))
        candidate = "\n".join(f"line {i}" for i in range(90))
        result = detect_lazy_deletion(baseline, candidate, threshold=0.5)
        assert not result.detected

    def test_empty_baseline_not_detected(self):
        result = detect_lazy_deletion("", "some code")
        assert not result.detected


class TestHardcodedOutput:
    def test_detects_large_tensor_literal(self):
        src = 'output = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])'
        result = detect_hardcoded_output(src)
        assert result.detected

    def test_detects_zeros_return(self):
        src = "return torch.zeros(batch_size, hidden_dim)"
        result = detect_hardcoded_output(src)
        assert result.detected

    def test_normal_code_not_detected(self):
        src = """
def optimised_kernel(x, w):
    return torch.matmul(x, w)
"""
        result = detect_hardcoded_output(src)
        assert not result.detected


class TestTestGaming:
    def test_detects_suspiciously_fast(self):
        evaluation = {
            "correct": True,
            "candidate_ms": {"case_0": 0.01},
        }
        baseline_ms = {"case_0": 10.0}
        result = detect_test_gaming(evaluation, baseline_ms, suspicion_threshold=0.1)
        assert result.detected

    def test_normal_speedup_not_detected(self):
        evaluation = {
            "correct": True,
            "candidate_ms": {"case_0": 5.0},
        }
        baseline_ms = {"case_0": 10.0}
        result = detect_test_gaming(evaluation, baseline_ms)
        assert not result.detected

    def test_incorrect_not_detected(self):
        evaluation = {
            "correct": False,
            "candidate_ms": {"case_0": 0.01},
        }
        result = detect_test_gaming(evaluation, {"case_0": 10.0})
        assert not result.detected


class TestCopyPaste:
    def test_detects_identical(self):
        src = "def kernel(): pass"
        result = detect_copy_paste(src, src)
        assert result.detected
        assert result.severity == 0.0

    def test_different_not_detected(self):
        result = detect_copy_paste("def a(): pass", "def b(): pass")
        assert not result.detected


class TestTimeout:
    def test_detects_timeout(self):
        evaluation = {
            "correctness": {"timed_out": True},
        }
        result = detect_timeout(evaluation)
        assert result.detected

    def test_no_timeout(self):
        evaluation = {
            "correctness": {"timed_out": False},
            "performance": {"timed_out": False},
        }
        result = detect_timeout(evaluation)
        assert not result.detected


class TestHackingDetector:
    def test_full_check_clean(self):
        detector = HackingDetector()
        report = detector.check(
            baseline_src="def kernel():\n    return x * w",
            candidate_src="def kernel():\n    return torch.matmul(x, w)",
            evaluation={"compiled": True, "correct": True, "speedup_geomean": 1.5},
            baseline_ms={"case_0": 10.0},
        )
        assert not report.any_detected

    def test_full_check_with_hacking(self):
        detector = HackingDetector()
        baseline = "\n".join(f"line {i}" for i in range(100))
        candidate = "return torch.zeros(1)"
        report = detector.check(
            baseline_src=baseline,
            candidate_src=candidate,
            evaluation={"compiled": True, "correct": True, "candidate_ms": {"c": 0.001}},
            baseline_ms={"c": 10.0},
        )
        assert report.any_detected
        assert report.total_penalty < 0


class TestComputeHackingPenalty:
    def test_penalty_is_negative(self):
        from hacking_detection import HackingDetection, HackingReport

        report = HackingReport([
            HackingDetection("test", True, -2.0),
            HackingDetection("test2", True, -1.0),
        ])
        assert compute_hacking_penalty(report) == -3.0

    def test_no_detection_zero_penalty(self):
        from hacking_detection import HackingDetection, HackingReport

        report = HackingReport([
            HackingDetection("test", False, -2.0),
        ])
        assert compute_hacking_penalty(report) == 0.0
