"""RL trajectory recorder for kernel GRPO training.

Records rollout results for analysis: prompts, responses, rewards,
compile/correctness/performance outcomes, per operator and lane.
Compatible with multi-tune-agent's trajectory format.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RolloutRecord:
    """Single rollout (one response to one prompt)."""
    step: int
    prompt_idx: int
    generation_idx: int
    task_id: str
    family: str
    lane: str
    reward: float
    compile_ok: bool = False
    correct_ok: bool = False
    speedup: float = 0.0
    perf_ms: float = 0.0
    error_stage: str = ""
    error_msg: str = ""
    response_len: int = 0
    gen_time_s: float = 0.0


@dataclass
class StepSummary:
    """Aggregated metrics for one training step."""
    step: int
    num_prompts: int = 0
    num_generations: int = 0
    mean_reward: float = 0.0
    compile_rate: float = 0.0
    correct_rate: float = 0.0
    pass_rate: float = 0.0
    mean_speedup: float = 0.0
    best_speedup: float = 0.0
    by_family: dict = field(default_factory=dict)
    by_lane: dict = field(default_factory=dict)
    policy_loss: float = 0.0
    lr: float = 0.0
    wall_time_s: float = 0.0


class RLTrajectoryRecorder:
    """Records and analyzes RL training trajectories."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[RolloutRecord] = []
        self.step_summaries: list[StepSummary] = []
        self._step_start = 0.0

        self._rollout_file = open(self.output_dir / "rollouts.jsonl", "a")
        self._summary_file = open(self.output_dir / "step_summaries.jsonl", "a")

    def begin_step(self, step: int):
        self._step_start = time.time()
        self._current_step = step

    def record_rollout(self, record: RolloutRecord):
        self.records.append(record)
        self._rollout_file.write(json.dumps(asdict(record)) + "\n")
        self._rollout_file.flush()

    def end_step(self, step: int, policy_loss: float = 0.0, lr: float = 0.0):
        step_records = [r for r in self.records if r.step == step]
        if not step_records:
            return

        n = len(step_records)
        summary = StepSummary(
            step=step,
            num_prompts=len(set(r.prompt_idx for r in step_records)),
            num_generations=n,
            mean_reward=sum(r.reward for r in step_records) / n,
            compile_rate=sum(r.compile_ok for r in step_records) / n,
            correct_rate=sum(r.correct_ok for r in step_records) / n,
            pass_rate=sum(r.speedup > 1.0 for r in step_records) / n,
            mean_speedup=sum(r.speedup for r in step_records if r.speedup > 0) / max(1, sum(r.speedup > 0 for r in step_records)),
            best_speedup=max((r.speedup for r in step_records), default=0.0),
            policy_loss=policy_loss,
            lr=lr,
            wall_time_s=time.time() - self._step_start,
        )

        # Per-family breakdown
        family_stats = defaultdict(lambda: {"total": 0, "compiled": 0, "correct": 0, "rewards": []})
        for r in step_records:
            f = family_stats[r.family]
            f["total"] += 1
            f["compiled"] += r.compile_ok
            f["correct"] += r.correct_ok
            f["rewards"].append(r.reward)
        summary.by_family = {
            k: {"total": v["total"], "compiled": v["compiled"], "correct": v["correct"],
                "mean_reward": sum(v["rewards"]) / len(v["rewards"])}
            for k, v in family_stats.items()
        }

        # Per-lane breakdown
        lane_stats = defaultdict(lambda: {"total": 0, "compiled": 0, "correct": 0})
        for r in step_records:
            l = lane_stats[r.lane]
            l["total"] += 1
            l["compiled"] += r.compile_ok
            l["correct"] += r.correct_ok
        summary.by_lane = dict(lane_stats)

        self.step_summaries.append(summary)
        self._summary_file.write(json.dumps(asdict(summary)) + "\n")
        self._summary_file.flush()

        # Log
        logger.info(
            "Step %d: reward=%.2f compile=%.0f%% correct=%.0f%% speedup=%.2f loss=%.4f (%.0fs)",
            step, summary.mean_reward, summary.compile_rate * 100,
            summary.correct_rate * 100, summary.mean_speedup,
            policy_loss, summary.wall_time_s,
        )

    def save_final_report(self):
        """Save comprehensive training report."""
        all_records = self.records
        if not all_records:
            return

        report = {
            "total_rollouts": len(all_records),
            "total_steps": len(self.step_summaries),
            "final_compile_rate": sum(r.compile_ok for r in all_records) / len(all_records),
            "final_correct_rate": sum(r.correct_ok for r in all_records) / len(all_records),
            "final_mean_reward": sum(r.reward for r in all_records) / len(all_records),
            "best_speedup_ever": max((r.speedup for r in all_records), default=0.0),
            "reward_trajectory": [s.mean_reward for s in self.step_summaries],
            "compile_trajectory": [s.compile_rate for s in self.step_summaries],
            "correct_trajectory": [s.correct_rate for s in self.step_summaries],
        }

        with open(self.output_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Saved training report to %s", self.output_dir / "training_report.json")

    def close(self):
        self._rollout_file.close()
        self._summary_file.close()
        self.save_final_report()
