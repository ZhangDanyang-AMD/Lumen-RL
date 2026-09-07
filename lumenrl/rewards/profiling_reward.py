"""Profiling-based reward using rocprof hardware counters.

Complements correctness + speedup rewards with hardware profiling signals
to prevent lazy optimisation (reward hacking).  Uses rocprof output to
assess bandwidth utilisation, occupancy, and instruction efficiency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)


@dataclass
class ProfilingMetrics:
    """Parsed hardware profiling metrics from rocprof output."""

    kernel_duration_ns: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    occupancy_pct: float = 0.0
    valu_instructions: int = 0
    vmem_instructions: int = 0
    lds_instructions: int = 0
    total_wavefronts: int = 0

    @property
    def has_data(self) -> bool:
        return self.kernel_duration_ns > 0.0


@dataclass
class ProfilingRewardConfig:
    bandwidth_weight: float = 0.3
    occupancy_weight: float = 0.3
    instruction_efficiency_weight: float = 0.4
    max_reward: float = 1.0
    min_instruction_ratio: float = 0.3


_DURATION_RE = re.compile(
    r"(?:DurationNs|KernelDuration|duration_ns)\s*[,:=]\s*(\d+)", re.I
)
_BW_RE = re.compile(
    r"(?:MemoryBW|FETCH_SIZE|TCC_HIT|bandwidth)\s*[,:=]\s*([\d.]+)", re.I
)
_OCC_RE = re.compile(
    r"(?:Occupancy|occupancy_pct|GRBM_GUI_ACTIVE)\s*[,:=]\s*([\d.]+)", re.I
)
_VALU_RE = re.compile(r"(?:SQ_INSTS_VALU|valu_insts)\s*[,:=]\s*(\d+)", re.I)
_VMEM_RE = re.compile(r"(?:SQ_INSTS_VMEM|vmem_insts)\s*[,:=]\s*(\d+)", re.I)
_LDS_RE = re.compile(r"(?:SQ_INSTS_LDS|lds_insts)\s*[,:=]\s*(\d+)", re.I)
_WAVE_RE = re.compile(r"(?:SQ_WAVES|total_wavefronts)\s*[,:=]\s*(\d+)", re.I)


def parse_rocprof_output(text: str) -> ProfilingMetrics:
    """Extract profiling metrics from rocprof text output (CSV or key=value)."""
    m = ProfilingMetrics()

    match = _DURATION_RE.search(text)
    if match:
        m.kernel_duration_ns = float(match.group(1))

    match = _BW_RE.search(text)
    if match:
        m.memory_bandwidth_gb_s = float(match.group(1))

    match = _OCC_RE.search(text)
    if match:
        m.occupancy_pct = float(match.group(1))

    match = _VALU_RE.search(text)
    if match:
        m.valu_instructions = int(match.group(1))

    match = _VMEM_RE.search(text)
    if match:
        m.vmem_instructions = int(match.group(1))

    match = _LDS_RE.search(text)
    if match:
        m.lds_instructions = int(match.group(1))

    match = _WAVE_RE.search(text)
    if match:
        m.total_wavefronts = int(match.group(1))

    return m


def compute_profiling_reward(
    baseline: ProfilingMetrics,
    candidate: ProfilingMetrics,
    config: ProfilingRewardConfig | None = None,
) -> dict[str, float]:
    """Compute profiling-based reward components.

    Returns dict with ``bandwidth``, ``occupancy``, ``instruction_efficiency``,
    and ``total`` reward values.
    """
    cfg = config or ProfilingRewardConfig()

    if not baseline.has_data or not candidate.has_data:
        return {
            "bandwidth": 0.0,
            "occupancy": 0.0,
            "instruction_efficiency": 0.0,
            "total": 0.0,
        }

    # Bandwidth: reward for improving bandwidth utilisation
    if baseline.memory_bandwidth_gb_s > 0:
        bw_ratio = candidate.memory_bandwidth_gb_s / baseline.memory_bandwidth_gb_s
        r_bw = min(1.0, max(-1.0, bw_ratio - 1.0))
    else:
        r_bw = 0.0

    # Occupancy: reward for maintaining/improving occupancy
    if baseline.occupancy_pct > 0:
        occ_ratio = candidate.occupancy_pct / baseline.occupancy_pct
        r_occ = min(1.0, max(-1.0, occ_ratio - 1.0))
    else:
        r_occ = 0.0

    # Instruction efficiency: penalise if total instructions dropped
    # significantly (suggests removed computation, not optimisation)
    baseline_insts = baseline.valu_instructions + baseline.vmem_instructions + baseline.lds_instructions
    candidate_insts = candidate.valu_instructions + candidate.vmem_instructions + candidate.lds_instructions

    if baseline_insts > 0:
        inst_ratio = candidate_insts / baseline_insts
        if inst_ratio < cfg.min_instruction_ratio:
            r_inst = -1.0
        else:
            r_inst = min(1.0, max(-0.5, 1.0 - abs(inst_ratio - 1.0)))
    else:
        r_inst = 0.0

    total = (
        cfg.bandwidth_weight * r_bw
        + cfg.occupancy_weight * r_occ
        + cfg.instruction_efficiency_weight * r_inst
    )
    total = max(-cfg.max_reward, min(cfg.max_reward, total))

    return {
        "bandwidth": r_bw,
        "occupancy": r_occ,
        "instruction_efficiency": r_inst,
        "total": total,
    }


def profiling_reward_batch(
    baseline_texts: list[str],
    candidate_texts: list[str],
    config: ProfilingRewardConfig | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Batch profiling reward computation.

    Returns (rewards [B], details [list of dicts]).
    """
    rewards = []
    details = []
    for base_text, cand_text in zip(baseline_texts, candidate_texts):
        baseline = parse_rocprof_output(base_text)
        candidate = parse_rocprof_output(cand_text)
        result = compute_profiling_reward(baseline, candidate, config)
        rewards.append(result["total"])
        details.append(result)
    return torch.tensor(rewards, dtype=torch.float32), details
