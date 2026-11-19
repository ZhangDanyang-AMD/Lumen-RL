"""MMKCA gfx950 GEMM schedule environment for LumenRL AgenticRL.

One episode scores one regime (prefill or decode) and one A/B type (bf16, or
fp8 e4m3 with 1x128 / 128x128 block scales). The agent submits typed
``mmkca.gemm`` schedules; MMKCA verifies, lowers to MFMA, checks correctness on
the GPU and times against the shared baseline. Step, reward, duplicate and
budget semantics are those of :class:`MicakeGemmEnvironment`.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from lumenrl.agentic.core.protocols import EnvironmentCapabilities, ResetResult
from sandbox.micake.environment import FIDELITY_LEVELS, MicakeGemmEnvironment
from sandbox.micake.tool_runtime import tool_schemas
from sandbox.mmkca.evaluator import UNKNOWN_STAGES, MmkcaWorkerClient

BACKEND_NAME = "mmkca_gemm"
PORTFOLIOS = ("decode", "prefill")
SPLITS = ("train", "heldout")
INPUT_DTYPES = {
    "bf16": {"name": "BF16", "element_bytes": 2},
    "fp8": {"name": "FP8 e4m3 blockwise", "element_bytes": 1},
}

_RULES = """- When history is empty, submit the packet's opening config. Prefill opens at 128x128. Decode opens at tile_m 32, tile_n 128, split_k 2.
- After the first measurement, change one field to an adjacent domain value. Do not resubmit a config already in history.
- LDS bytes = stages * (tile_m + tile_n) * (tile_k + lds_pad) * element_bytes. element_bytes is 2 for bf16 and 1 for fp8. The cap is 163840.
- bf16 mfma is 32x32x16 or 16x16x32. fp8 mfma is 32x32x16, 32x32x64, or 16x16x128.
- waves_m * waves_n is 1, 2, 4, or 8. tile_m is a multiple of waves_m times the MFMA M, tile_n of waves_n times the MFMA M, and tile_k of the MFMA K.
- split_k above 1 splits K across workgroups. Use it when M is small."""

_INTRO = """The compiler lowers one typed schedule to MFMA kernels with f32 accumulation and a bf16 result. bf16 operands have no scales. fp8 operands are e4m3 with blockwise scales: activation 1x128, weight 128x128. Score is the geomean speedup against the baseline across this portfolio only. Prefill and decode never share an episode."""

PLANNER_PROMPT = f"""You are the Planner specialist for gfx950 GEMM scheduling (MMKCA, MI355X, wave64).

{_INTRO}

Rules:
- Call exactly one tool per turn.
{_RULES}
- Call report when the sub-budget is spent or another one-axis move is unlikely to help.
"""

SYSTEM_PROMPT = f"""You are an AMD GPU kernel optimization agent for gfx950 GEMM scheduling (MMKCA, MI355X, wave64).

{_INTRO}

Rules:
- Act only through tools, exactly one tool call per turn.
{_RULES}
- Call `finish` with a short summary when further submissions are unlikely to help.
"""


class MmkcaGemmEnvironment(MicakeGemmEnvironment):
    """AgentEnvironment backed by MMKCA's typed ``mmkca.gemm`` schedule workflow."""

    planner_prompt = PLANNER_PROMPT

    def __init__(
        self,
        mmkca_root: str,
        image: str,
        gpu_id: int,
        rocm_llvm: str,
        request_timeout_s: float,
        fidelity: str,
        max_submissions: int,
        measurement: Mapping[str, int],
        reward: Mapping[str, float],
        evaluator: MmkcaWorkerClient | None = None,
        new_best_min_gain: float = 0.0,
    ) -> None:
        """Args:
            mmkca_root: MMKCA worktree with a built ``mmkca-opt``.
            image: GPU image that runs ``scripts/mmkca_worker.py``.
            gpu_id: Physical GPU exposed to the worker container.
            rocm_llvm: ROCm LLVM bin directory with gfx950 ``opt``/``llc``/``ld.lld``.
            request_timeout_s: Per-evaluation timeout; expiry is an unknown outcome.
            fidelity: Default highest stage: verify | compile | correctness | performance.
            max_submissions: Default submit_schedule budget per episode.
            measurement: ``paired_repeats`` and ``batch`` for graph-replay timing.
            reward: :class:`RewardWeights` values.
            evaluator: Injected worker client (tests).
            new_best_min_gain: Relative margin a speedup must clear to become the new best.
        """
        super().__init__(
            micake_root=mmkca_root, micake_python="", gpu_id=gpu_id, rocm_path="", request_timeout_s=request_timeout_s,
            fidelity=fidelity, max_submissions=max_submissions, measurement=measurement, reward=reward,
            evaluator=evaluator or MmkcaWorkerClient(mmkca_root, image, gpu_id, rocm_llvm, request_timeout_s),
            new_best_min_gain=new_best_min_gain,
        )
        self._descriptions: dict[tuple, Mapping[str, Any]] = {}
        self._task_axes: dict[str, Any] = {"dtype": "bf16", "portfolio": "decode", "split": "train", "bundle": None}

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(backend=BACKEND_NAME, snapshot=True, async_step=True, fidelity_levels=FIDELITY_LEVELS)

    @property
    def dtype(self) -> str:
        return self._task_axes["dtype"]

    @property
    def axes(self) -> dict[str, Any]:
        """Regime, dtype, split and bundle of the current episode."""
        return dict(self._task_axes)

    async def _evaluate(self, config: Mapping[str, Any], fidelity: str) -> tuple[dict[str, Any], bool]:
        axes = self._task_axes
        key = (json.dumps(config, sort_keys=True), fidelity, axes["dtype"], axes["portfolio"], axes["split"], axes["bundle"])
        if key in self._result_cache:
            return copy.deepcopy(self._result_cache[key]), True
        result = await self._call_evaluator(config, fidelity)
        if result.get("stage") not in UNKNOWN_STAGES:
            self._result_cache[key] = copy.deepcopy(result)
        return result, False

    async def _call_evaluator(self, config: Mapping[str, Any], fidelity: str) -> dict[str, Any]:
        axes = self._task_axes
        return await self._evaluator.evaluate(config, fidelity, self._measurement, axes["dtype"], axes["portfolio"],
                                              axes["split"], axes["bundle"])

    async def reset(self, task: Mapping[str, Any]) -> ResetResult:
        dtype = task.get("dtype", "bf16")
        portfolio = task.get("portfolio", "decode")
        split = task.get("split", "train")
        bundle = task.get("bundle")
        if dtype not in INPUT_DTYPES:
            raise ValueError(f"task dtype {dtype!r} not in {sorted(INPUT_DTYPES)}")
        if portfolio not in PORTFOLIOS or split not in SPLITS:
            raise ValueError(f"task portfolio/split {portfolio!r}/{split!r} not in {PORTFOLIOS}/{SPLITS}")
        if (split == "heldout") != (bundle is None):
            raise ValueError("held-out tasks take no bundle; training tasks need one")
        fidelity = task.get("fidelity", self._default_fidelity)
        if fidelity not in FIDELITY_LEVELS:
            raise ValueError(f"task fidelity {fidelity!r} not in {FIDELITY_LEVELS}")
        self._task_axes = {"dtype": dtype, "portfolio": portfolio, "split": split,
                           "bundle": None if bundle is None else int(bundle)}
        key = tuple(self._task_axes.values())
        if key not in self._descriptions:
            self._descriptions[key] = await self._evaluator.describe(dtype, portfolio, split, self._task_axes["bundle"])
        self._description = description = self._descriptions[key]
        max_submissions = int(task.get("max_submissions", self._default_max_submissions))
        self._state = {"dtype": dtype}
        baseline, _ = await self._evaluate(description["baseline_config"], "compile")
        bundle_tag = f".b{bundle}" if bundle is not None else ""
        self._state = {
            "dtype": dtype,
            "task_id": task.get("task_id", f"mmkca.{split}.{portfolio}.{dtype}{bundle_tag}"),
            "fidelity": fidelity,
            "max_submissions": max_submissions,
            "submissions": 0,
            "seen": {},
            "best": {"speedup": 1.0, "config": description["baseline_config"], "source": "baseline"},
            "correct_count": 0,
            "finished_summary": None,
            "history": [],
        }
        element = INPUT_DTYPES[dtype]
        self._state["task_packet"] = observation = {
            "task_id": self._state["task_id"],
            "objective": description["objective"],
            "regime": portfolio,
            "input_dtype": element["name"],
            "target": description["target"],
            "portfolio": description["portfolio"],
            "baseline_config": description["baseline_config"],
            "opening": description["opening"],
            "baseline_compiled_facts": baseline.get("compiled_facts"),
            "domain": description["domain"],
            "resource_limits": {**description["resource_limits"], "lds_element_bytes": element["element_bytes"]},
            "evaluation_fidelity": fidelity,
            "budget": {"max_submissions": max_submissions},
        }
        return ResetResult(
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            tools=tool_schemas(description["domain"]),
            metadata={
                "backend": BACKEND_NAME,
                "task_id": self._state["task_id"],
                "candidate_profile": description["candidate_profile"],
                "portfolio_profile": description["portfolio_profile"],
                **self._task_axes,
                "fidelity": fidelity,
                "max_submissions": max_submissions,
                "measurement": self._measurement,
                "reward_weights": self._weights.__dict__,
            },
        )

    async def finalize(self, terminal_state: str) -> tuple[float, dict[str, Any]]:
        reward, info = await super().finalize(terminal_state)
        return reward, {**info, **self._task_axes}
