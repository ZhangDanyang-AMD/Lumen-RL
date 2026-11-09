"""Hierarchical Director/TechLead/Engineer/Verifier/Integrator flow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .agents import CodeRoleAgent, RolePromptLibrary, StructuredRoleAgent
from .config import MultiTuneConfig
from .geak_tool import GEAKToolEnvironment
from .models import Candidate, Direction
from .runtime import ModelBackend, gather_limited
from .sft_collector import SFTCollector
from .trajectory import TrajectoryWriter


class MultiTuneFlow:
    def __init__(
        self,
        config: MultiTuneConfig,
        backend: ModelBackend,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.event_sink = event_sink

    async def run_all(
        self,
        case_ids: Optional[list[str]] = None,
        *,
        user_request: Optional[str] = None,
        resume_workspace: Optional[Path] = None,
        resume_context: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        probe_env = GEAKToolEnvironment(self.config)
        selected = case_ids or list(probe_env.cases)
        if (user_request or resume_workspace or resume_context) and len(selected) != 1:
            raise ValueError("request/resume context can target exactly one case")
        unknown = sorted(set(selected) - set(probe_env.cases))
        if unknown:
            raise ValueError("unknown case ids: %s" % ", ".join(unknown))
        results = []
        for case_id in selected:
            results.append(
                await self.run_case(
                    case_id,
                    user_request=user_request,
                    resume_workspace=resume_workspace,
                    resume_context=resume_context,
                )
            )
        return results

    async def run_case(
        self,
        case_id: str,
        *,
        user_request: Optional[str] = None,
        resume_workspace: Optional[Path] = None,
        baseline_override: Optional[Mapping[str, Any]] = None,
        resume_context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        run_started = time.monotonic()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_dir = self._unique_run_dir(case_id, stamp)
        collector = (
            SFTCollector(run_dir, self.config) if self.config.sft_enabled else None
        )
        trajectory = TrajectoryWriter(
            run_dir,
            event_sink=self.event_sink,
            sft_sink=collector.append if collector is not None else None,
        )
        environment = GEAKToolEnvironment(self.config, trajectory)
        prompts = RolePromptLibrary(self.config.geak_root)
        structured = StructuredRoleAgent(self.backend, prompts, trajectory)
        code_agent = CodeRoleAgent(self.backend, prompts, environment, trajectory)
        case = environment._case(case_id)
        case_input = environment.case_observation(case_id)
        if user_request:
            case_input["user_request"] = user_request.strip()
        if resume_workspace is not None:
            case_input["resumed_from_workspace"] = str(resume_workspace)
        timing: dict[str, float] = {}
        history: list[dict[str, Any]] = []
        task_type = self.config.sft_task_type
        if baseline_override is None and isinstance(resume_context, Mapping):
            context_baseline = resume_context.get("baseline")
            if isinstance(context_baseline, Mapping):
                baseline_override = context_baseline

        setup_started = time.monotonic()
        create_options: dict[str, Any] = {
            "role": "director",
            "establish_baseline": baseline_override is None,
        }
        if resume_workspace is not None:
            create_options["source_path"] = resume_workspace
            create_options["baseline_override"] = baseline_override
        root_session_id, initial = await asyncio.to_thread(
            environment.create,
            case_id,
            **create_options,
        )
        timing["environment_setup"] = time.monotonic() - setup_started
        source_context = self._source_context(
            environment, root_session_id, initial["allowed_write_paths"]
        )
        trajectory.append(
            "run_start",
            {
                "case": case_input,
                "config": {
                    "max_rounds": self.config.max_rounds,
                    "engineers_per_round": self.config.engineers_per_round,
                    "candidate_floor": self.config.candidate_floor,
                    "min_improvement": self.config.min_improvement,
                    "target_speedup": self.config.target_speedup,
                    "gpu_ids": self.config.gpu_ids,
                    "sft_task_type": task_type,
                },
                "initial": initial,
                "resume_context": dict(resume_context or {}),
            },
        )

        phase_started = time.monotonic()
        director_setup = await structured.run(
            "director",
            "setup",
            {
                "case": case_input,
                "baseline": initial["baseline"],
                "allowed_write_paths": initial["allowed_write_paths"],
                "source_context": source_context,
                "resume_context": dict(resume_context or {}),
                "objective": "Create the optimization charter and identify validation risks.",
            },
            case_type=case.case_type,
        )
        timing["director_setup"] = time.monotonic() - phase_started

        analysis: dict[str, Any] = {}
        if task_type == "direction_conditioned":
            phase_started = time.monotonic()
            analysis = await structured.run(
                "tech_lead",
                "analyze",
                {
                    "case": case_input,
                    "baseline": initial["baseline"],
                    "source_context": source_context,
                    "resume_context": dict(resume_context or {}),
                    "director_charter": director_setup,
                },
                case_type=case.case_type,
            )
            timing["tech_lead_analyze"] = time.monotonic() - phase_started

        current_session_id = root_session_id
        current_speedup = 1.0
        best_candidate: Optional[Candidate] = None
        no_improve = 0

        for round_index in range(1, self.config.max_rounds + 1):
            plan_started = time.monotonic()
            if task_type == "direction_conditioned":
                plan = await structured.run(
                    "tech_lead",
                    "plan_round",
                    {
                        "round": round_index,
                        "user_request": user_request or case.direction,
                        "analysis": analysis,
                        "current_speedup": current_speedup,
                        "previous_rounds": history,
                        "direction_limit": self.config.engineers_per_round,
                        "allowed_write_paths": initial["allowed_write_paths"],
                        "direction_constraint": (
                            "Every direction must be implementable using only "
                            "allowed_write_paths. Do not propose wrapper, binding, "
                            "harness, test, or configuration edits unless that exact "
                            "path is listed."
                        ),
                        "required_output": {"directions": [{"id": "unique id"}]},
                    },
                    case_type=case.case_type,
                    round_index=round_index,
                )
                directions = self._directions(
                    plan, self.config.engineers_per_round
                )
            else:
                plan = {}
                directions = [
                    Direction(task_type, task_type, task_type, task_type)
                ]
            timing["round_%d_plan" % round_index] = time.monotonic() - plan_started
            if collector is not None:
                if task_type == "direction_conditioned":
                    collector.record_plan(
                        round_index,
                        plan,
                        [item.__dict__ for item in directions],
                        user_request=user_request or case.direction,
                    )

            frozen_input = self._frozen_input(
                task_type,
                case_input,
                initial["baseline"],
                source_context,
                resume_context or {},
            )
            if collector is not None:
                collector.record_frozen_input(round_index, frozen_input)

            engineer_started = time.monotonic()
            outputs = await gather_limited(
                [
                    code_agent.engineer(
                        case_id,
                        case.case_type,
                        current_session_id,
                        direction,
                        max_turns=self.config.engineer_tool_rounds,
                        round_index=round_index,
                        task_type=task_type,
                        frozen_input=frozen_input,
                    )
                    for direction in directions
                ],
                self.config.engineers_per_round,
            )
            timing["round_%d_engineers" % round_index] = (
                time.monotonic() - engineer_started
            )

            verify_started = time.monotonic()
            candidates: list[Candidate] = []
            for direction, output in zip(directions, outputs):
                result, reward, _ = await asyncio.to_thread(
                    environment.independent_verify, output.session_id
                )
                evaluation = dict(result.get("evaluation") or {})
                accepted = self._candidate_accepted(
                    task_type,
                    result,
                    evaluation,
                    initial["baseline"],
                )
                candidate = Candidate(
                    candidate_id="%s-r%d" % (direction.direction_id, round_index),
                    session_id=output.session_id,
                    direction=direction,
                    evaluation=evaluation,
                    reward=reward,
                    accepted=accepted,
                    agent_text=output.final_text,
                )
                candidate.verifier = await structured.run(
                    "verifier",
                    "verify",
                    {
                        "candidate": candidate.to_dict(),
                        "deterministic_gate": {
                            "accepted": accepted,
                            "candidate_floor": self.config.candidate_floor,
                            "rule": "compiled && correct && speedup >= candidate_floor",
                        },
                    },
                    case_type=case.case_type,
                    round_index=round_index,
                )
                if collector is not None:
                    collector.record_candidate(
                        environment,
                        current_session_id,
                        output.session_id,
                        candidate.to_dict(),
                        result,
                        round_index=round_index,
                    )
                candidates.append(candidate)
            timing["round_%d_verify" % round_index] = time.monotonic() - verify_started

            accepted_candidates = sorted(
                (item for item in candidates if item.accepted),
                key=lambda item: item.speedup,
                reverse=True,
            )
            winner = accepted_candidates[0] if accepted_candidates else None

            if task_type == "direction_conditioned" and len(accepted_candidates) >= 2:
                integrate_started = time.monotonic()
                integration = await code_agent.integrator(
                    case_id,
                    case.case_type,
                    current_session_id,
                    [item.to_dict() for item in accepted_candidates],
                    max_turns=self.config.integrator_tool_rounds,
                    round_index=round_index,
                )
                result, reward, _ = await asyncio.to_thread(
                    environment.independent_verify, integration.session_id
                )
                evaluation = dict(result.get("evaluation") or {})
                integrated = Candidate(
                    candidate_id="integrated-r%d" % round_index,
                    session_id=integration.session_id,
                    direction=Direction(
                        "integrated", "integration", "merge verified candidates", ""
                    ),
                    evaluation=evaluation,
                    reward=reward,
                    accepted=bool(result.get("ok") and evaluation.get("correct")),
                    agent_text=integration.final_text,
                )
                if collector is not None:
                    collector.record_candidate(
                        environment,
                        current_session_id,
                        integration.session_id,
                        integrated.to_dict(),
                        result,
                        round_index=round_index,
                        role="integrator",
                    )
                if integrated.accepted and integrated.speedup >= winner.speedup:
                    winner = integrated
                timing["round_%d_integrate" % round_index] = (
                    time.monotonic() - integrate_started
                )

            committed = bool(
                winner
                and (
                    task_type == "error_recovery"
                    or winner.speedup
                    >= current_speedup * (1.0 + self.config.min_improvement)
                )
            )
            if committed and winner is not None:
                current_session_id = winner.session_id
                current_speedup = winner.speedup
                best_candidate = winner
                no_improve = 0
            else:
                no_improve += 1

            round_record = {
                "round": round_index,
                "directions": [item.__dict__ for item in directions],
                "candidates": [item.to_dict() for item in candidates],
                "winner": winner.to_dict() if winner else None,
                "committed": committed,
                "current_speedup": current_speedup,
                "no_improve": no_improve,
            }
            history.append(round_record)
            trajectory.append(
                "round_end",
                round_record,
                role="tech_lead",
                phase="update_memory",
                round_index=round_index,
            )
            if current_speedup >= self.config.target_speedup or no_improve >= 2:
                break

        final_started = time.monotonic()
        final_result, final_reward, _ = await asyncio.to_thread(
            environment.independent_verify, current_session_id
        )
        final_evaluation = dict(final_result.get("evaluation") or {})
        final_kernel_performance = self._final_kernel_performance(
            initial["baseline"], final_evaluation
        )
        director_validation = await structured.run(
            "director",
            "validate",
            {
                "case": case_input,
                "round_history": history,
                "final_evaluation": final_evaluation,
                "final_kernel_performance": final_kernel_performance,
                "final_reward": final_reward.to_dict(),
                "required_rule": "accept only deterministic compile+correctness+performance evidence",
            },
            case_type=case.case_type,
        )
        timing["director_validate"] = time.monotonic() - final_started
        timing["total"] = time.monotonic() - run_started
        success = bool(
            best_candidate is not None
            and current_session_id != root_session_id
            and final_result.get("ok")
            and final_evaluation.get("correct")
            and (
                task_type == "error_recovery"
                or float(final_evaluation.get("speedup_geomean") or 0.0) >= 1.0
            )
        )
        summary = {
            "case_id": case_id,
            "case_type": case.case_type,
            "user_request": user_request or case.direction,
            "resumed_from": (
                dict(resume_context or {}) if resume_workspace is not None else None
            ),
            "status": "success" if success else "no_improvement",
            "root_session_id": root_session_id,
            "final_session_id": current_session_id,
            "best_candidate": best_candidate.to_dict() if best_candidate else None,
            "final_evaluation": final_evaluation,
            "final_kernel_performance": final_kernel_performance,
            "final_reward": final_reward.to_dict(),
            "director_validation": director_validation,
            "rounds": history,
            "timing_seconds": timing,
            "run_dir": str(run_dir),
        }
        finalized = trajectory.finalize(summary)
        if collector is not None:
            collector.finalize(finalized)
        return finalized

    @staticmethod
    def _frozen_input(
        task_type: str,
        case_input: Mapping[str, Any],
        baseline: Mapping[str, Any],
        source_context: Mapping[str, Any],
        resume_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_hash = hashlib.sha256(
            json.dumps(source_context, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        provenance = case_input.get("provenance")
        case_seed = (
            provenance.get("case_seed")
            if isinstance(provenance, Mapping)
            else None
        )
        split_group = (
            case_seed.get("split_group")
            if isinstance(case_seed, Mapping)
            else resume_context.get("split_group")
        )
        if split_group == "held_out":
            raise ValueError("held-out cases cannot be used for SFT collection")
        source_lineage_id = (
            case_seed.get("source_lineage_id")
            if isinstance(case_seed, Mapping)
            else resume_context.get("source_lineage_id")
        )
        if task_type != "direction_conditioned" and (
            not source_lineage_id or split_group not in {"train", "dev"}
        ):
            raise ValueError(
                "%s requires a frozen non-held-out source lineage and split"
                % task_type
            )
        frozen: dict[str, Any] = {
            "contract": dict(case_input),
            "parent_source": dict(source_context),
            "parent_source_hash": source_hash,
            "baseline": dict(baseline),
            "source_lineage_id": source_lineage_id,
            "split_group": split_group,
            "split_version": (
                case_seed.get("split_version")
                if isinstance(case_seed, Mapping)
                else resume_context.get("split_version")
            ),
        }
        if task_type == "profile_guided":
            profile = resume_context.get("profile")
            frozen["profile"] = (
                dict(profile)
                if isinstance(profile, Mapping)
                else {
                    "schema_version": "geak_parent_profile_v1",
                    "source_hash": source_hash,
                    "per_case_ms": dict(baseline.get("per_case_ms") or {}),
                    "geomean_ms": baseline.get("geomean_ms"),
                    "measurement": "independent_parent_baseline",
                }
            )
            if frozen["profile"].get("source_hash") != source_hash:
                raise ValueError("profile source hash does not match parent source")
        elif task_type == "error_recovery":
            failure = resume_context.get("error_feedback") or resume_context.get(
                "last_failure"
            )
            if not isinstance(failure, Mapping) or not failure:
                raise ValueError("error_recovery requires exact frozen error feedback")
            frozen["error_feedback"] = dict(failure)
        elif task_type == "regression_balance":
            report = resume_context.get("per_case_benchmark")
            constraints = resume_context.get("regression_constraints")
            if not isinstance(report, Mapping) or not report:
                raise ValueError(
                    "regression_balance requires a frozen per-case benchmark"
                )
            if not isinstance(constraints, Mapping) or not constraints:
                raise ValueError(
                    "regression_balance requires frozen regression constraints"
                )
            frozen["per_case_benchmark"] = dict(report)
            frozen["regression_constraints"] = dict(constraints)
        return frozen

    @staticmethod
    def _candidate_accepted(
        task_type: str,
        result: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        baseline: Mapping[str, Any],
    ) -> bool:
        if not (
            result.get("ok")
            and evaluation.get("compiled")
            and evaluation.get("correct")
        ):
            return False
        if task_type == "error_recovery":
            return True
        if task_type == "regression_balance":
            base = evaluation.get("baseline_ms") or baseline.get("per_case_ms")
            candidate = evaluation.get("candidate_ms")
            if not isinstance(base, Mapping) or not isinstance(candidate, Mapping):
                return False
            shared = set(base) & set(candidate)
            return bool(shared) and all(
                float(candidate[name]) <= float(base[name]) for name in shared
            )
        return (
            float(evaluation.get("speedup_geomean") or 0.0)
            >= 1.0
            if task_type in {"cold_start", "profile_guided"}
            else float(evaluation.get("speedup_geomean") or 0.0)
            >= 1.0
        )

    @staticmethod
    def _final_kernel_performance(
        baseline: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> dict[str, Any]:
        baseline_value = evaluation.get("baseline_ms") or baseline.get("per_case_ms")
        candidate_value = evaluation.get("candidate_ms")
        baseline_cases = (
            {str(name): float(value) for name, value in baseline_value.items()}
            if isinstance(baseline_value, Mapping)
            else {}
        )
        candidate_cases = (
            {str(name): float(value) for name, value in candidate_value.items()}
            if isinstance(candidate_value, Mapping)
            else {}
        )
        measurement_valid = bool(
            evaluation.get("compiled")
            and evaluation.get("correct")
            and float(evaluation.get("speedup_geomean") or 0.0) > 0.0
            and candidate_cases
        )
        if not measurement_valid:
            candidate_cases = {}
        shared = sorted(set(baseline_cases) & set(candidate_cases))

        def geomean(values: list[float]) -> Optional[float]:
            positive = [value for value in values if value > 0.0]
            if len(positive) != len(values) or not positive:
                return None
            return math.exp(sum(math.log(value) for value in positive) / len(positive))

        return {
            "measurement_valid": measurement_valid,
            "unit": "ms",
            "baseline_geomean_ms": geomean(
                [baseline_cases[name] for name in shared]
            ),
            "final_kernel_geomean_ms": geomean(
                [candidate_cases[name] for name in shared]
            ),
            "speedup_geomean": (
                float(evaluation.get("speedup_geomean") or 0.0)
                if measurement_valid
                else None
            ),
            "baseline_per_case_ms": baseline_cases,
            "final_kernel_per_case_ms": candidate_cases,
        }

    @staticmethod
    def _source_context(
        environment: GEAKToolEnvironment,
        session_id: str,
        allowed_paths: list[str],
        limit: int = 12000,
    ) -> dict[str, str]:
        state = environment.get(session_id)
        context: dict[str, str] = {}
        remaining = limit
        for path in allowed_paths:
            if remaining <= 0:
                break
            try:
                content = state.sandbox.read_file(path, 1, 400)
            except (AttributeError, OSError, RuntimeError, ValueError):
                continue
            text = str(content)
            context[path] = text[:remaining]
            remaining -= len(context[path])
        return context

    def _unique_run_dir(self, case_id: str, stamp: str) -> Path:
        base = self.config.trajectory_root / "runs" / ("%s_%s" % (case_id, stamp))
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = Path(str(base) + "_%d" % suffix)
            suffix += 1
        return candidate

    @staticmethod
    def _directions(plan: dict[str, Any], limit: int) -> list[Direction]:
        raw = plan.get("directions")
        if isinstance(raw, list):
            values = [
                Direction.from_mapping(item, index)
                for index, item in enumerate(raw[:limit])
                if isinstance(item, dict)
            ]
            if values:
                return values
        defaults = [
            ("algorithm", "Change tiling or decomposition based on measured regimes."),
            ("memory", "Reduce global traffic and improve coalescing or reuse."),
            ("compute", "Tune launch geometry, pipeline depth, and instruction mix."),
            ("host_runtime", "Remove avoidable allocation or launch overhead."),
        ]
        return [
            Direction("fallback_%d" % (index + 1), specialty, strategy, strategy)
            for index, (specialty, strategy) in enumerate(defaults[:limit])
        ]

