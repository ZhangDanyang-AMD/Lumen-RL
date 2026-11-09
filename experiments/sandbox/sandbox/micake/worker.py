"""MICake GEMM evaluation worker (runs inside MICake's own Python environment).

Speaks a JSON-lines protocol on stdin/stdout so that LumenRL never imports
MICake or Triton. MICake's C144 modules remain the authority for candidate
validation, source materialization, compilation and GPU correctness/timing.

Requests::

    {"op": "describe"}
    {"op": "evaluate", "config": {...}, "fidelity": "verify|compile|correctness|performance",
     "measurement": {"warmup_count": int, "paired_repeats": int, "dispatches_per_block": int}}

Every response is one JSON object on one line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import statistics
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("micake.worker")

FIDELITY_LEVELS = ("verify", "compile", "correctness", "performance")
FACT_KEYS = (
    "vgpr_count", "agpr_count", "sgpr_count", "spill_count", "scratch_size_bytes",
    "shared_bytes", "reported_occupancy_waves", "code_size_bytes",
)
MESSAGE_LIMIT = 800


class Worker:
    """Caches compiled artifacts per canonical config and evaluates candidates."""

    def __init__(self, micake_root: Path) -> None:
        sys.path.insert(0, str(micake_root))
        from experiments.cake_gemm_optimization_workflow_v1 import (  # noqa: PLC0415
            candidate, compile_runtime, contract, emitter, gpu_runtime,
        )

        self.candidate = candidate
        self.compile_runtime = compile_runtime
        self.contract = contract
        self.emitter = emitter
        self.gpu = gpu_runtime
        self.workdir = Path(tempfile.mkdtemp(prefix="micake-worker-"))
        self.artifacts: dict[str, tuple[Any, Any]] = {}
        self.session_counter = 0

    def describe(self) -> dict[str, Any]:
        c = self.contract
        return {
            "candidate_profile": c.CANDIDATE_PROFILE,
            "portfolio_profile": c.WORKLOAD_PORTFOLIO_PROFILE,
            "portfolio": [item.to_dict() for item in c.WORKLOAD_PORTFOLIO],
            "baseline_config": c.BASELINE_CONFIG.to_dict(),
            "domain": {
                "schedule": {
                    "tile_m": list(c.TILE_M_VALUES),
                    "tile_n": list(c.TILE_N_VALUES),
                    "tile_k": list(c.TILE_K_VALUES),
                    "pipeline_depth": list(c.PIPELINE_DEPTH_VALUES),
                    "unroll_factor": list(c.UNROLL_FACTOR_VALUES),
                    "vectorize_factor": list(c.VECTORIZE_FACTOR_VALUES),
                },
                "realization": {
                    "program_order": list(c.PROGRAM_ORDER_VALUES),
                    "group_m": list(c.GROUP_M_VALUES),
                    "num_warps": list(c.NUM_WARPS_VALUES),
                    "num_stages": list(c.NUM_STAGES_VALUES),
                    "waves_per_eu": list(c.WAVES_PER_EU_VALUES),
                    "matrix_instr_nonkdim": list(c.MATRIX_INSTR_NONKDIM_VALUES),
                    "kpack": list(c.KPACK_VALUES),
                },
            },
            "target": dict(self.compile_runtime.TARGET),
            "resource_limits": {"max_group_segment_bytes": self.gpu.GFX950_MAX_GROUP_SEGMENT_BYTES},
        }

    def _compile(self, config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        key = json.dumps(config, sort_keys=True)
        if key in self.artifacts:
            record, artifact = self.artifacts[key]
            return record, artifact, {}
        record = self.candidate.build_candidate(self.contract.CandidateConfig.from_dict(config))
        source = self.emitter.emit_candidate_source(record)
        root = self.workdir / source.source_sha256[:16]
        root.mkdir(exist_ok=True)
        path = root / "kernel.py"
        path.write_text(source.source, encoding="utf-8")
        artifact = self.compile_runtime.compile_candidate(
            path, root, source.source_sha256, source.entrypoint, record.config,
        )
        self.artifacts[key] = (record, artifact)
        return record, artifact, {"source_sha256": source.source_sha256}

    def _session(self, record: Any, artifact: Any) -> Any:
        self.session_counter += 1
        session = self.gpu.PreparedSession(artifact, 0, record.config, self.contract.WORKLOAD_PORTFOLIO)
        session.authorize(f"lumenrl-micake-{self.session_counter}")
        return session

    def evaluate(self, config: dict[str, Any], fidelity: str, measurement: dict[str, Any] | None) -> dict[str, Any]:
        if fidelity not in FIDELITY_LEVELS:
            return {"stage": "request_invalid", "diagnostic": {"code": f"UNKNOWN_FIDELITY:{fidelity}"}}
        result: dict[str, Any] = {"fidelity": fidelity}
        try:
            record = self.candidate.build_candidate(self.contract.CandidateConfig.from_dict(config))
        except self.contract.WorkflowError as exc:
            return {**result, "stage": "verify_fail", "diagnostic": {"code": exc.code}}
        result["candidate_digest"] = record.candidate_digest
        if fidelity == "verify":
            return {**result, "stage": "verified"}

        try:
            record, artifact, extra = self._compile(config)
        except self.compile_runtime.CompileRejected as exc:
            return {**result, "stage": "compile_fail", "diagnostic": _compile_diagnostic(exc)}
        except self.compile_runtime.CompileUnknown as exc:
            return {**result, "stage": "compile_unknown", "diagnostic": _compile_diagnostic(exc)}
        observation = artifact.observation
        facts = observation.get("binary_facts") or {}
        result.update(extra)
        result["source_sha256"] = artifact.source_sha256
        result["compiled_facts"] = {key: facts.get(key) for key in FACT_KEYS}
        result["isa_family_counts"] = observation.get("isa_family_counts")
        if fidelity == "compile":
            return {**result, "stage": "compiled"}

        rejected = _group_segment_rejection(result["compiled_facts"], self.gpu.GFX950_MAX_GROUP_SEGMENT_BYTES)
        if rejected is not None:
            return {**result, "stage": "resource_rejected", "diagnostic": rejected}

        try:
            with self._session(record, artifact) as session:
                correctness = session.run_correctness()
                session.close()
        except self.gpu.GpuRuntimeUnknown as exc:
            return {**result, "stage": "gpu_unknown", "diagnostic": _gpu_diagnostic(exc.code, self.gpu.cause_message(exc))}
        except self.gpu.GpuRuntimeRefused as exc:
            if exc.code == self.gpu.GROUP_SEGMENT_EXCEEDED:
                return {**result, "stage": "resource_rejected", "diagnostic": _gpu_diagnostic(exc.code, str(exc))}
            return {**result, "stage": "gpu_refused", "diagnostic": _gpu_diagnostic(exc.code, self.gpu.cause_message(exc))}
        except self.gpu.GpuRuntimeError as exc:
            return {**result, "stage": "gpu_refused", "diagnostic": _gpu_diagnostic(exc.code, self.gpu.cause_message(exc))}
        result["correctness"] = _correctness_summary(correctness)
        if correctness.get("reason_code") == self.gpu.GROUP_SEGMENT_EXCEEDED:
            return {**result, "stage": "resource_rejected", "diagnostic": _gpu_diagnostic(
                correctness.get("reason_code"), correctness.get("message") or "",
            )}
        if correctness.get("status") == "UNKNOWN":
            message = correctness.get("message") or ""
            if correctness.get("dispatch_completed_count") == 0 and _resource_launch_text(message):
                return {**result, "stage": "resource_rejected", "diagnostic": _gpu_diagnostic(correctness.get("reason_code"), message)}
            return {**result, "stage": "gpu_unknown", "diagnostic": _gpu_diagnostic(correctness.get("reason_code"), message)}
        if correctness.get("status") != "PASS":
            return {**result, "stage": "correctness_fail", "diagnostic": _gpu_diagnostic(
                correctness.get("reason_code"), correctness.get("message") or "",
            )}
        if fidelity == "correctness":
            return {**result, "stage": "correct"}

        baseline = self.contract.BASELINE_CONFIG.to_dict()
        try:
            base_record, base_artifact, _ = self._compile(baseline)
            with self._session(base_record, base_artifact) as session:
                measured = session.measure_pair(artifact, dict(measurement or {}), record.config)
                session.close()
        except self.gpu.GpuRuntimeUnknown as exc:
            return {**result, "stage": "gpu_unknown", "diagnostic": _gpu_diagnostic(exc.code, self.gpu.cause_message(exc))}
        except self.gpu.GpuRuntimeError as exc:
            return {**result, "stage": "gpu_refused", "diagnostic": _gpu_diagnostic(exc.code, self.gpu.cause_message(exc))}
        result["performance"] = _performance_summary(measured)
        return {**result, "stage": "measured"}


def _group_segment_rejection(facts: dict[str, Any], limit: int) -> dict[str, Any] | None:
    """Candidate failure for a compiled kernel that cannot be launched on gfx950."""
    shared = facts.get("shared_bytes")
    if type(shared) is not int or shared <= limit:
        return None
    return {
        "code": "C144_GROUP_SEGMENT_EXCEEDED",
        "message": f"shared_bytes {shared} exceeds gfx950 group segment limit {limit}",
        "shared_bytes": shared,
        "limit_bytes": limit,
    }


def _resource_launch_text(message: str) -> bool:
    text = message.lower()
    markers = ("shared memory", "group segment", "out of resource", "lds")
    return any(marker in text for marker in markers)


def _gpu_diagnostic(code: str | None, message: str) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"code": code}
    if message:
        diagnostic["message"] = message[-MESSAGE_LIMIT:]
    return diagnostic


def _compile_diagnostic(exc: Any) -> dict[str, Any]:
    cause = exc.__cause__
    message = f"{type(cause).__name__}: {cause}" if cause is not None else ""
    return {"code": exc.code, "message": message[-MESSAGE_LIMIT:]}


def _correctness_summary(observation: dict[str, Any]) -> dict[str, Any]:
    cases = observation.get("cases") or []
    return {
        "status": observation.get("status"),
        "reason_code": observation.get("reason_code"),
        "cases": [
            {
                "workload_id": case["workload"]["workload_id"],
                "status": case.get("status"),
                "mismatch_count": case.get("mismatch_count"),
                "checked_elements": case.get("checked_elements"),
            }
            for case in cases
        ],
    }


def _performance_summary(observation: dict[str, Any]) -> dict[str, Any]:
    samples: dict[str, dict[str, list[int]]] = {}
    for sample in observation.get("raw_samples") or []:
        roles = samples.setdefault(sample["workload_id"], {"incumbent": [], "challenger": []})
        roles[sample["candidate_role"]].append(sample["elapsed_ns"])
    per_workload = {}
    for workload_id, roles in samples.items():
        if roles["incumbent"] and roles["challenger"]:
            per_workload[workload_id] = statistics.median(roles["incumbent"]) / statistics.median(roles["challenger"])
    geomean = (
        math.exp(sum(math.log(value) for value in per_workload.values()) / len(per_workload))
        if per_workload else None
    )
    return {
        "status": observation.get("status"),
        "validity": observation.get("measurement_validity"),
        "reason_code": observation.get("reason_code"),
        "baseline": "C144_BASELINE_CONFIG",
        "per_workload_speedup": per_workload,
        "geomean_speedup": geomean,
        "sample_count": observation.get("sample_count"),
        "scope": "LOCAL_DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
    }


def serve(worker: Worker, protocol_out: Any) -> None:
    """Serve requests until stdin closes; third-party prints go to stderr."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("op") == "describe":
                    response = {"ok": True, "result": worker.describe()}
                elif request.get("op") == "evaluate":
                    response = {
                        "ok": True,
                        "result": worker.evaluate(request["config"], request["fidelity"], request.get("measurement")),
                    }
                else:
                    response = {"ok": False, "error": f"unknown op {request.get('op')!r}"}
        except Exception as exc:  # noqa: BLE001 - report every worker fault to the client
            logger.error("worker fault: %s", traceback.format_exc())
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[-MESSAGE_LIMIT:]}
        protocol_out.write(json.dumps(response, sort_keys=True) + "\n")
        protocol_out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--micake-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    serve(Worker(args.micake_root), protocol_out)


if __name__ == "__main__":
    main()
