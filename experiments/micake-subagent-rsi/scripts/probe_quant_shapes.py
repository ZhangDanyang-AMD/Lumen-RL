"""Probe the champion GEMM schedule on more shapes, and try an FP8 compile.

The C144 harness measures the existing fp16 Triton kernel. This script keeps
that kernel and adds the C145 tail shapes plus several larger GEMM shapes.
It also asks the same gfx950 compiler to compile a float8e4m3fn variant of the
champion schedule. The FP8 kernel is not timed: the harness allocates fp16
inputs, so an FP8 launch would not be a meaningful measurement.

Example:
    HIP_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm \\
        /home/danyzhan/MICake/.venv/bin/python experiments/micake-subagent-rsi/scripts/probe_quant_shapes.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

MICAKE_ROOT = Path("/home/danyzhan/MICake")
sys.path.insert(0, str(MICAKE_ROOT))

from experiments.cake_gemm_optimization_workflow_v1 import (  # noqa: E402
    candidate, compile_runtime, contract, emitter, gpu_runtime,
)

SHAPES = (
    ("gemm.small-square", 256, 256, 256),
    ("gemm.large-square", 2048, 2048, 2048),
    ("gemm.tall", 4096, 256, 1024),
    ("gemm.wide", 256, 4096, 1024),
    ("gemm.deep-k", 512, 512, 4096),
    ("gemm.tail-balanced", 272, 272, 288),
    ("gemm.tail-rectangular", 1025, 257, 513),
    ("gemm.square-4k", 4096, 4096, 4096),
    ("gemm.prefill-wide", 8192, 1024, 4096),
    ("gemm.prefill-tall", 1024, 8192, 4096),
    ("gemm.decode", 128, 4096, 4096),
    ("gemm.mlp-7168", 4096, 7168, 2048),
)
CHAMPION = {
    "schedule": {"tile_m": 64, "tile_n": 64, "tile_k": 32, "pipeline_depth": 2, "unroll_factor": 2, "vectorize_factor": 4},
    "realization": {"program_order": "TWO_DIMENSIONAL", "group_m": 1, "num_warps": 8, "num_stages": 2,
                    "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 1},
}
PROTOCOL = {"warmup_count": 2, "paired_repeats": 3, "dispatches_per_block": 4}


def _compile(config: dict, workdir: Path):
    record = candidate.build_candidate(contract.CandidateConfig.from_dict(config))
    source = emitter.emit_candidate_source(record)
    root = workdir / source.source_sha256[:16]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "kernel.py"
    path.write_text(source.source, encoding="utf-8")
    artifact = compile_runtime.compile_candidate(path, root, source.source_sha256, source.entrypoint, record.config)
    return record, artifact


def _speedups(measured: dict) -> dict[str, float]:
    buckets: dict[str, dict[str, list[int]]] = {}
    for sample in measured.get("raw_samples") or []:
        roles = buckets.setdefault(sample["workload_id"], {"incumbent": [], "challenger": []})
        roles[sample["candidate_role"]].append(int(sample["elapsed_ns"]))
    out = {}
    for workload_id, roles in buckets.items():
        if roles["incumbent"] and roles["challenger"]:
            out[workload_id] = statistics_median(roles["incumbent"]) / statistics_median(roles["challenger"])
    return out


def statistics_median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _measure_shape(baseline, champion, shape, workdir_unused=None) -> dict:
    name, m, n, k = shape
    record_b, artifact_b = baseline
    record_c, artifact_c = champion
    session = gpu_runtime.PreparedSession(artifact_b, 0, record_b.config, [{"workload_id": name, "m": m, "n": n, "k": k}])
    session.authorize(f"quant-shape-{name}")
    try:
        with session:
            measured = session.measure_pair(artifact_c, PROTOCOL, record_c.config)
            session.close()
    except Exception as exc:  # noqa: BLE001
        return {"workload": name, "m": m, "n": n, "k": k, "status": type(exc).__name__, "message": str(exc)}
    per = _speedups(measured)
    return {
        "workload": name, "m": m, "n": n, "k": k,
        "status": measured.get("status"),
        "validity": measured.get("measurement_validity"),
        "reason": measured.get("reason_code"),
        "speedup_vs_baseline": per.get(name),
    }


def _fp8_compile(workdir: Path) -> dict:
    record = candidate.build_candidate(contract.CandidateConfig.from_dict(CHAMPION))
    source = emitter.emit_candidate_source(record)
    text = source.source.replace(".to(tl.float16)", ".to(tl.float8e4m3fn)")
    if text == source.source:
        return {"status": "source_unchanged"}
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    root = workdir / "fp8"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "kernel.py"
    path.write_text(text, encoding="utf-8")
    try:
        artifact = compile_runtime.compile_candidate(path, root, digest, source.entrypoint, record.config)
    except compile_runtime.CompileRejected as exc:
        return {"status": "rejected", "code": exc.code, "message": str(exc.__cause__ or exc)}
    except compile_runtime.CompileUnknown as exc:
        return {"status": "unknown", "code": exc.code, "message": str(exc.__cause__ or exc)}
    facts = (artifact.observation.get("binary_facts") or {})
    return {"status": "compiled", "shared_bytes": facts.get("shared_bytes"), "source_sha256": digest}


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="quant-shape-"))
    baseline = _compile(contract.BASELINE_CONFIG.to_dict(), workdir)
    champion = _compile(CHAMPION, workdir)
    rows = []
    for shape in SHAPES:
        row = _measure_shape(baseline, champion, shape)
        rows.append(row)
        print(json.dumps(row), flush=True)
    fp8 = _fp8_compile(workdir)
    print("fp8", json.dumps(fp8), flush=True)
    speedups = [row["speedup_vs_baseline"] for row in rows if row.get("speedup_vs_baseline")]
    document = {
        "input_dtype": "fp16",
        "note": "C144 kernel loads fp16. FP8 was compiled only; the HIP harness still feeds fp16 inputs.",
        "champion": CHAMPION,
        "geomean_speedup_vs_baseline": math.exp(sum(math.log(v) for v in speedups) / len(speedups)) if speedups else None,
        "rows": rows,
        "fp8_compile": fp8,
    }
    out = Path("/home/danyzhan/Lumen-RL/output/micake-subagent-rsi/quant_shape_probe/probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
