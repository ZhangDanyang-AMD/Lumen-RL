#!/usr/bin/env python3
"""Measure real FlyDSL baseline kernel performance — subprocess-isolated.

V6: Each benchmark runs in a subprocess so one GPU fault can't kill the rest.
Uses ptr_arg() wrapping and calling conventions from aiter test files.

Run inside aiter-baseline-bench:
  docker exec -e HIP_VISIBLE_DEVICES=2 aiter-baseline-bench \
    python3 /home/danyzhan/Lumen-RL/experiments/multi-tune-agent/scripts/measure_flydsl_baselines_v6.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ARTIFACTS = Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel")
OUTPUT = Path("/home/danyzhan/agent-phase2-held-out-aiter/receipts/flydsl_real_baselines.json")
OVERRIDE_OUTPUT = Path("/home/danyzhan/agent-phase2-held-out-aiter/receipts/baseline_overrides.json")
TIMEOUT = 600

BENCH_SCRIPT = Path(__file__).parent / "_bench_one_flydsl.py"


def get_flydsl_tasks():
    tasks = []
    for task_dir in sorted(ARTIFACTS.iterdir()):
        if not task_dir.is_dir():
            continue
        cfg_path = task_dir / "initial" / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("language") == "flydsl":
            meta_path = task_dir / "initial" / "metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            tasks.append((task_dir.name, task_dir, cfg, meta))
    return tasks


def run_bench_subprocess(source_key):
    """Run a single benchmark in a subprocess and return (ms, error)."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        r = subprocess.run(
            [sys.executable, str(BENCH_SCRIPT), source_key],
            capture_output=True, text=True, timeout=TIMEOUT, env=env,
        )
        output = r.stdout + "\n" + r.stderr
        m = re.search(r'RESULT_MS:\s*([0-9]+(?:\.[0-9]+)?)', output)
        if m:
            return float(m.group(1)), None
        em = re.search(r'RESULT_ERR:\s*(.*)', output)
        if em:
            return 0.0, em.group(1).strip()[:200]
        return 0.0, f"no result (rc={r.returncode}): {output[-500:]}"
    except subprocess.TimeoutExpired:
        return 0.0, f"timeout ({TIMEOUT}s)"
    except Exception as e:
        return 0.0, str(e)[:200]


# Source → bench key mapping
SOURCE_BENCH_KEYS = {
    "aiter/ops/flydsl/kernels/fused_compress_attn.py": "compress_attn",
    "aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py": "compress_attn",
    "aiter/ops/flydsl/causal_conv1d_flydsl.py": "causal_conv1d",
    "aiter/ops/flydsl/kernels/causal_conv1d_update.py": "causal_conv1d",
    "aiter/ops/flydsl/kernels/chunk_gated_delta_h_mfma16x16x16.py": "chunk_gated_delta",
    "aiter/ops/flydsl/kernels/gdn_prepare.py": "gdn_prepare",
    "aiter/ops/flydsl/kernels/chunk_gated_delta_h.py": "chunk_gated_delta",
    "aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py": "dispatch_combine",
    "aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py": "dispatch_combine",
    "aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_kernel.py": "fmha_gfx950",
    "aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py": "gemm_a8w8",
    "aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py": "gemm_a8w8",
    "aiter/ops/flydsl/kernels/gemm_common_gfx1250.py": "hgemm",
    "aiter/ops/flydsl/kernels/buffer_ops.py": "hgemm",
    "aiter/ops/flydsl/kernels/mega_moe/gemm_util.py": "hgemm",
    "aiter/ops/flydsl/kernels/mla_reduce.py": "mla_reduce",
    "aiter/ops/flydsl/kernels/moe_contiguous_psum.py": "moe_contiguous_psum",
    "aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py": "moe_2stage_gemm",
    "aiter/ops/flydsl/kernels/moe_m_tile_map.py": "moe_route_maps",
    "aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py": "moe_2stage_gemm",
    "aiter/ops/flydsl/kernels/moe_gather_reduce.py": "moe_gather_reduce",
    "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py": "moe_2stage_gemm",
    "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py": "moe_2stage_gemm",
    "aiter/aot/flydsl/moe.py": "moe_2stage_gemm",
    "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm2.py": "hgemm",
    "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm1.py": "hgemm",
    "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py": "hgemm",
    "aiter/ops/flydsl/kernels/moe_reduce.py": "moe_reduce",
    "aiter/ops/flydsl/kernels/moe_g2l_lut.py": "moe_g2l_lut",
    "aiter/ops/flydsl/kernels/moe_route_maps.py": "moe_route_maps",
    "aiter/ops/flydsl/kernels/mega_moe/dispatch.py": "moe_route_maps",
    "aiter/ops/flydsl/kernels/moe_sorting_kernel.py": "moe_sorting",
    "aiter/ops/flydsl/kernels/mxfp4_gemm2.py": "preshuffle_gemm",
    "aiter/ops/flydsl/kernels/mxfp4_gemm1.py": "preshuffle_gemm",
    "aiter/ops/flydsl/batched_gemm_mxfp4.py": "preshuffle_gemm",
    "aiter/ops/flydsl/kernels/preshuffle_gemm.py": "preshuffle_gemm",
    "aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py": "preshuffle_gemm",
    "aiter/ops/flydsl/kernels/mega_moe/quant.py": "quantization",
    "aiter/ops/flydsl/kernels/silu_and_mul_fq.py": "silu_and_mul_fq",
    "aiter/ops/flydsl/kernels/splitk_hgemm.py": "hgemm",
    "aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py": "chunk_gated_delta",
    "aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py": "gdn_prepare",
    "aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py": "topk",
}


def main():
    tasks = get_flydsl_tasks()
    print(f"Found {len(tasks)} flydsl tasks\n")

    results = {}
    bench_cache = {}

    for i, (task_id, task_dir, cfg, meta) in enumerate(tasks, 1):
        source = meta.get("source", "?")
        source_key = source.split("::")[0] if "::" in source else source
        operator = cfg.get("operator", "unknown")
        cases = cfg.get("benchmark_shapes", [])
        case_id = cases[0].get("case_id", task_id) if cases else task_id
        old_bl = cfg.get("baseline_ms", {})
        old_ms = list(old_bl.values())[0] if old_bl else 0.0

        print(f"[{i}/{len(tasks)}] {task_id}")
        print(f"  source={source_key}, op={operator}")

        bench_key = SOURCE_BENCH_KEYS.get(source_key)
        if bench_key is None:
            ms, err = 0.0, f"no bench for {source_key}"
            print(f"  SKIP: {err}")
        elif bench_key == "dispatch_combine":
            ms, err = 0.0, "requires multi-GPU (mori)"
            print(f"  SKIP: {err}")
        elif bench_key in bench_cache:
            ms, err = bench_cache[bench_key]
            if ms > 0:
                ratio = old_ms / ms if old_ms > 0 else 0
                print(f"  CACHED: {ms:.6f} ms (old={old_ms:.6f}, ratio={ratio:.2f}x)")
            else:
                print(f"  CACHED FAIL: {err}")
        else:
            print(f"  Running bench '{bench_key}' in subprocess...")
            ms, err = run_bench_subprocess(bench_key)
            bench_cache[bench_key] = (ms, err)
            if ms > 0:
                ratio = old_ms / ms if old_ms > 0 else 0
                print(f"  OK: {ms:.6f} ms (old={old_ms:.6f}, ratio={ratio:.2f}x)")
            else:
                print(f"  FAIL: {err}")

        results[task_id] = {
            "task_id": task_id,
            "case_id": case_id,
            "operator": operator,
            "source": source,
            "old_baseline_ms": old_ms,
            "measured_ms": ms,
            "error": err,
            "ratio": old_ms / ms if ms > 0 and old_ms > 0 else None,
        }

    # Write results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)

    # Update baseline_overrides
    existing = json.loads(OVERRIDE_OUTPUT.read_text()) if OVERRIDE_OUTPUT.exists() else {}
    updated = 0
    for r in results.values():
        if r["measured_ms"] > 0:
            existing[r["task_id"]] = {
                "per_case_ms": {r["case_id"]: r["measured_ms"]},
                "geomean_ms": r["measured_ms"],
                "repeats": 50,
                "measured": True,
            }
            updated += 1
    with open(OVERRIDE_OUTPUT, 'w') as f:
        json.dump(existing, f, indent=2)

    # Summary
    measured = [r for r in results.values() if r["measured_ms"] > 0]
    failed = [r for r in results.values() if r["measured_ms"] == 0]
    print(f"\n{'='*70}")
    print(f"Summary: {len(measured)}/{len(results)} measured, {len(failed)} failed")
    print(f"\nMeasured:")
    for r in sorted(measured, key=lambda x: x["operator"]):
        ratio = r["ratio"]
        f = " ***" if ratio and (ratio > 5 or ratio < 0.2) else ""
        print(f"  {r['task_id']}: {r['measured_ms']:.6f} ms (old={r['old_baseline_ms']:.6f}, r={ratio:.2f}x){f}")
    print(f"\nFailed ({len(failed)}):")
    for r in sorted(failed, key=lambda x: x["operator"]):
        print(f"  {r['task_id']}: {(r.get('error') or '?')[:80]}")


if __name__ == "__main__":
    main()
