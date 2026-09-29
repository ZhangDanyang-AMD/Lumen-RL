"""Build 50 Triton + 50 HIP gfx942 benchmark manifest, excluding training data."""

import hashlib
import json
import yaml
from pathlib import Path
from collections import defaultdict


EXTRA_CONTRACTS = [
    # GEMM with novel shapes not in training data
    {"operator": "gemm", "shape": {"M": 256, "N": 3072, "K": 1024}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "TN"},
    {"operator": "gemm", "shape": {"M": 512, "N": 2048, "K": 512}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "TN"},
    {"operator": "gemm", "shape": {"M": 64, "N": 5120, "K": 2048}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "TN"},
    {"operator": "gemm", "shape": {"M": 1024, "N": 1024, "K": 1024}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "TN"},
    {"operator": "gemm", "shape": {"M": 2048, "N": 4096, "K": 768}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "TN"},
    {"operator": "gemm", "shape": {"M": 16, "N": 8192, "K": 3072}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "TN"},
    # Softmax with novel shapes
    {"operator": "softmax", "shape": {"M": 512, "N": 512}, "dtype": {"input": "bf16", "output": "bf16"}, "axis": 1, "layout": "row_major"},
    {"operator": "softmax", "shape": {"M": 2048, "N": 256}, "dtype": {"input": "bf16", "output": "bf16"}, "axis": 1, "layout": "row_major"},
    {"operator": "softmax", "shape": {"M": 128, "N": 4096}, "dtype": {"input": "bf16", "output": "bf16"}, "axis": 1, "layout": "row_major"},
    # RMSNorm with novel shapes
    {"operator": "rms_norm", "shape": {"M": 128, "N": 2048}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "contiguous"},
    {"operator": "rms_norm", "shape": {"M": 256, "N": 8192}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "contiguous"},
    {"operator": "rms_norm", "shape": {"M": 64, "N": 3072}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "contiguous"},
    # SiLU and Mul
    {"operator": "silu_and_mul", "shape": {"M": 256, "N": 3072}, "dtype": {"input": "bf16", "output": "bf16"}, "layout": "contiguous"},
    {"operator": "silu_and_mul", "shape": {"M": 1024, "N": 1536}, "dtype": {"input": "bf16", "output": "bf16"}, "layout": "contiguous"},
    # Fused add rmsnorm
    {"operator": "fused_add_rms_norm", "shape": {"M": 128, "N": 3072}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "contiguous"},
    {"operator": "fused_add_rms_norm", "shape": {"M": 256, "N": 2048}, "dtype": {"accum": "fp32", "input": "bf16", "output": "bf16", "weight": "bf16"}, "layout": "contiguous"},
    # Dynamic quant with novel shapes
    {"operator": "dynamic_per_token_quant", "shape": {"M": 64, "N": 2048}, "dtype": {"input": "bf16", "output": "fp8_e4m3fnuz", "scale": "fp32_per_token"}, "layout": "contiguous"},
    {"operator": "dynamic_per_token_quant", "shape": {"M": 512, "N": 4096}, "dtype": {"input": "bf16", "output": "fp8_e4m3fnuz", "scale": "fp32_per_token"}, "layout": "contiguous"},
    {"operator": "dynamic_per_tensor_quant", "shape": {"M": 128, "N": 1024}, "dtype": {"input": "bf16", "output": "fp8_e4m3fnuz", "scale": "fp32_scalar"}, "layout": "contiguous"},
    # Static quant
    {"operator": "static_per_tensor_quant", "shape": {"M": 256, "N": 3072}, "dtype": {"input": "bf16", "output": "fp8_e4m3fnuz", "scale": "fp32_scalar"}, "layout": "contiguous"},
    {"operator": "static_per_tensor_quant", "shape": {"M": 64, "N": 8192}, "dtype": {"input": "bf16", "output": "fp8_e4m3fnuz", "scale": "fp32_scalar"}, "layout": "contiguous"},
    # Fused SiLU mul
    {"operator": "fused_silu_mul", "shape": {"M": 512, "N": 2048}, "dtype": {"input": "bf16", "output": "bf16"}, "layout": "contiguous"},
]

ORACLES = {
    "gemm": {"method": "torch.nn.functional.linear", "tier": "pure_torch"},
    "softmax": {"method": "torch.softmax", "tier": "pure_torch"},
    "rms_norm": {"method": "torch.nn.functional.rms_norm", "tier": "pure_torch"},
    "silu_and_mul": {"method": "torch_silu_mul", "tier": "pure_torch"},
    "fused_add_rms_norm": {"method": "torch_add_rms_norm", "tier": "pure_torch"},
    "dynamic_per_token_quant": {"method": "torch_quantize", "tier": "pure_torch"},
    "dynamic_per_tensor_quant": {"method": "torch_quantize", "tier": "pure_torch"},
    "static_per_tensor_quant": {"method": "torch_quantize", "tier": "pure_torch"},
    "fused_silu_mul": {"method": "torch_fused_silu_mul", "tier": "pure_torch"},
}


def build_manifest(candidates_path, output_path, exclusions_path, n_per_lane=50):
    with open(candidates_path) as f:
        data = yaml.safe_load(f)
    candidates = data.get("candidates", [])

    with open(exclusions_path) as f:
        exclusions = json.load(f)
    excluded_hashes = set(exclusions.get("contract_hashes", []))
    excluded_op_shapes = set(exclusions.get("op_shape_keys", []))

    by_lane = defaultdict(list)
    for c in candidates:
        for lane in c.get("target_lanes", []):
            if lane in ("triton_gfx942", "hip_gfx942"):
                by_lane[lane].append(c)

    selected = []
    excluded_count = 0

    for lane in ("triton_gfx942", "hip_gfx942"):
        lang = "TRITON" if "triton" in lane else "HIP"
        count = 0
        seen_ops = defaultdict(int)

        # First pass: candidates from expanded pool
        for c in by_lane[lane]:
            if count >= n_per_lane:
                break
            op = c["contract"]["operator"]
            if seen_ops[op] >= 15:
                continue

            contract_json = json.dumps(c["contract"], sort_keys=True)
            contract_hash = hashlib.sha256(contract_json.encode()).hexdigest()
            op_shape_key = f"{op}|{json.dumps(c['contract'].get('shape', {}), sort_keys=True)}"

            if contract_hash in excluded_hashes or op_shape_key in excluded_op_shapes:
                excluded_count += 1
                continue

            oracle = c.get("oracle", ORACLES.get(op, {"method": "torch_reference", "tier": "pure_torch"}))
            oracle_json = json.dumps(oracle, sort_keys=True)
            req_id = f"bench-{lane.replace('_', '-')}-{op}-{count:03d}"

            entry = {
                "id": req_id,
                "request": f"Generate a standalone {lang} {op} kernel for AMD gfx942. Frozen contract JSON: {contract_json}. Independent oracle JSON: {oracle_json}. Implement independently; do not call AITER, CK, ASM, OPUS, hipBLASLt, torch operators, or the oracle at runtime. Create a deterministic GEAK harness and require compile, correctness, performance, and fresh-workspace verification.",
                "recognized_contract": contract_json,
                "seed_provenance": {"contract_hash": contract_hash, "source_candidate_id": c["id"], "target_lane": lane, "split_group": "held_out"},
            }
            selected.append(entry)
            seen_ops[op] += 1
            count += 1

        # Second pass: extra synthetic contracts for remaining slots
        for extra in EXTRA_CONTRACTS:
            if count >= n_per_lane:
                break
            op = extra["operator"]
            contract_json = json.dumps(extra, sort_keys=True)
            contract_hash = hashlib.sha256(contract_json.encode()).hexdigest()
            op_shape_key = f"{op}|{json.dumps(extra.get('shape', {}), sort_keys=True)}"

            if contract_hash in excluded_hashes or op_shape_key in excluded_op_shapes:
                continue

            oracle = ORACLES.get(op, {"method": "torch_reference", "tier": "pure_torch"})
            oracle_json = json.dumps(oracle, sort_keys=True)
            req_id = f"bench-{lane.replace('_', '-')}-{op}-{count:03d}"

            entry = {
                "id": req_id,
                "request": f"Generate a standalone {lang} {op} kernel for AMD gfx942. Frozen contract JSON: {contract_json}. Independent oracle JSON: {oracle_json}. Implement independently; do not call AITER, CK, ASM, OPUS, hipBLASLt, torch operators, or the oracle at runtime. Create a deterministic GEAK harness and require compile, correctness, performance, and fresh-workspace verification.",
                "recognized_contract": contract_json,
                "seed_provenance": {"contract_hash": contract_hash, "source_candidate_id": f"synthetic-{op}-{count}", "target_lane": lane, "split_group": "held_out"},
            }
            selected.append(entry)
            seen_ops[op] += 1
            count += 1

    manifest = {"version": 1, "purpose": "benchmark-sft-eval-100-no-overlap", "requests": selected}
    with open(output_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, width=200, allow_unicode=True)

    triton_count = sum(1 for s in selected if "triton" in s["seed_provenance"]["target_lane"])
    hip_count = sum(1 for s in selected if "hip" in s["seed_provenance"]["target_lane"])
    print(f"Excluded {excluded_count} candidates overlapping with training data")
    print(f"Generated manifest: {len(selected)} entries ({triton_count} Triton, {hip_count} HIP)")

    op_counts = defaultdict(lambda: defaultdict(int))
    for s in selected:
        lane = s["seed_provenance"]["target_lane"]
        contract = json.loads(s["recognized_contract"])
        op_counts[lane][contract["operator"]] += 1
    for lane, ops in sorted(op_counts.items()):
        print(f"  {lane}: {dict(ops)}")


if __name__ == "__main__":
    build_manifest(
        "cases/phase1-case-candidates-gfx942-expanded.yaml",
        "cases/benchmark-100-generation-requests.yaml",
        "/tmp/train_exclusions.json",
        n_per_lane=50,
    )
