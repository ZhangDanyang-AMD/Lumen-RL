"""Build AITER-baseline held-out dataset: replace template-generated kernels with AITER implementations."""

import json
import shutil
from pathlib import Path

SRC_DIR = Path("/home/danyzhan/held-out-benchmark")
AITER_KERNELS = Path("/home/danyzhan/aiter_standalone_kernels")
OUT_DIR = Path("/home/danyzhan/held-out-benchmark-aiter")
PROD_SHAPES = AITER_KERNELS / "production_shapes.json"

AITER_REPO = "https://github.com/ROCm/aiter.git"
AITER_LOCAL = Path("/home/danyzhan/Lumen/third_party/aiter")

# Map (operator, lane) -> standalone kernel file
KERNEL_MAP = {}
for f in AITER_KERNELS.glob("*.py"):
    name = f.stem
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        op, lane = parts
        KERNEL_MAP[(op, lane)] = f

# AITER source provenance for each operator
AITER_PROVENANCE = {
    ("rms_norm", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py",
        "aiter_wrapper": "aiter/ops/triton/normalization/rmsnorm.py",
        "contract_test": "op_tests/triton_tests/normalization/test_rmsnorm.py",
        "description": "Persistent-grid RMSNorm with blocked/non-blocked paths, 128-bit loads, fp32 accumulate",
    },
    ("rms_norm", "hip"): {
        "aiter_source": "vllm/csrc/layernorm_kernels.cu (reference), GEAK perf_knowledge",
        "contract_test": "op_tests/triton_tests/normalization/test_rmsnorm.py",
        "description": "HIP vectorized RMSNorm with warp shuffle reduce, float4 I/O",
    },
    ("gemm", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py",
        "contract_test": "op_tests/test_gemm_a16w16.py",
        "description": "Tiled BF16 GEMM (TN layout) with group-M scheduling",
    },
    ("gemm", "hip"): {
        "aiter_source": "GEAK perf_knowledge/operators/gemm/backends/hip.md",
        "contract_test": "op_tests/test_gemm_a16w16.py",
        "description": "HIP tiled GEMM with shared memory, vectorized loads",
    },
    ("mha", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/",
        "contract_test": "op_tests/triton_tests/attention/test_mha_with_sink.py",
        "description": "Flash attention forward with causal masking, tiled Q/K/V",
    },
    ("mha", "hip"): {
        "aiter_source": "GEAK perf_knowledge/operators/attention_prefill_fmha/backends/hip.md",
        "contract_test": "op_tests/triton_tests/attention/test_mha_with_sink.py",
        "description": "HIP flash attention with shared memory tiling",
    },
    ("mla", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/attention/mla.py",
        "contract_test": "op_tests/test_mla.py",
        "description": "Multi-Latent Attention decode kernel for DeepSeek-V3/Qwen3",
    },
    ("paged_attention", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/attention/",
        "contract_test": "op_tests/test_pa_v1.py",
        "description": "Paged attention with block table lookup, decode-optimized",
    },
    ("paged_attention", "hip"): {
        "aiter_source": "GEAK perf_knowledge/operators/paged_attention/backends/hip.md",
        "contract_test": "op_tests/test_pa_v1.py",
        "description": "HIP paged attention with vectorized KV cache access",
    },
    ("rope_kv_cache", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/rope/rotary_embedding.py",
        "contract_test": "op_tests/test_rope.py",
        "description": "Fused RoPE rotation + KV cache write",
    },
    ("rope_kv_cache", "hip"): {
        "aiter_source": "GEAK perf_knowledge, aiter/ops/cache.py reference",
        "contract_test": "op_tests/test_rope.py",
        "description": "HIP fused RoPE + KV cache with vectorized I/O",
    },
    ("fused_moe", "triton"): {
        "aiter_source": "aiter/ops/triton/_triton_kernels/moe/",
        "contract_test": "op_tests/test_moe.py",
        "description": "MoE token routing + expert GEMM, top-K gating",
    },
    ("fused_moe", "hip"): {
        "aiter_source": "GEAK perf_knowledge/operators/fused_moe/backends/hip.md",
        "contract_test": "op_tests/test_moe.py",
        "description": "HIP MoE with shared memory routing",
    },
    ("sampling", "triton"): {
        "aiter_source": "aiter/ops/sample.py (reference), custom Triton impl",
        "contract_test": "op_tests/test_sampling.py",
        "description": "Triton top-k/top-p sampling with temperature scaling",
    },
    ("sampling", "hip"): {
        "aiter_source": "GEAK perf_knowledge, custom HIP impl",
        "contract_test": "op_tests/test_sampling.py",
        "description": "HIP top-k/top-p sampling with warp-level sort",
    },
    ("blockscale_gemm", "triton"): {
        "aiter_source": "aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py",
        "contract_test": "op_tests/test_gemm_a8w8_blockscale.py",
        "description": "FP8 block-scaled GEMM with [1,128]/[128,128] block scales",
    },
}


def get_op_lane(task_id):
    parts = task_id.split("-")
    op = parts[2] if len(parts) > 2 else ""
    lane = "triton" if "triton" in task_id else "hip"
    return op, lane


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy structure
    for item in ["tasks", "README.md", "manifest.json", "checksums.sha256", "license-evidence"]:
        src = SRC_DIR / item
        dst = OUT_DIR / item
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Process each task
    artifacts_dir = OUT_DIR / "artifacts" / "kernel"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(SRC_DIR / "tasks" / "kernel.jsonl") as f:
        tasks = [json.loads(l) for l in f]

    updated = 0
    skipped = 0
    stats = {"replaced": [], "kept_original": [], "skipped": []}

    for task in tasks:
        tid = task["task_id"]
        op, lane = get_op_lane(tid)

        src_task = SRC_DIR / "artifacts" / "kernel" / tid / "initial"
        dst_task = artifacts_dir / tid / "initial"

        if not src_task.exists():
            stats["skipped"].append(tid)
            skipped += 1
            continue

        # Copy full task directory
        if dst_task.exists():
            shutil.rmtree(dst_task)
        shutil.copytree(src_task, dst_task)

        # Replace kernel.py with AITER version if available
        aiter_key = (op, lane)
        if aiter_key in KERNEL_MAP:
            aiter_src = KERNEL_MAP[aiter_key]
            dst_kernel = dst_task / "kernel.py"

            aiter_code = aiter_src.read_text()

            # Keep original template kernel as backup
            if dst_kernel.exists():
                shutil.copy2(dst_kernel, dst_task / "kernel_template_original.py")

            dst_kernel.write_text(aiter_code)

            # Write AITER provenance metadata
            prov = AITER_PROVENANCE.get(aiter_key, {})
            held_meta_path = dst_task / "metadata.json"
            if held_meta_path.exists():
                held_meta = json.loads(held_meta_path.read_text())
            else:
                held_meta = {}
            held_meta["aiter_baseline"] = {
                "source_repo": AITER_REPO,
                "aiter_local_path": str(AITER_LOCAL),
                "aiter_kernel_source": prov.get("aiter_source", ""),
                "aiter_wrapper": prov.get("aiter_wrapper", ""),
                "contract_shape_source": prov.get("contract_test", ""),
                "description": prov.get("description", ""),
                "standalone_file": aiter_src.name,
                "kernel_baseline_type": "aiter",
                "original_template_backup": "kernel_template_original.py",
            }
            # Add shape/dtype from the task contract
            task_meta = next((t for t in tasks if t["task_id"] == tid), {})
            contract = task_meta.get("contract", "")
            if isinstance(contract, str):
                try:
                    contract = eval(contract)
                except Exception:
                    contract = {}
            if isinstance(contract, dict):
                kc = contract.get("kernel_contract", {})
                held_meta["aiter_baseline"]["shape"] = kc.get("shape", {})
                held_meta["aiter_baseline"]["input_dtype"] = kc.get("input_dtype", "")
                held_meta["aiter_baseline"]["output_dtype"] = kc.get("output_dtype", "")

            held_meta_path.write_text(json.dumps(held_meta, indent=2))
            updated += 1
            stats["replaced"].append(f"{tid} <- {aiter_src.name}")
        else:
            stats["kept_original"].append(f"{tid} (no AITER for {op}/{lane})")

    # Update tasks jsonl with aiter metadata
    updated_tasks = []
    for task in tasks:
        tid = task["task_id"]
        op, lane = get_op_lane(tid)
        task["kernel_baseline"] = "aiter" if (op, lane) in KERNEL_MAP else "template"
        updated_tasks.append(task)

    with open(OUT_DIR / "tasks" / "kernel.jsonl", "w") as f:
        for t in updated_tasks:
            f.write(json.dumps(t) + "\n")

    # Write dataset info
    info = {
        "name": "agent-phase1-held-out-aiter",
        "description": "Held-out kernel optimization tasks with AITER-level baseline kernels",
        "base_dataset": "agent-phase1-held-out-private",
        "modification": "Replaced template-generated naive kernels with standalone AITER implementations",
        "total_tasks": len(tasks),
        "aiter_replaced": updated,
        "kept_original": len(stats["kept_original"]),
        "skipped": skipped,
        "operators": {
            "aiter_triton": [k[0] for k in KERNEL_MAP if k[1] == "triton"],
            "aiter_hip": [k[0] for k in KERNEL_MAP if k[1] == "hip"],
        },
    }
    (OUT_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2))

    print(f"Total tasks: {len(tasks)}")
    print(f"AITER replaced: {updated}")
    print(f"Kept original: {len(stats['kept_original'])}")
    print(f"Skipped: {skipped}")
    print(f"\nReplaced by operator:")
    from collections import Counter
    replaced_ops = Counter()
    for item in stats["replaced"]:
        op = item.split()[0].split("-")[2]
        replaced_ops[op] += 1
    for op, cnt in replaced_ops.most_common():
        print(f"  {op}: {cnt}")
    print(f"\nKept original (no AITER):")
    for item in stats["kept_original"]:
        print(f"  {item}")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
