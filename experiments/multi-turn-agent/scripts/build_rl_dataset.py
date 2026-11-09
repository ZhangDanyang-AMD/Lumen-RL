"""Build RL prompt dataset from SFT data, regenerating missing harnesses."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_tune_agent.top10_canonical_templates import render_template, Top10Request

SFT_DATA = Path("/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/train.jsonl")
OUTPUT = Path("/home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/rl_prompts.jsonl")
REGEN_ROOT = Path("/home/danyzhan/geak_rl_workspaces")

SYSTEM_MSG = (
    "You are an expert GPU kernel engineer. "
    "Optimize the given kernel for AMD MI300X (gfx942). "
    "Return only a unified diff patch."
)

SHAPE_DIM_NAMES = {
    "rms_norm": ["TOKENS", "HIDDEN"],
    "gemm": ["M", "N", "K"],
    "scaled_quant_gemm": ["M", "N", "K"],
    "blockscale_gemm": ["M", "N", "K"],
    "fused_moe": ["TOKENS", "MODEL", "INTER", "EXPERTS", "TOPK"],
    "all_reduce": ["TOKENS", "HIDDEN"],
    "mha": ["BATCH", "SEQLEN", "HEADS", "HEAD_DIM"],
    "mla": ["BATCH", "SEQLEN", "HEADS", "HEAD_DIM"],
    "paged_attention": ["BATCH", "SEQLEN", "HEADS", "HEAD_DIM"],
    "rope_kv_cache": ["S", "B", "H", "D"],
    "sampling": ["B", "VOCAB"],
    "softmax": ["BATCH", "SEQ"],
}


def _extract_dtype(field):
    """Extract dtype string from nested dict or string."""
    if isinstance(field, dict):
        return field.get("dtype", "bf16") or "bf16"
    return str(field) if field else "bf16"


def _normalize_contract(inner, op):
    """Add missing fields templates expect: operator, shape dict, flat dtypes."""
    inner = dict(inner)
    if "operator" not in inner:
        inner["operator"] = op

    # Convert nested shapes list to named shape dict
    shapes_raw = inner.get("shapes")
    if isinstance(shapes_raw, list) and shapes_raw and isinstance(shapes_raw[0], list):
        dims = shapes_raw[0]
        dim_names = SHAPE_DIM_NAMES.get(op, [f"D{i}" for i in range(len(dims))])
        shape_dict = {}
        for i, val in enumerate(dims):
            name = dim_names[i] if i < len(dim_names) else f"D{i}"
            shape_dict[name] = val
        inner["shape"] = shape_dict

    # Flatten nested dtype dicts to top-level strings for sequence/dense templates
    if "input" in inner and isinstance(inner["input"], dict):
        if "input_dtype" not in inner:
            inner["input_dtype"] = _extract_dtype(inner["input"])
    if "output" in inner and isinstance(inner["output"], dict):
        if "output_dtype" not in inner:
            inner["output_dtype"] = _extract_dtype(inner["output"])
    if "weight" in inner and isinstance(inner["weight"], dict):
        if "weight_dtype" not in inner:
            wd = _extract_dtype(inner["weight"])
            if wd and wd != "None":
                inner["weight_dtype"] = wd

    # Defaults for sequence templates
    if op in ("rope_kv_cache", "sampling") and "mode" not in inner:
        if op == "rope_kv_cache":
            inner["mode"] = "sbhd"
        elif op == "sampling":
            inner["mode"] = "top_k_top_p"

    # rope_kv_cache D must be even
    if op == "rope_kv_cache" and isinstance(inner.get("shape"), dict):
        d = inner["shape"].get("D", 128)
        if d % 2 != 0:
            inner["shape"]["D"] = d + 1

    # blockscale_gemm needs fp8 weight dtype, not bf16
    if op == "blockscale_gemm":
        wd = inner.get("weight_dtype", "")
        if wd in ("bf16", "bfloat16", "", "None", None):
            inner["weight_dtype"] = "fp8_e4m3fnuz"
        # Needs layout
        if "layout" not in inner:
            inner["layout"] = "TN"
        # Needs block_k
        if "block_k" not in inner:
            inner["block_k"] = 128
        # Needs accum_dtype
        if "accum_dtype" not in inner:
            inner["accum_dtype"] = "fp32"
        if "scale" not in inner or not isinstance(inner.get("scale"), dict):
            inner["scale"] = {
                "activation_block": [1, 128],
                "weight_block": [128, 128],
            }
        if "scale_block_k" not in inner:
            inner["scale_block_k"] = 128
        if "mode" not in inner:
            inner["mode"] = "fp8_block128"

    # fused_moe needs bf16 input
    if op == "fused_moe":
        inner.setdefault("input_dtype", "bf16")
        inner.setdefault("output_dtype", "bf16")

    # all_reduce needs world_size
    if op == "all_reduce":
        inner.setdefault("world_size", 2)

    return inner


def main():
    total = 0
    existing = 0
    regenerated = 0
    failed = 0
    fail_ops = {}

    with open(SFT_DATA) as f_in, open(OUTPUT, "w") as f_out:
        for line in f_in:
            total += 1
            row = json.loads(line)

            inp = row.get("input", {})
            contract = inp.get("contract", {})
            parent_source = inp.get("parent_source", {})
            baseline = inp.get("baseline", {})
            task_type = row.get("task_type", "cold_start")
            sample_id = row.get("sample_id", f"task_{total}")
            op = contract.get("operator", "")
            backend = contract.get("backend", "")

            kernel_path = contract.get("kernel_path", "")
            task_dir = Path(kernel_path) if kernel_path else None

            if task_dir and task_dir.exists() and (task_dir / "scripts" / "task_runner.py").exists():
                existing += 1
            else:
                inner = contract.get("contract", {})
                prov = contract.get("provenance", {}).get("case_seed", {})

                try:
                    normalized = _normalize_contract(inner, op)
                    shapes_raw = inner.get("shapes", [[]])
                    shape = tuple(shapes_raw[0]) if shapes_raw and shapes_raw[0] else ()

                    req = Top10Request(
                        request_id=contract.get("case_id", sample_id),
                        request_text=contract.get("direction", ""),
                        family=op,
                        language="triton" if backend == "triton" else "hip",
                        shape=shape,
                        seed_provenance=prov,
                        recognized_contract={"contract": normalized},
                    )
                    rendered = render_template(req)

                    ws_dir = REGEN_ROOT / sample_id
                    ws_dir.mkdir(parents=True, exist_ok=True)

                    kernel_src = parent_source.get("kernel.py", rendered.get("kernel.py", ""))
                    (ws_dir / "kernel.py").write_text(kernel_src)
                    (ws_dir / "config.yaml").write_text(rendered["config.yaml"])
                    scripts_dir = ws_dir / "scripts"
                    scripts_dir.mkdir(exist_ok=True)
                    (scripts_dir / "task_runner.py").write_text(rendered["scripts/task_runner.py"])
                    (ws_dir / "metadata.json").write_text(rendered["metadata.json"])

                    task_dir = ws_dir
                    regenerated += 1
                except Exception as e:
                    failed += 1
                    key = f"{op}/{backend}"
                    if key not in fail_ops:
                        fail_ops[key] = str(e)[:120]
                    continue

            frozen_input = json.dumps({
                "input": {
                    "contract": contract,
                    "parent_source": parent_source,
                    "baseline": {"commands": baseline.get("commands", {})},
                    "direction": inp.get("direction"),
                    "profile": inp.get("profile"),
                    "error_feedback": None,
                },
                "task_type": task_type,
            })

            prompt_messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": frozen_input},
            ]

            gt_meta = json.dumps({
                "task_id": sample_id,
                "task_dir": str(task_dir),
                "baseline_ms": baseline.get("geomean_ms", 1.0),
                "family": op,
            })

            rl_row = {
                "prompt": prompt_messages,
                "reward_model": {"ground_truth": gt_meta},
            }
            f_out.write(json.dumps(rl_row) + "\n")

    written = existing + regenerated
    print(f"Total SFT samples: {total}")
    print(f"Existing harness: {existing}")
    print(f"Regenerated: {regenerated}")
    print(f"Failed: {failed}")
    print(f"Written to RL dataset: {written}")

    if fail_ops:
        print("\nFailed operators:")
        for k, v in fail_ops.items():
            print(f"  {k}: {v}")

    # Distribution
    families = {}
    with open(OUTPUT) as f:
        for line in f:
            gt = json.loads(json.loads(line)["reward_model"]["ground_truth"])
            fam = gt["family"]
            families[fam] = families.get(fam, 0) + 1
    print(f"\nOperator distribution ({sum(families.values())} tasks):")
    for k, v in sorted(families.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
