"""Deterministic Top10 gfx942 inventory derived from explicit AITER tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

AITER_ROOT = Path("/home/danyzhan/aiter")
AITER_GIT_SHA = "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
TOP10_FAMILIES = (
    "mha",
    "mla",
    "paged_attention",
    "fused_moe",
    "gemm",
    "rms_norm",
    "rope_kv_cache",
    "blockscale_gemm",
    "all_reduce",
    "sampling",
)
DUAL_LANES = ("hip_gfx942", "triton_gfx942")

# These are deliberately selected source contracts, not recursive discovery counts.
# Every family has two independent source lineages.
SOURCES: tuple[dict[str, Any], ...] = (
    {"family": "mha", "path": "op_tests/triton_tests/attention/test_mha.py", "test": "test_mha", "sha": "4d7a96c8277dfe419cefb9506de63aa3c958b2bdf13ff6a7e787ec59cd46ec95", "mode": "causal", "shape": {"B": 1, "SQ": 128, "SK": 128, "HQ": 8, "HK": 8, "D": 128}},
    {"family": "mha", "path": "op_tests/triton_tests/attention/test_mha_with_sink.py", "test": "test_mha_with_sink", "sha": "8d6b5733e7ac691d1ae3103d628ec3d5a45404addc5ef125643aebab4148f3ab", "mode": "attention_sink", "shape": {"B": 2, "SQ": 256, "SK": 512, "HQ": 16, "HK": 4, "D": 128}},
    {"family": "mla", "path": "op_tests/test_mla.py", "test": "test_mla", "sha": "0c41314db2b155f2331dbcc59df12662450c025e6bfe970cb05539ef4a74a318", "mode": "decode", "shape": {"B": 4, "S": 2048, "H": 16, "KV": 512, "ROPE": 64}},
    {"family": "mla", "path": "op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py", "test": "test_triton_unified_attn", "sha": "33a45b9576e773a091e05f741d14a0a4c27f98841cf1d227ecaabd09ce16da53", "mode": "sparse", "shape": {"B": 2, "S": 4096, "H": 16, "KV": 512, "ROPE": 64}},
    {"family": "paged_attention", "path": "op_tests/triton_tests/attention/test_pa_decode.py", "test": "test_paged_attn", "sha": "4d76d43a7a4cb5848cd5827c686d51834eea9d12842d097f64a46a75d7ecab7a", "mode": "decode", "shape": {"B": 4, "HQ": 8, "HK": 1, "BLOCK": 16, "S": 1024, "D": 128}},
    {"family": "paged_attention", "path": "op_tests/test_pa_v1.py", "test": "test_paged_attention", "sha": "a755b2a29b2a157f80c70e92567c7062a97efdf2991549d3c9c94dad6a570432", "mode": "sliding_window", "shape": {"B": 8, "HQ": 16, "HK": 4, "BLOCK": 16, "S": 2048, "D": 128}},
    {"family": "fused_moe", "path": "op_tests/test_moe.py", "test": "test_fmoe", "sha": "e9a91a1f9c4edaaa7e23768820047002bd64990e2bfc3e874a6a91dc5416ee53", "mode": "silu", "shape": {"TOKENS": 128, "MODEL": 4096, "INTER": 14336, "EXPERTS": 8, "TOPK": 2}},
    {"family": "fused_moe", "path": "op_tests/triton_tests/moe/test_moe_gemm_a8w8.py", "test": "test_op", "sha": "ef6c8ba6fe452e8748c11eefd07ab48e0f991a7487bc5a643f4ffe2af825940b", "mode": "fp8", "shape": {"TOKENS": 64, "MODEL": 4096, "INTER": 8192, "EXPERTS": 8, "TOPK": 2}},
    {"family": "gemm", "path": "op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py", "test": "test_gemm_a16_w16", "sha": "07bb9e3f7ef794ea8e4a962b8083c7469fb132b3fe7bf9a627d59ad73a896172", "mode": "bf16", "shape": {"M": 128, "N": 4096, "K": 4096}},
    {"family": "gemm", "path": "op_tests/test_gemm_a16w16.py", "test": "test_gemm", "sha": "9fb8baa1f7aa6526e274a907a9e8b25dfa04de76c4cff5350dfdd031e810fb46", "mode": "narrow_m", "shape": {"M": 16, "N": 8192, "K": 4096}},
    {"family": "gemm", "path": "op_tests/test_gemm_a16w16.py", "test": "test_skinny_gemm", "sha": "9fb8baa1f7aa6526e274a907a9e8b25dfa04de76c4cff5350dfdd031e810fb46", "mode": "skinny", "shape": {"M": 4, "N": 4096, "K": 8192}},
    {"family": "rms_norm", "path": "op_tests/triton_tests/normalization/test_rmsnorm.py", "test": "test_rmsnorm", "sha": "a4e9977897e44ab54295eb7e79c9d52fced26596db9ee9b4c4d161a954367098", "mode": "plain", "shape": {"M": 128, "N": 4096}},
    {"family": "rms_norm", "path": "op_tests/test_rmsnorm2d.py", "test": "test_rmsnorm2d", "sha": "3813dc34558d086bd50cb093f0d01e990cbc421538160cdd57856a279c94eaa3", "mode": "2d", "shape": {"M": 2048, "N": 8192}},
    {"family": "rope_kv_cache", "path": "op_tests/triton_tests/fusions/test_fused_kv_cache.py", "test": "test_fused_qk_rope_reshape_and_cache", "sha": "72d3a61ad6cf449abd7c53227947313c703349a5f3809a421f312db66be4f91f", "mode": "fused_write", "shape": {"TOKENS": 128, "HEADS": 32, "D": 128, "BLOCK": 16}},
    {"family": "rope_kv_cache", "path": "op_tests/test_rope.py", "test": "test_rope_sbhd", "sha": "686e60a729594d8609ea4d394a94d701ad1286465fba06120ed8bd2275336464", "mode": "sbhd", "shape": {"S": 2048, "B": 2, "H": 32, "D": 128}},
    {"family": "blockscale_gemm", "path": "op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py", "test": "test_gemm", "sha": "c232187c0a5ba42f069cafeb49e561e296bdcb0671c254e156c661e46de38b67", "mode": "fp8_block128", "shape": {"M": 128, "N": 4096, "K": 4096}},
    {"family": "blockscale_gemm", "path": "op_tests/test_gemm_a8w8_blockscale.py", "test": "test_gemm", "sha": "2a9cd0fb5246a390385cdb889ca98f52d020d76ef90cc2781ce468c6739d863d", "mode": "preshuffle", "shape": {"M": 16, "N": 2112, "K": 7168}},
    {"family": "all_reduce", "path": "op_tests/multigpu_tests/test_quick_all_reduce.py", "test": "test_allreduce_quick", "sha": "c2af4f6e0a1cecbabcd1d1d992aa504b6f4f0315315f55eb03604f7e0a1726ae", "mode": "quick", "shape": {"TOKENS": 128, "HIDDEN": 4096}},
    {"family": "all_reduce", "path": "op_tests/multigpu_tests/test_quick_all_reduce_rmsnorm.py", "test": "test_qr_all_reduce_rmsnorm_matches_torch_reference", "sha": "3b1f1b756adb21bb85c95d5782103fcce198359fc18e27c20a12342e33911b72", "mode": "rmsnorm", "shape": {"TOKENS": 64, "HIDDEN": 8192}},
    {"family": "sampling", "path": "op_tests/test_sampling.py", "test": "test_top_p_sampling", "sha": "d442a72e207a0836b14f2d53051f1217aac28be76dded1924227ceab16b3de4b", "mode": "top_p", "shape": {"B": 19, "VOCAB": 32000}},
    {"family": "sampling", "path": "op_tests/test_sampling.py", "test": "test_top_k_top_p_joint_sampling_from_probs", "sha": "d442a72e207a0836b14f2d53051f1217aac28be76dded1924227ceab16b3de4b", "mode": "top_k_top_p", "shape": {"B": 99, "VOCAB": 128256}},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_locked_root(root: Path) -> None:
    if root.resolve() != AITER_ROOT:
        raise ValueError(f"AITER root must be locked to {AITER_ROOT}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != AITER_GIT_SHA:
        raise ValueError("locked AITER revision mismatch")


def _operator(family: str) -> str:
    return {
        "mha": "multi_head_attention",
        "mla": "multi_latent_attention",
        "paged_attention": "paged_attention",
        "fused_moe": "fused_moe",
        "gemm": "gemm",
        "rms_norm": "rms_norm",
        "rope_kv_cache": "rope_kv_cache",
        "blockscale_gemm": "blockscale_gemm",
        "all_reduce": "all_reduce",
        "sampling": "sampling",
    }[family]


def build_inventory(root: Path = AITER_ROOT) -> dict[str, Any]:
    """Build and validate the pinned inventory in deterministic order."""
    _verify_locked_root(root)
    candidates: list[dict[str, Any]] = []
    for source in SOURCES:
        path = root / source["path"]
        actual_hash = _sha256(path)
        if actual_hash != source["sha"]:
            raise ValueError(f"locked source hash mismatch: {source['path']}")
        family = source["family"]
        ranks = (2, 4) if family == "all_reduce" else (None,)
        for rank in ranks:
            suffix = f"-rank{rank}" if rank else ""
            source_language = "hip" if "/triton_tests/" not in source["path"] else "triton"
            contract = {
                "operator": _operator(family),
                "mode": source["mode"],
                "shape": source["shape"],
                "input_dtype": "bf16",
                "output_dtype": "bf16",
            }
            if family == "all_reduce":
                contract["world_size"] = rank
            if family == "sampling":
                contract["output_dtype"] = "int32"
            if family == "blockscale_gemm":
                contract["input_dtype"] = "fp8_e4m3fnuz"
                contract["weight_dtype"] = "fp8_e4m3fnuz"
                contract["scale"] = {"activation": "block128", "weight": "block128"}
            candidate_id = f"top10-{family}-{source['mode']}{suffix}".replace("_", "-")
            candidates.append(
                {
                    "id": candidate_id,
                    "priority": "P0",
                    "top10_family": family,
                    "contract_family_id": f"TOP10-{family.upper()}-{source['mode'].upper()}{suffix}".replace("_", "-"),
                    "source_lineage_id": f"aiter:{source['path']}:{source['test']}",
                    "source": {
                        "revision": "aiter_locked_gfx942",
                        "test_path": source["path"],
                        "test_id": source["test"],
                        "source_language": source_language,
                        "source_backend": "aiter",
                        "source_test_sha256": source["sha"],
                    },
                    "contract": contract,
                    "oracle": {
                        "tier": "independent_torch",
                        "method": "torch_reference",
                        "tolerances": {"rtol": 0.01, "atol": 0.01},
                    },
                    "target_lanes": (
                        ["hip_gfx942"] if family == "all_reduce" else list(DUAL_LANES)
                    ),
                }
            )
    family_weights = Counter(str(source["family"]) for source in SOURCES)
    return {
        "schema_version": "geak_top10_inventory_v1",
        "architecture": "gfx942",
        "families": list(TOP10_FAMILIES),
        "family_weights": {
            family: family_weights[family] for family in TOP10_FAMILIES
        },
        "source_revisions": {
            "aiter_locked_gfx942": {
                "repository": "https://github.com/ROCm/aiter.git",
                "branch": "main",
                "license": "MIT",
                "local_root": str(AITER_ROOT),
                "git_sha": AITER_GIT_SHA,
            }
        },
        "candidates": sorted(candidates, key=lambda item: item["id"]),
    }


def allocate_weighted_quotas(
    total: int, weights: dict[str, int]
) -> dict[str, int]:
    """Allocate an exact total with deterministic largest remainders."""
    if total < 0 or set(weights) != set(TOP10_FAMILIES):
        raise ValueError("weights must cover all Top10 families and total must be non-negative")
    if any(isinstance(value, bool) or value <= 0 for value in weights.values()):
        raise ValueError("family weights must be positive integers")
    denominator = sum(weights.values())
    quotas = {
        family: total * weights[family] // denominator for family in TOP10_FAMILIES
    }
    order = sorted(
        TOP10_FAMILIES,
        key=lambda family: (
            -(total * weights[family] % denominator),
            TOP10_FAMILIES.index(family),
        ),
    )
    for family in order[: total - sum(quotas.values())]:
        quotas[family] += 1
    return quotas


def write_inventory(output: Path, root: Path = AITER_ROOT) -> dict[str, Any]:
    inventory = build_inventory(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix in {".yaml", ".yml"}:
        output.write_text(yaml.safe_dump(inventory, sort_keys=False), encoding="utf-8")
    else:
        output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory


def merge_with_catalog(
    inventory: dict[str, Any], base_catalog: Path
) -> dict[str, Any]:
    """Append the pinned inventory without replacing existing candidates."""
    payload = yaml.safe_load(base_catalog.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base catalog must contain a mapping")
    merged = dict(payload)
    revisions = dict(payload.get("source_revisions") or {})
    for key, value in inventory["source_revisions"].items():
        if key in revisions and revisions[key] != value:
            raise ValueError(f"conflicting source revision {key}")
        revisions[key] = value
    candidates = {
        str(candidate["id"]): dict(candidate)
        for candidate in payload.get("candidates", [])
    }
    for candidate in inventory["candidates"]:
        candidate_id = str(candidate["id"])
        if candidate_id in candidates and candidates[candidate_id] != candidate:
            raise ValueError(f"conflicting candidate {candidate_id}")
        candidates[candidate_id] = dict(candidate)
    merged["source_revisions"] = revisions
    merged["candidates"] = [candidates[key] for key in sorted(candidates)]
    merged["top10_inventory"] = {
        "schema_version": inventory["schema_version"],
        "families": list(inventory["families"]),
        "family_weights": dict(inventory["family_weights"]),
        "candidate_count": len(inventory["candidates"]),
        "aiter_git_sha": AITER_GIT_SHA,
    }
    return merged


def write_catalog(
    output: Path, root: Path = AITER_ROOT, base_catalog: Path | None = None
) -> dict[str, Any]:
    inventory = build_inventory(root)
    payload = (
        merge_with_catalog(inventory, base_catalog)
        if base_catalog is not None
        else inventory
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix in {".yaml", ".yml"}:
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aiter-root", type=Path, default=AITER_ROOT)
    parser.add_argument(
        "--base-catalog",
        type=Path,
        help="append Top10 candidates to an existing Phase 1 candidate catalog",
    )
    args = parser.parse_args(argv)
    inventory = build_inventory(args.aiter_root)
    payload = write_catalog(args.output, args.aiter_root, args.base_catalog)
    print(
        json.dumps(
            {
                "candidates": len(payload["candidates"]),
                "top10_candidates": len(inventory["candidates"]),
                "families": len(inventory["families"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
