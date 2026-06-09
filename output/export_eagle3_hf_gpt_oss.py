"""Export LumenRL Eagle3 checkpoint for gpt-oss-120b to HuggingFace safetensors.

Parallel to output/export_eagle3_hf.py (Kimi). Adapted for the gpt-oss-120b
draft trained by examples/GPT_OSS_120b_MI308_ATOM:

- 1-layer Llama-style decoder, GQA 64:8, head_dim=64, ffn=17280
- llama3 RoPE (theta=500000, factor=8, ctx 8192 -> 65536)
- no bias anywhere, RMSNorm eps 1e-5
- Eagle3 aux hidden layer ids: [1, 17, 32]
- vocab = gpt-oss vocab (201088, o200k_harmony)

Trained checkpoint has 12 tensors (all bfloat16); lm_head + embed_tokens are
NOT in it — both are copied from the base model. The base ships natively in
MXFP4 but lm_head/embed_tokens are in `modules_to_not_convert`, so they live
in the safetensors as plain bfloat16 and can be lifted as-is.

Usage:
    python3 output/export_eagle3_hf_gpt_oss.py \
        [--ckpt /dev/shm/checkpoints/gpt_oss_120b_eagle3_vllm/checkpoint_21000.pt] \
        [--base /dev/shm/gpt-oss-120b] \
        [--out  /dev/shm/gpt_oss_120b_eagle3_HF]
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _glob_latest_ckpt(d: str) -> str:
    import glob
    import re
    files = glob.glob(os.path.join(d, "checkpoint_*.pt"))
    if not files:
        raise FileNotFoundError(f"no checkpoint_*.pt in {d}")
    step = lambda p: int(re.search(r"checkpoint_(\d+)\.pt$", p).group(1))
    return max(files, key=step)


def _resolve_msd(ckpt: dict) -> tuple[dict, int | str]:
    sd = ckpt.get("state_dict", ckpt)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        return sd["model_state_dict"], sd.get("step", ckpt.get("step", "?"))
    return sd, ckpt.get("step", "?")


def _load_base_tensor(base_dir: str, key: str) -> torch.Tensor:
    """Read one tensor from the base model's sharded safetensors by HF key."""
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Specific .pt; defaults to latest checkpoint_*.pt under --ckpt-dir.")
    ap.add_argument("--ckpt-dir", default="/dev/shm/checkpoints/gpt_oss_120b_eagle3_vllm")
    ap.add_argument("--base", default="/dev/shm/gpt-oss-120b")
    ap.add_argument("--out", default="/dev/shm/gpt_oss_120b_eagle3_HF")
    ap.add_argument("--draft-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="dtype to save the trained draft weights in.")
    args = ap.parse_args()

    ckpt_path = args.ckpt or _glob_latest_ckpt(args.ckpt_dir)
    print(f"[load] ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msd, step = _resolve_msd(ckpt)
    print(f"[load] step={step}, {len(msd)} draft tensors")

    draft_dt = getattr(torch, args.draft_dtype)

    print(f"[load] embed_tokens + lm_head from {args.base}")
    embed_tokens = _load_base_tensor(args.base, "model.embed_tokens.weight")
    lm_head = _load_base_tensor(args.base, "lm_head.weight")
    print(f"       embed_tokens: {list(embed_tokens.shape)} {embed_tokens.dtype}")
    print(f"       lm_head:      {list(lm_head.shape)} {lm_head.dtype}")

    os.makedirs(args.out, exist_ok=True)

    # ---- Map LumenRL keys -> HF Eagle3 keys ----
    # HF Eagle3 layout (vLLM-compatible LlamaForCausalLMEagle3): one decoder
    # block exposed as `midlayer.*`, plus embed_tokens / fc / norm / lm_head.
    def take(k: str, dt: torch.dtype = draft_dt) -> torch.Tensor:
        if k not in msd:
            raise KeyError(f"missing draft key: {k}")
        return msd[k].to(dt).contiguous()

    hf: dict[str, torch.Tensor] = {
        # frozen base weights (keep their native dtype)
        "embed_tokens.weight": embed_tokens.contiguous(),
        "lm_head.weight":      lm_head.contiguous(),
        # trained draft weights
        "fc.weight":                                take("fc.weight"),
        "midlayer.hidden_norm.weight":              take("layers.0.hidden_norm.weight"),
        "midlayer.input_layernorm.weight":          take("layers.0.input_layernorm.weight"),
        "midlayer.self_attn.q_proj.weight":         take("layers.0.self_attn.q_proj.weight"),
        "midlayer.self_attn.k_proj.weight":         take("layers.0.self_attn.k_proj.weight"),
        "midlayer.self_attn.v_proj.weight":         take("layers.0.self_attn.v_proj.weight"),
        "midlayer.self_attn.o_proj.weight":         take("layers.0.self_attn.o_proj.weight"),
        "midlayer.post_attention_layernorm.weight": take("layers.0.post_attention_layernorm.weight"),
        "midlayer.mlp.gate_proj.weight":            take("layers.0.mlp.gate_proj.weight"),
        "midlayer.mlp.up_proj.weight":              take("layers.0.mlp.up_proj.weight"),
        "midlayer.mlp.down_proj.weight":            take("layers.0.mlp.down_proj.weight"),
        "norm.weight":                              take("out_norm.weight"),
    }

    # ---- Shard: lm_head alone in shard 2 (big — same convention as Kimi) ----
    shard1 = {k: v for k, v in hf.items() if k != "lm_head.weight"}
    shard2 = {"lm_head.weight": hf["lm_head.weight"]}
    shard1_file = "model-00001-of-00002.safetensors"
    shard2_file = "model-00002-of-00002.safetensors"

    print(f"[save] {shard1_file} ({len(shard1)} tensors)")
    save_file(shard1, os.path.join(args.out, shard1_file))
    print(f"[save] {shard2_file} ({len(shard2)} tensors)")
    save_file(shard2, os.path.join(args.out, shard2_file))

    # ---- Index ----
    total_params = sum(v.numel() for v in hf.values())
    total_size = sum(v.numel() * v.element_size() for v in hf.values())
    weight_map = {k: shard1_file for k in shard1}
    weight_map.update({k: shard2_file for k in shard2})
    index = {
        "metadata": {"total_parameters": total_params, "total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[save] index: {total_params:,} params, {total_size/1e9:.2f} GB")

    # ---- config.json ----
    # Matches the draft block in examples/GPT_OSS_120b_MI308_ATOM/configs/opd_gpt_oss_120b.yaml
    # and the o200k_harmony vocab from the gpt-oss-120b base.
    base_cfg_path = os.path.join(args.base, "config.json")
    with open(base_cfg_path) as f:
        base_cfg = json.load(f)

    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": base_cfg.get("bos_token_id", 199998),
        "eos_token_id": base_cfg.get("eos_token_id", 200002),
        "pad_token_id": base_cfg.get("pad_token_id", 199999),
        "hidden_act": "silu",
        "hidden_size": 2880,
        "intermediate_size": 17280,
        "head_dim": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "initializer_range": 0.02,
        "max_position_embeddings": 65536,   # 8192 * factor=8 (llama3 scaling)
        "mlp_bias": False,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        "sliding_window": None,
        "use_sliding_window": False,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": base_cfg["vocab_size"],            # 201088
        "draft_vocab_size": base_cfg["vocab_size"],      # no draft-vocab compression
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 17, 32],
            "use_aux_hidden_state": True,
            "use_input_layernorm_in_first_layer": True,
            "use_last_layernorm": True,
            "use_mtp_layernorm": False,
        },
        "dtype": args.draft_dtype,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Tokenizer passthrough ----
    # Copy the tokenizer files from the base so HF AutoTokenizer / vLLM can
    # load the draft dir directly. Symlink instead of copying to save IO.
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "chat_template.jinja", "generation_config.json"):
        src = os.path.join(args.base, fn)
        dst = os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                import shutil
                shutil.copy(src, dst)

    # ---- Summary ----
    print("\n=== Export complete ===")
    for fn in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, fn)
        if os.path.islink(p):
            print(f"  {fn} -> {os.readlink(p)}")
            continue
        sz = os.path.getsize(p)
        unit = "GB" if sz > 1e9 else "MB" if sz > 1e6 else "KB"
        div = 1e9 if sz > 1e9 else 1e6 if sz > 1e6 else 1e3
        print(f"  {fn}: {sz/div:.2f} {unit}")


if __name__ == "__main__":
    main()
