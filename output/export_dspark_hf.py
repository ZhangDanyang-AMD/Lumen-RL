"""Export LumenRL DSpark checkpoint to HuggingFace safetensors format.

Architecture aligned with Inferact/Kimi-K3-DSpark:
- K3DSparkModel with MLA attention (q_lora_rank=1536, kv_lora_rank=512)
- 5 dense layers, VanillaMarkov head (rank=256), confidence head
- embed_tokens + lm_head frozen from K3 base model (bfloat16)
- Trained weights in bfloat16
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def find_safetensor_shard(model_dir: str, key: str) -> str:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    for candidate in [key, f"model.{key}", f"language_model.model.{key}"]:
        if candidate in weight_map:
            return os.path.join(model_dir, weight_map[candidate]), candidate
    raise KeyError(f"{key} not found in {idx_path}")


def load_base_weight(model_dir: str, key: str) -> torch.Tensor:
    shard_path, actual_key = find_safetensor_shard(model_dir, key)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(actual_key)


def main():
    parser = argparse.ArgumentParser(description="Export DSpark checkpoint to HF format")
    parser.add_argument("--ckpt", default="/dev/shm/checkpoints/kimi_k3_dspark_vllm/latest.pt",
                        help="Path to LumenRL training checkpoint")
    parser.add_argument("--base-model", default="/dev/shm/Kimi-K3",
                        help="Path to Kimi-K3 base model (for frozen embed_tokens/lm_head)")
    parser.add_argument("--output", default="/home/danyzhan/Lumen-RL/output/Kimi_K3_DSpark_HF",
                        help="Output directory for HF model")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    msd = ckpt["state_dict"]["model_state_dict"]
    step = ckpt["state_dict"].get("step", ckpt.get("step", "unknown"))
    print(f"Checkpoint step: {step}")
    print(f"Keys in checkpoint: {len(msd)}")
    for k in sorted(msd.keys())[:20]:
        print(f"  {k}: {list(msd[k].shape)} {msd[k].dtype}")

    print(f"\nLoading frozen weights from base model: {args.base_model}")
    embed_tokens = load_base_weight(args.base_model, "embed_tokens.weight")
    lm_head = load_base_weight(args.base_model, "lm_head.weight")
    print(f"  embed_tokens: {list(embed_tokens.shape)} {embed_tokens.dtype}")
    print(f"  lm_head: {list(lm_head.shape)} {lm_head.dtype}")

    # --- Map LumenRL keys to HF keys ---
    hf_sd = {}

    # Frozen weights from K3 (keep bfloat16)
    hf_sd["embed_tokens.weight"] = embed_tokens.to(torch.bfloat16)
    hf_sd["lm_head.weight"] = lm_head.to(torch.bfloat16)

    # fc: hidden state fusion [7168, 5*7168] → [7168, 35840]
    hf_sd["fc.weight"] = msd["fc.weight"].to(torch.bfloat16)

    # hidden_norm (post-fusion RMSNorm)
    if "hidden_norm.weight" in msd:
        hf_sd["hidden_norm.weight"] = msd["hidden_norm.weight"].to(torch.bfloat16)

    # 5 transformer layers with MLA attention
    for i in range(5):
        prefix_src = f"layers.{i}."
        prefix_dst = f"layers.{i}."

        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            # MLA attention projections
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            # MLP
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            src_key = prefix_src + suffix
            if src_key in msd:
                hf_sd[prefix_dst + suffix] = msd[src_key].to(torch.bfloat16)

        # Also check for non-MLA attention (standard q/k/v projections)
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            src_key = prefix_src + suffix
            if src_key in msd:
                hf_sd[prefix_dst + suffix] = msd[src_key].to(torch.bfloat16)

    # Final norm
    for key in ["norm.weight", "out_norm.weight"]:
        if key in msd:
            hf_sd["norm.weight"] = msd[key].to(torch.bfloat16)
            break

    # Markov head
    if "markov_head.markov_w1.weight" in msd:
        hf_sd["markov_head.markov_w1.weight"] = msd["markov_head.markov_w1.weight"].to(torch.bfloat16)
        hf_sd["markov_head.markov_w2.weight"] = msd["markov_head.markov_w2.weight"].to(torch.bfloat16)
    elif "markov_head.embed.weight" in msd:
        hf_sd["markov_head.embed.weight"] = msd["markov_head.embed.weight"].to(torch.bfloat16)
        hf_sd["markov_head.proj.weight"] = msd["markov_head.proj.weight"].to(torch.bfloat16)

    # Confidence head
    if "confidence_head.proj.weight" in msd:
        hf_sd["confidence_head.proj.weight"] = msd["confidence_head.proj.weight"].to(torch.bfloat16)
        if "confidence_head.proj.bias" in msd:
            hf_sd["confidence_head.proj.bias"] = msd["confidence_head.proj.bias"].to(torch.bfloat16)

    print(f"\nHF state dict: {len(hf_sd)} tensors")
    for k in sorted(hf_sd.keys()):
        print(f"  {k}: {list(hf_sd[k].shape)} {hf_sd[k].dtype}")

    # --- Split into shards (embed+lm_head are large) ---
    large_keys = {"embed_tokens.weight", "lm_head.weight"}
    shard1_keys = [k for k in sorted(hf_sd.keys()) if k not in large_keys]
    shard2_keys = ["embed_tokens.weight"]
    shard3_keys = ["lm_head.weight"]

    shards = [
        ("model-00001-of-00003.safetensors", {k: hf_sd[k] for k in shard1_keys}),
        ("model-00002-of-00003.safetensors", {k: hf_sd[k] for k in shard2_keys}),
        ("model-00003-of-00003.safetensors", {k: hf_sd[k] for k in shard3_keys}),
    ]

    weight_map = {}
    for fname, shard in shards:
        print(f"Saving {fname} ({len(shard)} tensors)...")
        save_file(shard, os.path.join(args.output, fname))
        for k in shard:
            weight_map[k] = fname

    # --- Index ---
    total_params = sum(v.numel() for v in hf_sd.values())
    total_size = sum(v.numel() * v.element_size() for v in hf_sd.values())

    index = {
        "metadata": {"total_parameters": total_params, "total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size / 1e9:.2f} GB")

    # --- config.json (matching Inferact/Kimi-K3-DSpark) ---
    config = {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": False,
        "mla_use_output_gate": False,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 1048576,
        "rope_theta": 50000.0,
        "num_target_layers": 5,
        "target_hidden_size": 7168,
        "target_num_hidden_layers": 93,
        "target_layer_ids": [2, 23, 47, 71, 89],
        "mask_token_id": 163837,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "tie_word_embeddings": False,
        "draft_vocab_size": 163840,
        "_torchspec_version": "0.1.0",
        "torch_dtype": "bfloat16",
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- .gitattributes ---
    with open(os.path.join(args.output, ".gitattributes"), "w") as f:
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")

    print(f"\nExport complete: {args.output}")
    print(f"To upload: huggingface-cli upload <repo-id> {args.output}")


if __name__ == "__main__":
    main()
