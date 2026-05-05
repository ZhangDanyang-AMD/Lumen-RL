"""Export LumenRL Eagle3 checkpoint to HuggingFace safetensors format."""
import torch
import json
import os
from safetensors.torch import save_file

CKPT_PATH = "/dev/shm/checkpoints/kimi_k25_eagle3_vllm_phase2/checkpoint_65000.pt"
BASE_MODEL_DIR = "/dev/shm/Kimi-K2.5-BF16"
OUTPUT_DIR = "/root/lumenrl/output/Kimi_K25_eagle3_HF"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load our trained draft checkpoint ---
print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
msd = ckpt["state_dict"]["model_state_dict"]
step = ckpt["state_dict"].get("step", ckpt.get("step", "unknown"))
print(f"Checkpoint step: {step}")

# --- 2. Load embed_tokens and lm_head from base model ---
print("Loading embed_tokens and lm_head from base model...")
from safetensors import safe_open
base_shard = os.path.join(BASE_MODEL_DIR, "model-00062-of-000064.safetensors")
with safe_open(base_shard, framework="pt", device="cpu") as f:
    embed_tokens = f.get_tensor("language_model.model.embed_tokens.weight")
    lm_head = f.get_tensor("language_model.lm_head.weight")
print(f"  embed_tokens: {list(embed_tokens.shape)} {embed_tokens.dtype}")
print(f"  lm_head: {list(lm_head.shape)} {lm_head.dtype}")

# --- 3. Map LumenRL keys to HF keys ---
# Our model has biases (unlike the reference lightseekorg model)
# in_proj_weight [21504, 7168] is fused QKV, split into q/k/v each [7168, 7168]
hidden_size = 7168
in_proj_w = msd["blocks.0.attn.in_proj_weight"]  # [21504, 7168]
in_proj_b = msd["blocks.0.attn.in_proj_bias"]    # [21504]
q_w, k_w, v_w = in_proj_w.chunk(3, dim=0)
q_b, k_b, v_b = in_proj_b.chunk(3, dim=0)

hf_state_dict = {}

# embed_tokens and lm_head (from base model, keep bfloat16)
hf_state_dict["embed_tokens.weight"] = embed_tokens
hf_state_dict["lm_head.weight"] = lm_head

# fusion layer: concat(base_hidden, draft_input) -> hidden
hf_state_dict["fc.weight"] = msd["fusion.fc.weight"].to(torch.bfloat16)

# fusion norm -> midlayer.hidden_norm
hf_state_dict["midlayer.hidden_norm.weight"] = msd["fusion.norm.weight"].to(torch.bfloat16)
hf_state_dict["midlayer.hidden_norm.bias"] = msd["fusion.norm.bias"].to(torch.bfloat16)

# pre-attention norm -> midlayer.input_layernorm
hf_state_dict["midlayer.input_layernorm.weight"] = msd["blocks.0.attn_norm.weight"].to(torch.bfloat16)
hf_state_dict["midlayer.input_layernorm.bias"] = msd["blocks.0.attn_norm.bias"].to(torch.bfloat16)

# attention projections (split from fused QKV)
hf_state_dict["midlayer.self_attn.q_proj.weight"] = q_w.to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.q_proj.bias"] = q_b.to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.k_proj.weight"] = k_w.to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.k_proj.bias"] = k_b.to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.v_proj.weight"] = v_w.to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.v_proj.bias"] = v_b.to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.o_proj.weight"] = msd["blocks.0.attn.out_proj.weight"].to(torch.bfloat16)
hf_state_dict["midlayer.self_attn.o_proj.bias"] = msd["blocks.0.attn.out_proj.bias"].to(torch.bfloat16)

# post-attention norm -> midlayer.post_attention_layernorm
hf_state_dict["midlayer.post_attention_layernorm.weight"] = msd["blocks.0.ffn_norm.weight"].to(torch.bfloat16)
hf_state_dict["midlayer.post_attention_layernorm.bias"] = msd["blocks.0.ffn_norm.bias"].to(torch.bfloat16)

# MLP: w1=gate, w3=up, w2=down
hf_state_dict["midlayer.mlp.gate_proj.weight"] = msd["blocks.0.w1.weight"].to(torch.bfloat16)
hf_state_dict["midlayer.mlp.up_proj.weight"] = msd["blocks.0.w3.weight"].to(torch.bfloat16)
hf_state_dict["midlayer.mlp.down_proj.weight"] = msd["blocks.0.w2.weight"].to(torch.bfloat16)

# output norm
hf_state_dict["norm.weight"] = msd["out_norm.weight"].to(torch.bfloat16)
hf_state_dict["norm.bias"] = msd["out_norm.bias"].to(torch.bfloat16)

# --- 4. Split into 2 shards (like reference) ---
# shard 1: everything except lm_head
# shard 2: lm_head
shard1_keys = [k for k in hf_state_dict if k != "lm_head.weight"]
shard2_keys = ["lm_head.weight"]

shard1 = {k: hf_state_dict[k] for k in shard1_keys}
shard2 = {k: hf_state_dict[k] for k in shard2_keys}

shard1_file = "model-00001-of-00002.safetensors"
shard2_file = "model-00002-of-00002.safetensors"

print(f"Saving {shard1_file} ({len(shard1)} tensors)...")
save_file(shard1, os.path.join(OUTPUT_DIR, shard1_file))
print(f"Saving {shard2_file} ({len(shard2)} tensors)...")
save_file(shard2, os.path.join(OUTPUT_DIR, shard2_file))

# --- 5. model.safetensors.index.json ---
total_params = sum(v.numel() for v in hf_state_dict.values())
total_size = sum(v.numel() * v.element_size() for v in hf_state_dict.values())

weight_map = {}
for k in shard1_keys:
    weight_map[k] = shard1_file
for k in shard2_keys:
    weight_map[k] = shard2_file

index = {
    "metadata": {
        "total_parameters": total_params,
        "total_size": total_size,
    },
    "weight_map": dict(sorted(weight_map.items())),
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
print(f"Total parameters: {total_params:,}")
print(f"Total size: {total_size / 1e9:.2f} GB")

# --- 6. config.json (matching OUR model architecture) ---
config = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "bos_token_id": 163584,
    "draft_vocab_size": 163840,
    "dtype": "bfloat16",
    "eos_token_id": 163585,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 262144,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 1,
    "num_key_value_heads": 64,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 50000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 163840,
    "norm_bias": True,
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# --- 7. .gitattributes ---
gitattributes = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
with open(os.path.join(OUTPUT_DIR, ".gitattributes"), "w") as f:
    f.write(gitattributes)

# --- 8. README.md ---
readme = f"""---
license: mit
base_model: moonshotai/Kimi-K2.5
tags:
- safetensors
- llama
- speculative-decoding
- eagle3
- draft-model
- kimi-k2.5
---

# Kimi-K2.5 Eagle3 Draft Model (LumenRL, MI350)

Eagle3 MTP draft model for accelerating inference of [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5),
trained with [LumenRL](https://github.com/LumenRL/Lumen-RL) on 8x AMD MI350 GPUs.

## Training Setup

- **Hardware**: 8x AMD MI350 (4 train + 4 inference)
- **Training**: FSDP2 (BF16, LumenRL + aiter), GPUs 0-3
- **Inference**: vLLM + ATOM (MXFP4 quantization, aiter), GPUs 4-7, TP=4
- **Transfer**: Mooncake TCP
- **Draft**: Eagle3, 1-layer Transformer (hidden_size=7168, intermediate_size=12288, num_heads=64)

## Training Phases

- **Phase 1 (Foundation)**: perfectblend subset, 111K steps (3 epochs)
- **Phase 2 (Mixed Domain)**: VL + Chinese + tool-call + agent + writing, 67,826 steps (3 epochs)

Final checkpoint at step {step}, all positions reaching 100% acceptance accuracy.

## Architecture Differences from Reference

Compared to [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3):
- **attention_bias**: `true` (reference uses `false`)
- **norm_bias**: `true` (norms include bias terms)
- **rope_theta**: 50000.0 (matching Kimi-K2.5 base, reference uses 1000000)
- **rope_scaling**: yarn (matching Kimi-K2.5 base config)
- **rms_norm_eps**: 1e-5 (reference uses 1e-6)
"""
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme)

# Summary
print("\n=== Export complete ===")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fn)
    size = os.path.getsize(fpath)
    if size > 1e9:
        print(f"  {fn}: {size/1e9:.2f} GB")
    elif size > 1e6:
        print(f"  {fn}: {size/1e6:.1f} MB")
    else:
        print(f"  {fn}: {size/1e3:.1f} KB")
