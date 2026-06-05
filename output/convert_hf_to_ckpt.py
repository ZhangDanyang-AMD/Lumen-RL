"""Convert HF Eagle3 safetensors back to LumenRL checkpoint format.

Reverses the mapping in export_eagle3_hf_phase1.py:
  HF midlayer.* → LumenRL layers.0.*
  HF norm.weight → LumenRL out_norm.weight
  HF fc.weight → LumenRL fc.weight

Skips embed_tokens and lm_head (frozen, loaded from teacher at runtime).
"""
import argparse
import os
import torch
from safetensors import safe_open


HF_TO_LUMENRL = {
    "fc.weight": "fc.weight",
    "midlayer.hidden_norm.weight": "layers.0.hidden_norm.weight",
    "midlayer.input_layernorm.weight": "layers.0.input_layernorm.weight",
    "midlayer.self_attn.q_proj.weight": "layers.0.self_attn.q_proj.weight",
    "midlayer.self_attn.k_proj.weight": "layers.0.self_attn.k_proj.weight",
    "midlayer.self_attn.v_proj.weight": "layers.0.self_attn.v_proj.weight",
    "midlayer.self_attn.o_proj.weight": "layers.0.self_attn.o_proj.weight",
    "midlayer.post_attention_layernorm.weight": "layers.0.post_attention_layernorm.weight",
    "midlayer.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
    "midlayer.mlp.up_proj.weight": "layers.0.mlp.up_proj.weight",
    "midlayer.mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
    "norm.weight": "out_norm.weight",
}

SKIP_KEYS = {"embed_tokens.weight", "lm_head.weight"}


def convert(hf_dir: str, output_path: str, dtype: str = "bfloat16"):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    target_dtype = dtype_map.get(dtype, torch.bfloat16)

    state_dict = {}
    for fname in sorted(os.listdir(hf_dir)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(hf_dir, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in SKIP_KEYS:
                    continue
                if key in HF_TO_LUMENRL:
                    lumenrl_key = HF_TO_LUMENRL[key]
                    state_dict[lumenrl_key] = f.get_tensor(key).to(target_dtype)
                else:
                    print(f"  WARNING: unknown key {key}, skipping")

    payload = {"model_state_dict": state_dict}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved {len(state_dict)} tensors to {output_path}")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {list(v.shape)} {v.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    convert(args.hf_dir, args.output, args.dtype)
