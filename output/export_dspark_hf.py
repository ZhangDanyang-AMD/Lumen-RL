"""Export LumenRL DSpark checkpoint to HuggingFace safetensors format.

Architecture aligned with Inferact/Kimi-K3-DSpark:
- K3DSparkModel with MLA attention (q_lora_rank=1536, kv_lora_rank=512)
- 5 dense layers, VanillaMarkov head (rank=256), confidence head
- embed_tokens + lm_head frozen from K3 base model (bfloat16)
- Trained weights in bfloat16

The emitted config.json is what ATOM reads back to decide the draft's rotation
convention, softmax scale and which target layers to feed it. Every field below
that also exists in the training YAML is therefore cross-checked against it
(`--train-config`), because a silent disagreement here is invisible until the
draft benchmarks badly: the first run shipped at 6% acceptance against a
reference draft's 26% for exactly that class of mismatch.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _draft_cfg_from_yaml(path: str) -> dict:
    """Read the training YAML's draft/spec_distill blocks, flattened."""
    import yaml

    with open(path) as f:
        y = yaml.safe_load(f)
    algo = (y.get("algorithm") or {})
    return {
        "draft": (algo.get("draft") or {}),
        "spec_distill": (algo.get("spec_distill") or {}),
    }


def _check_against_training_config(config: dict, train_cfg: dict) -> list[str]:
    """Return a list of fields the exported config and the YAML disagree on."""
    draft = train_cfg["draft"]
    spec = train_cfg["spec_distill"]
    rope = config["rope_parameters"]

    # (label, exported value, YAML value) for every field that exists on both
    # sides. Anything the YAML does not carry is left to the export's own
    # defaults, which is why the architecture dims are absent here.
    pairs = [
        ("num_hidden_layers", config["num_hidden_layers"], draft.get("num_layers")),
        ("num_attention_heads", config["num_attention_heads"], draft.get("num_heads")),
        ("num_key_value_heads", config["num_key_value_heads"], draft.get("num_kv_heads")),
        ("intermediate_size", config["intermediate_size"], draft.get("ffn_dim")),
        ("q_lora_rank", config["q_lora_rank"], draft.get("q_lora_rank")),
        ("kv_lora_rank", config["kv_lora_rank"], draft.get("kv_lora_rank")),
        ("qk_nope_head_dim", config["qk_nope_head_dim"], draft.get("qk_nope_head_dim")),
        ("qk_rope_head_dim", config["qk_rope_head_dim"], draft.get("qk_rope_head_dim")),
        ("v_head_dim", config["v_head_dim"], draft.get("v_head_dim")),
        ("rms_norm_eps", config["rms_norm_eps"], draft.get("rms_norm_eps")),
        ("markov_rank", config["markov_rank"], draft.get("markov_rank")),
        ("markov_head_type", config["markov_head_type"], draft.get("markov_head_type")),
        ("mask_token_id", config["mask_token_id"], draft.get("mask_token_id")),
        ("enable_confidence_head", config["enable_confidence_head"],
         draft.get("enable_confidence_head")),
        ("confidence_head_with_markov", config["confidence_head_with_markov"],
         draft.get("confidence_head_with_markov")),
        ("rope_theta", config["rope_theta"], draft.get("rope_theta")),
        ("rope_parameters.rope_theta", rope["rope_theta"], draft.get("rope_theta")),
        ("rope_parameters.rope_type", rope["rope_type"], draft.get("rope_scaling_type")),
        ("rope_parameters.factor", rope["factor"], draft.get("rope_scaling_factor")),
        ("rope_parameters.original_max_position_embeddings",
         rope["original_max_position_embeddings"], draft.get("rope_original_max_pos")),
        ("rope_parameters.beta_fast", rope["beta_fast"], draft.get("rope_beta_fast")),
        ("rope_parameters.beta_slow", rope["beta_slow"], draft.get("rope_beta_slow")),
        ("rope_parameters.mscale", rope["mscale"], draft.get("rope_mscale")),
        ("rope_parameters.mscale_all_dim", rope["mscale_all_dim"],
         draft.get("rope_mscale_all_dim")),
        ("num_target_layers", config["num_target_layers"], spec.get("num_target_layers")),
        ("target_layer_ids", config["target_layer_ids"],
         spec.get("aux_hidden_state_layer_ids")),
    ]

    problems = []
    for label, exported, trained in pairs:
        if trained is None:
            continue
        if isinstance(exported, float) or isinstance(trained, float):
            same = abs(float(exported) - float(trained)) < 1e-12
        else:
            same = list(exported) == list(trained) if isinstance(exported, list) else exported == trained
        if not same:
            problems.append(f"  {label}: exporting {exported!r} but trained with {trained!r}")

    # rope_interleave has no YAML counterpart because the trainer has exactly one
    # rotation convention. Emitting the key at all would tell ATOM to use the
    # other one.
    if "rope_interleave" in config:
        problems.append(
            "  rope_interleave: must not be emitted -- ATOM maps it to "
            "is_neox_style, i.e. the half-split rotation the trainer does not use"
        )
    return problems


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
    parser.add_argument("--train-config",
                        default="examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml",
                        help="Training YAML to cross-check the emitted config.json against")
    parser.add_argument("--skip-config-check", action="store_true",
                        help="Export even if config.json disagrees with the training YAML")
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

    if not args.skip_config_check:
        if not os.path.exists(args.train_config):
            raise SystemExit(
                f"--train-config {args.train_config} not found. Point it at the YAML "
                f"this checkpoint was trained with, or pass --skip-config-check."
            )
        problems = _check_against_training_config(
            config, _draft_cfg_from_yaml(args.train_config),
        )
        if problems:
            raise SystemExit(
                "config.json disagrees with " + args.train_config + ":\n"
                + "\n".join(problems)
                + "\n\nATOM reads these back at serving time, so a mismatch here is a "
                  "draft served under different attention than it was trained with."
            )
        print(f"config.json cross-checked against {args.train_config}: OK")

    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- .gitattributes ---
    with open(os.path.join(args.output, ".gitattributes"), "w") as f:
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")

    print(f"\nExport complete: {args.output}")
    print(f"To upload: huggingface-cli upload <repo-id> {args.output}")


if __name__ == "__main__":
    main()
