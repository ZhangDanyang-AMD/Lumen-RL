"""Run the real SimWAM modules on Lumen kernels.

Video expert is Wan2.2-TI2V-5B (30 layers, hidden 3072). Action expert is the
1024-d ActionDiT with LoRA on q/k/v/o. VAE is WanVideoVAE38. Shapes match
configs/model/simwam_navsim.yaml; the token grid is tiny so this checks
kernels, not throughput.

Usage (inside simwam-rl):
  PYTHONPATH=$LUMEN_DIR:$LUMEN_RL:$LUMEN_RL/examples/SimWAM/src \
    python3 examples/Embodied_SimWAM_NAVSIM/bench_kernels.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn as nn

from simwam.models.wan22.action_dit import ActionDiT
from simwam.models.wan22.lora import apply_lora_to_module
from simwam.models.wan22.mot import MoT
from simwam.models.wan22.wan_video_dit import WanVideoDiT
from simwam.models.wan22.wan_video_vae import WanVideoVAE38

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from examples.Embodied_SimWAM_NAVSIM.plugin import enable_lumen


def _video():
    return WanVideoDiT(
        hidden_dim=3072,
        in_dim=48,
        ffn_dim=14336,
        out_dim=48,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
    )


def _action():
    return ActionDiT(
        hidden_dim=1024,
        action_dim=3,
        ffn_dim=4096,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        use_gradient_checkpointing=False,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    spec = os.environ.get("LUMEN_ENABLE", "hf_attn_patch,lumen_linear,aiter_conv")
    print("build video expert", flush=True)
    video = _video().to(device=device, dtype=dtype).eval()
    print("build action expert", flush=True)
    action = _action().to(device=device, dtype=dtype).train()
    print("wrap action lora", flush=True)
    n_lora = apply_lora_to_module(action, target_names=("q", "k", "v", "o"), r=16, alpha=32.0)
    print("lora_wrapped", n_lora, flush=True)

    B, T, H, W = 1, 2, 4, 4
    latents = torch.randn(B, 48, T, H, W, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    context = torch.randn(B, 4, 4096, device=device, dtype=dtype)
    context_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
    with torch.no_grad():
        pre = video.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
    tokens = pre["tokens"]
    block = video.blocks[0]
    with torch.no_grad():
        torch_out = block(
            tokens, pre["context"], pre["t_mod"], pre["freqs"],
            context_mask=pre["context_mask"],
            self_attn_mask=video.build_video_to_video_mask(
                tokens.shape[1], int(pre["meta"]["tokens_per_frame"]), tokens.device,
            ),
        )

    holder = nn.Module()
    holder.video_expert = video
    holder.action_expert = action
    holder.vae = WanVideoVAE38().to(device=device, dtype=dtype).eval()
    print("enable", spec, flush=True)
    enable_lumen(holder, spec)

    vmask = video.build_video_to_video_mask(
        tokens.shape[1], int(pre["meta"]["tokens_per_frame"]), tokens.device,
    )
    with torch.no_grad():
        lumen_out = block(
            tokens, pre["context"], pre["t_mod"], pre["freqs"],
            context_mask=pre["context_mask"], self_attn_mask=vmask,
        )
    block_err = (lumen_out.float() - torch_out.float()).abs().max().item()
    print("video_block0_max_abs", block_err, "finite", torch.isfinite(lumen_out).all().item(), flush=True)
    if block_err > 0.05 or not torch.isfinite(lumen_out).all():
        raise SystemExit(f"video block mismatch {block_err}")

    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    mot.mot_attention_mask_mode = "isolated"
    with torch.no_grad():
        cache = mot.prefill_video_cache(
            video_tokens=tokens,
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_attention_mask=vmask,
        )
    print("video_prefill_layers", len(cache), "k", tuple(cache[0]["k"].shape), flush=True)

    action_tokens = torch.randn(B, 8, 3, device=device, dtype=dtype)
    action_pre = action.pre_dit(
        action_tokens=action_tokens,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
    )
    video_len = tokens.shape[1]
    action_len = action_pre["tokens"].shape[1]
    total = video_len + action_len
    joint = torch.zeros(total, total, dtype=torch.bool, device=device)
    joint[:video_len, :video_len] = vmask
    joint[video_len:, video_len:] = True
    joint[video_len:, : int(pre["meta"]["tokens_per_frame"])] = True
    updated = mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
        video_kv_cache=cache,
        attention_mask=joint,
        video_seq_len=video_len,
    )
    pred = action.post_dit(updated, action_pre)
    loss = pred.float().square().mean()
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in action.parameters() if p.requires_grad)
    print("action_pred", tuple(pred.shape), "loss", float(loss.detach()), "lora_grad", grad_ok, flush=True)
    if not grad_ok or not torch.isfinite(pred).all():
        raise SystemExit("action expert backward failed")

    video_in = torch.randn(3, 1, 16, 16, device=device, dtype=dtype)
    with torch.no_grad():
        latent = holder.vae.encode([video_in], device)
    print("vae_latent", tuple(latent.shape), "finite", torch.isfinite(latent).all().item(), flush=True)
    if not torch.isfinite(latent).all():
        raise SystemExit("vae encode produced non-finite values")
    print("BENCH OK", flush=True)


if __name__ == "__main__":
    main()
