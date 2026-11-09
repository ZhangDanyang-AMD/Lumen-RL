"""Action-conditioned video world model on a Wan2.2-TI2V-5B DiT backbone.

The published checkpoint carries the TI2V-5B DiT (no text embedding) plus two
action encoders; the conditioning they implement was recovered empirically
from the weights and recorded trajectories:

* ``action_mlp1`` (7 -> D -> D) embeds each of the 8 target actions; the 8
  embeddings are the cross-attention context.
* ``action_mlp2`` (28 -> 4D -> D) embeds each group of 4 consecutive actions,
  i.e. the actions that produced one latent frame (VAE temporal factor 4). It
  is added to that latent frame's timestep embedding for all its tokens,
  including context frames. The first context latent encodes a single frame
  and gets no action embedding.
* Context latents are clean (timestep 0); generated latents share the
  sampling timestep. Sampling is rectified flow with a shifted sigma schedule.

Actions are raw LIBERO env actions (gripper -1 open / +1 close); the action
stored with frame ``i`` is the one that produced frame ``i``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

ACTION_DIM = 7
FRAMES_PER_LATENT = 4


@dataclass
class WanWorldModelConfig:
    dit_file: str = "model-00001.safetensors"
    vae_file: str = "Wan2.2_VAE.pth"
    sampling_steps: int = 5
    sigma_shift: float = 5.0
    chunk_size: int = 8


def _action_mlp(sd: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    w0, w2 = sd[f"{prefix}.0.weight"], sd[f"{prefix}.2.weight"]
    mlp = nn.Sequential(nn.Linear(w0.shape[1], w0.shape[0]), nn.SiLU(), nn.Linear(w2.shape[1], w2.shape[0]))
    mlp.load_state_dict({
        "0.weight": w0, "0.bias": sd[f"{prefix}.0.bias"],
        "2.weight": w2, "2.bias": sd[f"{prefix}.2.bias"],
    })
    return mlp


class WanActionWorldModel(nn.Module):
    """Frozen video world model: (context frames, actions) -> next frames."""

    # The first latent holds one frame, every later latent four.
    context_frame_step = FRAMES_PER_LATENT

    def __init__(
        self,
        ckpt_dir: str,
        cfg: Optional[WanWorldModelConfig] = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.wan_video_vae import WanVideoVAE38
        from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter
        from safetensors.torch import load_file

        self.cfg = cfg or WanWorldModelConfig()
        if self.cfg.chunk_size % FRAMES_PER_LATENT != 0:
            raise ValueError("chunk_size must be a multiple of the VAE temporal factor (4)")
        self.device = torch.device(device)
        self.dtype = dtype

        sd = load_file(os.path.join(ckpt_dir, self.cfg.dit_file))
        with torch.device(self.device):
            dit = WanModel(
                has_image_input=False, patch_size=[1, 2, 2], in_dim=48, dim=3072, ffn_dim=14336,
                freq_dim=256, text_dim=4096, out_dim=48, num_heads=24, num_layers=30, eps=1e-6,
                seperated_timestep=True, require_clip_embedding=False, require_vae_embedding=False,
                fuse_vae_embedding_in_latents=True,
            )
        dit_sd = {k: v for k, v in sd.items() if not k.startswith("action_mlp")}
        missing, unexpected = dit.load_state_dict(dit_sd, strict=False)
        missing = [k for k in missing if not k.startswith("text_embedding")]
        if missing or unexpected:
            raise RuntimeError(f"Wan DiT checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        dit.text_embedding = None
        self.dit = dit
        self.action_mlp1 = _action_mlp(sd, "action_mlp1")
        self.action_mlp2 = _action_mlp(sd, "action_mlp2")
        del sd, dit_sd

        vae = WanVideoVAE38()
        vae_sd = torch.load(os.path.join(ckpt_dir, self.cfg.vae_file), map_location="cpu", weights_only=True)
        vae.load_state_dict(WanVideoVAEStateDictConverter(vae_sd))
        self.vae = vae
        self.to(device=self.device, dtype=dtype)
        self.vae_scale = [vae.mean.to(self.device, dtype), (1.0 / vae.std).to(self.device, dtype)]
        self.freqs = tuple(f.to(self.device) for f in self.dit.freqs)
        self.requires_grad_(False)
        self.eval()

        sig = torch.linspace(1.0, 0.0, self.cfg.sampling_steps + 1)[:-1]
        sig = self.cfg.sigma_shift * sig / (1 + (self.cfg.sigma_shift - 1) * sig)
        self.sigmas = torch.cat([sig, torch.zeros(1)]).tolist()

    # ------------------------------------------------------------------ VAE

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, T, H, W, 3]`` (``T % 4 == 1``) -> latents ``[B, 48, 1 + (T-1)/4, H/16, W/16]``."""
        x = frames.to(self.device).permute(0, 4, 1, 2, 3).to(self.dtype).div_(127.5).sub_(1.0)
        return self.vae.model.encode(x, self.vae_scale)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> uint8 frames ``[B, T, H, W, 3]``."""
        video = self.vae.model.decode(latents.to(self.dtype), self.vae_scale).clamp_(-1, 1)
        video = ((video.float() + 1.0) * 127.5).round_().clamp_(0, 255).to(torch.uint8)
        return video.permute(0, 2, 3, 4, 1)

    # ------------------------------------------------------------------ DiT

    def _velocity(
        self,
        latents: torch.Tensor,
        n_ctx: int,
        sigma: float,
        ctx_actions: Optional[torch.Tensor],
        tgt_actions: torch.Tensor,
    ) -> torch.Tensor:
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        dit = self.dit
        bsz, _, n_lat, h, w = latents.shape
        x = dit.patch_embedding(latents)
        fh, ph, pw = x.shape[2:]
        tok = ph * pw
        x = rearrange(x, "b c f h w -> b (f h w) c")

        ts = torch.cat([
            torch.zeros(n_ctx * tok, device=self.device),
            torch.full(((fh - n_ctx) * tok,), sigma * 1000.0, device=self.device),
        ]).to(self.dtype)
        temb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, ts)).unsqueeze(0).expand(bsz, -1, -1)

        group = FRAMES_PER_LATENT * ACTION_DIM
        frame_emb = [torch.zeros(bsz, 1, dit.dim, device=self.device, dtype=self.dtype)]
        if n_ctx > 1:
            frame_emb.append(self.action_mlp2(ctx_actions.reshape(bsz, -1, group)))
        frame_emb.append(self.action_mlp2(tgt_actions.reshape(bsz, -1, group)))
        temb = temb + torch.cat(frame_emb, dim=1).repeat_interleave(tok, dim=1)
        context = self.action_mlp1(tgt_actions)

        t_mod = dit.time_projection(temb).unflatten(2, (6, dit.dim))
        freqs = torch.cat([
            self.freqs[0][:fh].view(fh, 1, 1, -1).expand(fh, ph, pw, -1),
            self.freqs[1][:ph].view(1, ph, 1, -1).expand(fh, ph, pw, -1),
            self.freqs[2][:pw].view(1, 1, pw, -1).expand(fh, ph, pw, -1),
        ], dim=-1).reshape(fh * ph * pw, 1, -1)
        for blk in dit.blocks:
            x = blk(x, context, t_mod, freqs)
        x = dit.head(x, temb)
        return dit.unpatchify(x, (fh, ph, pw))

    @torch.no_grad()
    def generate_latents(
        self,
        ctx_latents: torch.Tensor,
        ctx_actions: Optional[torch.Tensor],
        tgt_actions: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample the latents of the next ``chunk_size`` frames.

        ``ctx_actions`` ``[B, 4*(n_ctx-1), 7]`` are the actions that produced
        context frames 1..; ``tgt_actions`` ``[B, chunk, 7]`` the actions to apply.
        """
        bsz, ch, n_ctx, h, w = ctx_latents.shape
        n_new = self.cfg.chunk_size // FRAMES_PER_LATENT
        tgt = tgt_actions.to(self.device, self.dtype)
        ctx_act = ctx_actions.to(self.device, self.dtype) if (ctx_actions is not None and n_ctx > 1) else None
        x = torch.randn((bsz, ch, n_new, h, w), generator=generator, device=self.device, dtype=torch.float32)
        ctx = ctx_latents.to(self.dtype)
        for i in range(self.cfg.sampling_steps):
            lat = torch.cat([ctx, x.to(self.dtype)], dim=2)
            v = self._velocity(lat, n_ctx, self.sigmas[i], ctx_act, tgt)[:, :, n_ctx:].float()
            x = x + (self.sigmas[i + 1] - self.sigmas[i]) * v
        return x.to(self.dtype)

    @torch.no_grad()
    def predict_frames(
        self,
        ctx_frames: torch.Tensor,
        ctx_actions: Optional[torch.Tensor],
        tgt_actions: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """uint8 context ``[B, T, H, W, 3]`` + actions -> uint8 next frames ``[B, chunk, H, W, 3]``.

        Context and generated latents are decoded together because the VAE
        decoder is causal: a generated latent only decodes correctly after the
        context latents that precede it.
        """
        n_frames = ctx_frames.shape[1]
        lat = self.encode(ctx_frames)
        new = self.generate_latents(lat, ctx_actions, tgt_actions, generator)
        video = self.decode(torch.cat([lat, new], dim=2))
        return video[:, n_frames:]
