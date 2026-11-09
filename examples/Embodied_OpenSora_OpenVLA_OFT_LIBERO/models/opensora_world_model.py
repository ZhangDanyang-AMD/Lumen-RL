"""Action-conditioned OpenSora (STDiT3) world model over a per-frame SDXL VAE.

* Every pixel frame is its own latent frame (2D VAE, 8x spatial, 4 channels).
  A sample is ``context_frames`` (4) conditioning latents followed by
  ``chunk_size`` (8) latents to generate.
* Spatial and temporal transformer blocks alternate. Temporal self-attention
  is causal with interleaved rotary embeddings; spatial tokens get a fixed 2D
  sin/cos embedding (width coordinate in the first half). The blocks have no
  cross-attention (its weights in the checkpoint are unused).
* Frame-level action control: in the spatial blocks every generated frame is
  modulated by ``timestep embedding + embedding of the action that produced
  it``; the temporal blocks and the final layer only see the timestep.
  Context frames are clean and modulated by the embedding of timestep 15.
* Actions enter as OpenVLA-style bin indices: q01/q99-normalized to [-1, 1],
  then digitized against 256 uniform bin edges (values 1..256).
* Rectified flow over timesteps ``1000 * (1 - i / steps)``: the network
  predicts ``x0 - noise`` for ``x_t = (1 - t/1000) x0 + (t/1000) noise``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

NUM_ACTION_BINS = 256
CONTEXT_TIMESTEP = 15.0


def _timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _ScalarEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_timestep_embedding(t).to(self.mlp[0].weight.dtype))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.unflatten(-1, (-1, 2))
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return x * cos + rotated * sin


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = _RMSNorm(dim // heads)
        self.k_norm = _RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope=None, causal: bool = False) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = _rotary(q, *rope), _rotary(k, *rope)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class _CrossAttention(nn.Module):
    """Present in the checkpoint, not used by the forward pass."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)


class _Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _Block(nn.Module):
    """adaLN-modulated self-attention (spatial, or causal temporal) + MLP on ``[B, T, S, C]``."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, temporal: bool) -> None:
        super().__init__()
        self.temporal = temporal
        self.attn = _SelfAttention(dim, heads)
        self.cross_attn = _CrossAttention(dim)
        self.mlp = _Mlp(dim, mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))

    def forward(self, x: torch.Tensor, mod: torch.Tensor, rope) -> torch.Tensor:
        """``mod`` ``[B, T, 6, C]`` per-frame modulation."""
        b, t, s, c = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (self.scale_shift_table + mod).unsqueeze(2).unbind(3)
        h = F.layer_norm(x, (c,), eps=1e-6) * (1 + scale_a) + shift_a
        if self.temporal:
            h = rearrange(h, "b t s c -> (b s) t c")
            h = rearrange(self.attn(h, rope, causal=True), "(b s) t c -> b t s c", b=b, s=s)
        else:
            h = rearrange(h, "b t s c -> (b t) s c")
            h = rearrange(self.attn(h), "(b t) s c -> b t s c", b=b, t=t)
        x = x + gate_a * h
        h = F.layer_norm(x, (c,), eps=1e-6) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


def _pos_embed_2d(dim: int, n_h: int, n_w: int, scale: float, base: int, device) -> torch.Tensor:
    half = dim // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, 2, device=device).float() / half))
    gh = torch.arange(n_h, device=device).float() / scale * (base / n_h)
    gw = torch.arange(n_w, device=device).float() / scale * (base / n_w)
    gh, gw = torch.meshgrid(gh, gw, indexing="ij")

    def sincos(v: torch.Tensor) -> torch.Tensor:
        out = v.reshape(-1)[:, None] * inv[None]
        return torch.cat([out.sin(), out.cos()], dim=-1)

    return torch.cat([sincos(gw), sincos(gh)], dim=-1)[None]


class STDiT3(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d, heads, depth = cfg["hidden_size"], cfg["num_heads"], cfg["depth"]
        self.cfg = cfg
        self.dim = d
        self.patch = tuple(cfg["patch_size"])
        self.in_channels = cfg["in_channels"]
        self.out_channels = self.in_channels * (2 if cfg.get("pred_sigma", True) else 1)
        self.x_embedder = nn.Module()
        self.x_embedder.proj = nn.Conv3d(self.in_channels, d, kernel_size=self.patch, stride=self.patch)
        self.t_embedder = _ScalarEmbedder(d)
        self.fps_embedder = _ScalarEmbedder(d)  # unused
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.action_embedder = nn.Module()
        self.action_embedder.fc = nn.Sequential(
            nn.Linear(cfg["action_dim"], 4 * d), nn.SiLU(), nn.Linear(4 * d, d)
        )
        self.action_embedder.none_action = nn.Embedding(1, d)  # pads context frames, never reaches the output
        ratio = cfg.get("mlp_ratio", 4.0)
        self.spatial_blocks = nn.ModuleList([_Block(d, heads, ratio, False) for _ in range(depth)])
        self.temporal_blocks = nn.ModuleList([_Block(d, heads, ratio, True) for _ in range(depth)])
        self.final_layer = nn.Module()
        self.final_layer.linear = nn.Linear(d, math.prod(self.patch) * self.out_channels)
        self.final_layer.scale_shift_table = nn.Parameter(torch.zeros(2, d))
        self.register_buffer("rope_freqs", torch.zeros(d // heads // 2), persistent=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, action_bins: torch.Tensor, n_ctx: int, image_size: int):
        """``x`` [B, C, T, H, W]; ``timestep`` [B] for the generated frames; ``action_bins`` [B, T - n_ctx, A]."""
        b, _, t, h, w = x.shape
        pt, ph, pw = self.patch
        n_t, n_h, n_w = t // pt, h // ph, w // pw
        n_s = n_h * n_w
        dt = self.x_embedder.proj.weight.dtype

        tok = rearrange(self.x_embedder.proj(x.to(dt)), "b c t hh ww -> b t (hh ww) c")
        scale = image_size / self.cfg["input_sq_size"]
        tok = tok + _pos_embed_2d(self.dim, n_h, n_w, scale, round(n_s ** 0.5), x.device).to(dt)

        t_emb = self.t_embedder(timestep)                                         # [B, C]
        t0_emb = self.t_embedder(torch.full_like(timestep, CONTEXT_TIMESTEP))     # [B, C]
        act = self.action_embedder.fc(action_bins.to(dt))                         # [B, T_gen, C]
        n_gen = n_t - n_ctx
        ctx_mod = self.t_block(t0_emb)[:, None].expand(-1, n_ctx, -1)
        spatial_mod = torch.cat([ctx_mod, self.t_block(t_emb[:, None] + act)], 1).view(b, n_t, 6, self.dim)
        temporal_mod = torch.cat([ctx_mod, self.t_block(t_emb)[:, None].expand(-1, n_gen, -1)], 1)
        temporal_mod = temporal_mod.view(b, n_t, 6, self.dim)

        pos = torch.arange(n_t, device=x.device, dtype=torch.float32)
        ang = (pos[:, None] * self.rope_freqs.float()[None]).repeat_interleave(2, dim=-1)
        rope = (ang.cos().to(dt), ang.sin().to(dt))
        for sb, tb in zip(self.spatial_blocks, self.temporal_blocks):
            tok = sb(tok, spatial_mod, None)
            tok = tb(tok, temporal_mod, rope)

        final_t = torch.cat([t0_emb[:, None].expand(-1, n_ctx, -1), t_emb[:, None].expand(-1, n_gen, -1)], 1)
        shift, sc = (self.final_layer.scale_shift_table + final_t[:, :, None]).unsqueeze(2).unbind(3)
        tok = self.final_layer.linear(F.layer_norm(tok, (self.dim,), eps=1e-6) * (1 + sc) + shift)
        out = rearrange(
            tok, "b t (hh ww) (p q r c) -> b c (t p) (hh q) (ww r)",
            hh=n_h, ww=n_w, p=pt, q=ph, r=pw, c=self.out_channels,
        )
        return out[:, : self.in_channels]


def load_stdit3(ckpt_dir: str, device, dtype=torch.bfloat16) -> STDiT3:
    from safetensors.torch import load_file

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd.update(load_file(os.path.join(ckpt_dir, shard)))
    with torch.device("meta"):
        model = STDiT3(cfg)
    rope = sd.pop("rope.freqs")
    for k in [k for k in sd if k.startswith("y_embedder.")]:
        sd.pop(k)  # text-caption projector of the base text-to-video model, unused
    model.load_state_dict(sd, strict=True, assign=True)
    model.rope_freqs = rope.float()
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return model.to(device).eval().requires_grad_(False)


@dataclass
class OpenSoraWorldModelConfig:
    vae_subdir: str = "vae"
    sampling_steps: int = 10
    context_frames: int = 4
    chunk_size: int = 8


class OpenSoraActionWorldModel(nn.Module):
    """Frozen video world model: (context frames, actions) -> next frames."""

    # Any number of context frames is usable: one frame per latent.
    context_frame_step = 1

    def __init__(self, ckpt_dir: str, cfg: OpenSoraWorldModelConfig, device, dtype=torch.bfloat16) -> None:
        from diffusers import AutoencoderKL

        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.dit = load_stdit3(ckpt_dir, self.device, dtype)
        self.vae = AutoencoderKL.from_pretrained(os.path.join(ckpt_dir, cfg.vae_subdir), torch_dtype=dtype)
        self.vae = self.vae.to(self.device).eval().requires_grad_(False)
        self.vae_scale = self.vae.config.scaling_factor
        with open(os.path.join(ckpt_dir, "dataset_statistics.json")) as f:
            stats = json.load(f)["action"]
        self.q01 = torch.tensor(stats["q01"], dtype=torch.float32, device=self.device)
        self.q99 = torch.tensor(stats["q99"], dtype=torch.float32, device=self.device)
        self.bin_edges = torch.linspace(-1.0, 1.0, NUM_ACTION_BINS, device=self.device)
        steps = cfg.sampling_steps
        self.timesteps = [1000.0 * (1 - i / steps) for i in range(steps)] + [0.0]

    def action_bins(self, actions: torch.Tensor) -> torch.Tensor:
        """Env actions ``[..., 7]`` -> q01/q99-normalized, 256-bin quantized values in ``[-1, 1]``."""
        a = actions.to(self.device, torch.float32)
        n = (2 * (a - self.q01) / (self.q99 - self.q01) - 1).clamp(-1, 1)
        idx = torch.bucketize(n, self.bin_edges, right=True).float()
        return (idx - (NUM_ACTION_BINS + 1) / 2) / ((NUM_ACTION_BINS - 1) / 2)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, T, H, W, 3]`` -> latents ``[B, 4, T, h, w]``."""
        b, t = frames.shape[:2]
        x = frames.to(self.device).flatten(0, 1).permute(0, 3, 1, 2).to(self.dtype) / 127.5 - 1
        z = self.vae.encode(x).latent_dist.mode() * self.vae_scale
        return rearrange(z.float(), "(b t) c h w -> b c t h w", b=b, t=t)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents ``[B, 4, T, h, w]`` -> uint8 ``[B, T, H, W, 3]``."""
        b, _, t = latents.shape[:3]
        z = rearrange(latents, "b c t h w -> (b t) c h w").to(self.dtype) / self.vae_scale
        x = self.vae.decode(z).sample.float().clamp(-1, 1)
        x = ((x + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)
        return x.reshape(b, t, *x.shape[1:])

    @torch.no_grad()
    def predict_frames(
        self,
        ctx_frames: torch.Tensor,
        ctx_actions: Optional[torch.Tensor],
        tgt_actions: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """uint8 context ``[B, T, H, W, 3]`` + ``tgt_actions`` ``[B, chunk, 7]`` -> uint8 ``[B, chunk, H, W, 3]``.

        Short contexts are left-padded with their earliest frame; context
        actions are not used.
        """
        cfg = self.cfg
        c, n = cfg.context_frames, cfg.chunk_size
        if ctx_frames.shape[1] < c:
            pad = ctx_frames[:, :1].expand(-1, c - ctx_frames.shape[1], -1, -1, -1)
            ctx_frames = torch.cat([pad, ctx_frames], dim=1)
        ctx_frames = ctx_frames[:, -c:]
        b = ctx_frames.shape[0]

        ctx = self.encode(ctx_frames)
        bins = self.action_bins(tgt_actions)
        x = torch.randn((b, ctx.shape[1], n, *ctx.shape[3:]), generator=generator, device=self.device)
        for i in range(cfg.sampling_steps):
            t, t_next = self.timesteps[i], self.timesteps[i + 1]
            ts = torch.full((b,), t, device=self.device)
            v = self.dit(torch.cat([ctx, x], dim=2), ts, bins, c, ctx_frames.shape[2])[:, :, c:].float()
            x = x + (t - t_next) / 1000.0 * v
        return self.decode(x)
