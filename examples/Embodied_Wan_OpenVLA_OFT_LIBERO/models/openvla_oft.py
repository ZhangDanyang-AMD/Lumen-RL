"""OpenVLA-OFT policy (discrete action-token head, parallel decoding).

Self-contained re-implementation that loads the published OpenVLA-OFT
safetensors directly, so it runs on current ``transformers`` / ``timm``
without the remote code or the forked ``transformers`` it depends on.

Model layout (matches checkpoint key names):

* ``vision_backbone.featurizer``       DINOv2 ViT-L/14 (+4 registers)
* ``vision_backbone.fused_featurizer`` SigLIP So400m/14
* ``projector.fc{1,2,3}``             fused GELU MLP, vision -> LLM width
* ``language_model``                  Llama-2 7B, vocab padded to 32064

Sequence fed to the LLM, per sample (pads only at the very end)::

    [BOS] [256 patch embeddings] [prompt[1:]] [56 zeroed action slots] [STOP] [PAD...]

Attention is bidirectional over non-pad tokens. The logits that predict the
56 action tokens (8 steps x 7 dims) are read at the positions immediately
before each action slot and restricted to the 256 action-bin tokens.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
NUM_ACTION_TOKENS = NUM_ACTIONS_CHUNK * ACTION_DIM
NUM_BINS = 256
STOP_TOKEN_ID = 2
ACTION_PLACEHOLDER_ID = 1
EMPTY_TOKEN_ID = 29871
# Action bins occupy the last 256 ids of the un-padded 32000-token vocab.
ACTION_VOCAB_BEGIN = 32000 - NUM_BINS
ACTION_VOCAB_END = 32000

DINO_TIMM_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_TIMM_ID = "vit_so400m_patch14_siglip_224"
IMAGE_SIZE = 224
DINO_MEAN = (0.484375, 0.455078125, 0.40625)
DINO_STD = (0.228515625, 0.2236328125, 0.224609375)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# The checkpoint's text_config only overrides vocab/pad, so everything else is
# the LlamaConfig default it was trained with (notably rms_norm_eps=1e-6, not
# the 1e-5 of the original Llama-2 release).
LLAMA2_7B = dict(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    hidden_act="silu",
    tie_word_embeddings=False,
)


class _PenultimateViT(nn.Module):
    """timm ViT that returns patch tokens from its second-to-last block.

    The final block, final norm and pooling head are never used by the
    policy and are removed so they hold no parameters (FSDP would otherwise
    see trainable params that never receive gradients).
    """

    def __init__(self, timm_id: str) -> None:
        super().__init__()
        import timm

        vit = timm.create_model(timm_id, pretrained=False, num_classes=0, img_size=IMAGE_SIZE)
        vit.blocks = nn.Sequential(*list(vit.blocks)[:-1])
        vit.norm = nn.Identity()
        vit.fc_norm = nn.Identity()
        vit.head = nn.Identity()
        if getattr(vit, "attn_pool", None) is not None:
            vit.attn_pool = None
        self.vit = vit
        self.embed_dim = int(vit.embed_dim)
        self.num_patches = int(vit.patch_embed.num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)
        x = vit.blocks(x)
        return x[:, vit.num_prefix_tokens:]


class FusedDinoSigLIPBackbone(nn.Module):
    """6-channel input: channels 0-2 go to DINOv2, 3-5 to SigLIP."""

    def __init__(self) -> None:
        super().__init__()
        self.featurizer = _PenultimateViT(DINO_TIMM_ID)
        self.fused_featurizer = _PenultimateViT(SIGLIP_TIMM_ID)
        self.embed_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim
        self.num_patches = self.featurizer.num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dino = self.featurizer(pixel_values[:, :3])
        siglip = self.fused_featurizer(pixel_values[:, 3:])
        return torch.cat([dino, siglip], dim=-1)


class FusedGELUProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim)
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim)
        self.fc3 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(x)))))


class OpenVLAOFTPolicy(nn.Module):
    """Vision-language-action policy producing 56 action-bin logits per sample."""

    def __init__(self, vocab_size: int = 32064, pad_token_id: int = 32000) -> None:
        super().__init__()
        from transformers import LlamaConfig, LlamaForCausalLM

        self.vision_backbone = FusedDinoSigLIPBackbone()
        llm_cfg = LlamaConfig(vocab_size=vocab_size, pad_token_id=pad_token_id, **LLAMA2_7B)
        llm_cfg._attn_implementation = "sdpa"
        self.projector = FusedGELUProjector(self.vision_backbone.embed_dim, llm_cfg.hidden_size)
        self.language_model = LlamaForCausalLM(llm_cfg)
        self.num_patches = self.vision_backbone.num_patches

    @property
    def config(self):
        return self.language_model.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None) -> None:
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False}
        )

    def fsdp_wrap_modules(self) -> list[nn.Module]:
        """Transformer blocks to shard individually under FSDP2."""
        return [
            *self.vision_backbone.featurizer.vit.blocks,
            *self.vision_backbone.fused_featurizer.vit.blocks,
            *self.language_model.model.layers,
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        action_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return action-bin logits ``[B, 56, 256]`` (fp32, temperature-scaled).

        With ``action_tokens`` (bin indices in ``[0, 256)``) also returns
        ``log_probs`` ``[B, 56]``; ``entropy`` when requested.
        """
        bsz, text_len = input_ids.shape
        device = input_ids.device
        lm = self.language_model

        prompt_len = attention_mask.sum(dim=1) - (NUM_ACTION_TOKENS + 1)
        pos = torch.arange(text_len, device=device).unsqueeze(0)
        slots = (pos >= prompt_len[:, None]) & (pos < prompt_len[:, None] + NUM_ACTION_TOKENS)

        embeds = lm.model.embed_tokens(input_ids)
        embeds = embeds * (~slots).unsqueeze(-1).to(embeds.dtype)
        patches = self.projector(self.vision_backbone(pixel_values.to(embeds.dtype)))
        n_patch = patches.shape[1]

        hidden = torch.cat([embeds[:, :1], patches, embeds[:, 1:]], dim=1)
        keep = torch.cat(
            [
                attention_mask[:, :1],
                torch.ones(bsz, n_patch, dtype=attention_mask.dtype, device=device),
                attention_mask[:, 1:],
            ],
            dim=1,
        ).bool()
        seq_len = hidden.shape[1]
        mask4d = keep[:, None, None, :].expand(bsz, 1, seq_len, seq_len)

        out = lm.model(inputs_embeds=hidden, attention_mask=mask4d, use_cache=False)
        h = out.last_hidden_state
        read_pos = n_patch + prompt_len[:, None] - 1 + torch.arange(NUM_ACTION_TOKENS, device=device)
        h = h.gather(1, read_pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        logits = lm.lm_head(h)[..., ACTION_VOCAB_BEGIN:ACTION_VOCAB_END].float()
        if temperature != 1.0:
            logits = logits / temperature

        result: dict[str, torch.Tensor] = {"logits": logits}
        if action_tokens is not None or calculate_entropy:
            logp_all = torch.log_softmax(logits, dim=-1)
            if action_tokens is not None:
                result["log_probs"] = logp_all.gather(-1, action_tokens.long().unsqueeze(-1)).squeeze(-1)
            if calculate_entropy:
                result["entropy"] = -(logp_all.exp() * logp_all).sum(-1)
        return result


class OpenVLAOFTProcessor:
    """Prompt tokenization, image preprocessing and action (de)tokenization."""

    def __init__(self, ckpt_dir: str, unnorm_key: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
        with open(os.path.join(ckpt_dir, "dataset_statistics.json")) as f:
            stats = json.load(f)
        if unnorm_key not in stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not in dataset_statistics ({list(stats)})")
        act = stats[unnorm_key]["action"]
        self.q01 = torch.tensor(act["q01"], dtype=torch.float32)
        self.q99 = torch.tensor(act["q99"], dtype=torch.float32)
        self.norm_mask = torch.tensor(act["mask"], dtype=torch.bool)
        bins = torch.linspace(-1.0, 1.0, NUM_BINS)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self._prompt_cache: dict[str, list[int]] = {}
        self.num_action_tokens = NUM_ACTION_TOKENS

    @staticmethod
    def prompt_text(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def prompt_ids(self, instruction: str) -> list[int]:
        ids = self._prompt_cache.get(instruction)
        if ids is None:
            ids = self.tokenizer(self.prompt_text(instruction))["input_ids"]
            # Training sequences have the empty SentencePiece token between
            # "Out:" and the actions; without it every action after the first
            # in the chunk degenerates.
            if ids[-1] != EMPTY_TOKEN_ID:
                ids = ids + [EMPTY_TOKEN_ID]
            self._prompt_cache[instruction] = ids
        return ids

    def build_text_batch(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """``[prompt, 56 placeholders, STOP]`` per row, right-padded."""
        rows = [
            self.prompt_ids(s) + [ACTION_PLACEHOLDER_ID] * NUM_ACTION_TOKENS + [STOP_TOKEN_ID]
            for s in instructions
        ]
        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return ids, mask

    @staticmethod
    def preprocess_images(images: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, H, W, 3]`` -> float ``[B, 6, 224, 224]`` (DINO | SigLIP)."""
        x = images.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", antialias=True, align_corners=False)
            x = x.clamp_(0.0, 1.0)
        dmean = x.new_tensor(DINO_MEAN).view(1, 3, 1, 1)
        dstd = x.new_tensor(DINO_STD).view(1, 3, 1, 1)
        smean = x.new_tensor(SIGLIP_MEAN).view(1, 3, 1, 1)
        sstd = x.new_tensor(SIGLIP_STD).view(1, 3, 1, 1)
        return torch.cat([(x - dmean) / dstd, (x - smean) / sstd], dim=1)

    def tokens_to_normalized(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Bin indices ``[B, 56]`` -> normalized actions ``[B, 8, 7]`` in ``[-1, 1]``."""
        vocab_ids = action_tokens.long() + ACTION_VOCAB_BEGIN
        discretized = (32000 - vocab_ids - 1).clamp(0, self.bin_centers.numel() - 1)
        centers = self.bin_centers.to(action_tokens.device)
        return centers[discretized].view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)

    def unnormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(normalized.device)
        q99 = self.q99.to(normalized.device)
        mask = self.norm_mask.to(normalized.device)
        raw = 0.5 * (normalized + 1.0) * (q99 - q01 + 1e-8) + q01
        return torch.where(mask, raw, normalized)

    def tokens_to_env_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Bin indices -> LIBERO env actions ``[B, 8, 7]``.

        The policy's gripper output is the dataset convention (1 = open);
        LIBERO expects -1 = open, +1 = close.
        """
        actions = self.unnormalize(self.tokens_to_normalized(action_tokens))
        gripper = torch.where(actions[..., -1] > 0.5, -1.0, 1.0)
        return torch.cat([actions[..., :-1], gripper.unsqueeze(-1)], dim=-1)



def _checkpoint_state_dict(ckpt_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {ckpt_dir}")
    sd: dict[str, torch.Tensor] = {}
    for f in files:
        sd.update(load_file(f))
    out = {}
    for k, v in sd.items():
        k = k.replace(".ls1.scale_factor", ".ls1.gamma").replace(".ls2.scale_factor", ".ls2.gamma")
        k = k.replace("vision_backbone.featurizer.", "vision_backbone.featurizer.vit.")
        k = k.replace("vision_backbone.fused_featurizer.", "vision_backbone.fused_featurizer.vit.")
        out[k] = v
    return out


def load_openvla_oft(
    ckpt_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> OpenVLAOFTPolicy:
    """Build the policy on the meta device and assign checkpoint weights."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg: dict[str, Any] = json.load(f)
    vocab = int(cfg.get("text_config", {}).get("vocab_size", 32064))
    pad_id = int(cfg.get("pad_token_id", 32000))

    with torch.device("meta"):
        policy = OpenVLAOFTPolicy(vocab_size=vocab, pad_token_id=pad_id)

    sd = _checkpoint_state_dict(ckpt_dir)
    expected = set(policy.state_dict().keys())
    unused = sorted(k for k in sd if k not in expected)
    for k in unused:
        sd.pop(k)
    missing, _ = policy.load_state_dict(sd, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"OpenVLA-OFT checkpoint missing {len(missing)} keys, e.g. {missing[:5]}")
    allowed_unused = ("featurizer.vit.blocks.", "featurizer.vit.norm.", "featurizer.vit.attn_pool.")
    bad = [k for k in unused if not any(a in k for a in allowed_unused)]
    if bad:
        raise RuntimeError(f"unexpected OpenVLA-OFT checkpoint keys: {bad[:5]}")

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    lm = policy.language_model
    with torch.device("cpu"):
        lm.model.rotary_emb = LlamaRotaryEmbedding(config=lm.config)
    policy.to(device=device)
    # Parameters only: the rotary inv_freq buffer must stay fp32.
    for p in policy.parameters():
        if p.dtype != dtype:
            p.data = p.data.to(dtype)
    leftover = [n for n, t in list(policy.named_parameters()) + list(policy.named_buffers()) if t.is_meta]
    if leftover:
        raise RuntimeError(f"tensors left on meta device: {leftover[:5]}")
    logger.info("Loaded OpenVLA-OFT from %s (%d unused vision-head tensors dropped)", ckpt_dir, len(unused))
    return policy
