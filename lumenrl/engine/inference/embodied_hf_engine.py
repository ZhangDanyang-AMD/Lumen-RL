"""HF rollout for VLA policies interacting with a batched (world-model) env.

The rollout policy is a plain bf16 HuggingFace-style replica on the local
GPU. It decodes all action tokens of a chunk in one forward pass (parallel
decoding), samples them at the rollout temperature and keeps their
log-probabilities. Weights are refreshed from the FSDP2 actor every step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from lumenrl.envs.world_model_env import InitState, WorldModelEnv

logger = logging.getLogger(__name__)


def sample_action_tokens(
    logits: torch.Tensor, greedy: bool = False, generator: Optional[torch.Generator] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample bin indices from temperature-scaled logits ``[B, A, V]``; returns (tokens, log-probs)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    if greedy:
        tokens = logp.argmax(dim=-1)
    else:
        probs = logp.exp().reshape(-1, logp.shape[-1])
        tokens = torch.multinomial(probs, 1, generator=generator).view(logp.shape[:-1])
    return tokens, logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@dataclass
class EmbodiedTrajectoryBatch:
    """Flat per-step samples plus per-episode outcomes from one rollout."""

    images: torch.Tensor             # uint8 [S, H, W, 3] observation the action was chosen from
    instruction_ids: torch.Tensor    # long [S] index into ``instructions``
    action_tokens: torch.Tensor      # long [S, 56] action-bin indices
    rollout_log_probs: torch.Tensor  # float [S, 56] log-probs under the rollout policy
    env_ids: torch.Tensor            # long [S]
    step_ids: torch.Tensor           # long [S] chunk index within the episode
    instructions: list[str]
    success: torch.Tensor            # bool [N]
    episode_chunks: torch.Tensor     # long [N] action chunks executed
    metrics: dict[str, float] = field(default_factory=dict)
    returns: Optional[torch.Tensor] = None  # float [N], RLinf relative-reward return

    @property
    def num_samples(self) -> int:
        return int(self.action_tokens.shape[0])

    @classmethod
    def cat(cls, parts: list[EmbodiedTrajectoryBatch]) -> EmbodiedTrajectoryBatch:
        """Join rollout waves. ``env_ids`` are shifted so they still index ``success``."""
        if len(parts) == 1:
            return parts[0]
        vocab: list[str] = []
        index: dict[str, int] = {}
        inst_ids: list[torch.Tensor] = []
        env_ids: list[torch.Tensor] = []
        offset = 0
        for part in parts:
            local = []
            for text in part.instructions:
                if text not in index:
                    index[text] = len(vocab)
                    vocab.append(text)
                local.append(index[text])
            inst_ids.append(torch.tensor(local, dtype=torch.long)[part.instruction_ids])
            env_ids.append(part.env_ids + offset)
            offset += int(part.success.shape[0])
        episodes = [int(p.success.shape[0]) for p in parts]
        n_ep = sum(episodes)
        metrics: dict[str, float] = {
            "rollout/success_rate": sum(p.metrics["rollout/success_rate"] * n for p, n in zip(parts, episodes)) / n_ep,
            "rollout/episode_chunks": sum(p.metrics["rollout/episode_chunks"] * n for p, n in zip(parts, episodes)) / n_ep,
            "rollout/num_samples": float(sum(p.metrics["rollout/num_samples"] for p in parts)),
        }
        if all("rollout/episode_return" in p.metrics for p in parts):
            metrics["rollout/episode_return"] = sum(
                p.metrics["rollout/episode_return"] * n for p, n in zip(parts, episodes)
            ) / n_ep
        for key in (
            "timing/rollout_policy_s", "timing/rollout_env_s",
            "timing/env_reset_s", "timing/env_sim_s", "rollout/env_sim_steps",
        ):
            if any(key in p.metrics for p in parts):
                metrics[key] = float(sum(p.metrics.get(key, 0.0) for p in parts))
        return cls(
            images=torch.cat([p.images for p in parts]),
            instruction_ids=torch.cat(inst_ids),
            action_tokens=torch.cat([p.action_tokens for p in parts]),
            rollout_log_probs=torch.cat([p.rollout_log_probs for p in parts]),
            env_ids=torch.cat(env_ids),
            step_ids=torch.cat([p.step_ids for p in parts]),
            instructions=vocab,
            success=torch.cat([p.success for p in parts]),
            episode_chunks=torch.cat([p.episode_chunks for p in parts]),
            metrics=metrics,
            returns=torch.cat([p.returns for p in parts]) if all(p.returns is not None for p in parts) else None,
        )


class EmbodiedHFRollout:
    def __init__(
        self,
        policy: nn.Module,
        processor: Any,
        temperature: float = 1.0,
        batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.policy = policy.eval().requires_grad_(False)
        self.processor = processor
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = next(policy.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def update_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> int:
        params = dict(self.policy.named_parameters())
        n = 0
        for name, tensor in named_tensors:
            p = params.get(name)
            if p is None:
                raise KeyError(f"rollout policy has no parameter {name!r}")
            p.copy_(tensor.to(device=p.device, dtype=p.dtype))
            n += 1
        if n != len(params):
            raise RuntimeError(f"weight sync updated {n}/{len(params)} parameters")
        return n

    @torch.no_grad()
    def act(self, images: torch.Tensor, instructions: list[str], greedy: bool = False):
        """Observations -> (bin indices [B,56], log-probs [B,56], env actions [B,8,7])."""
        tokens, logps = [], []
        for s in range(0, len(instructions), self.batch_size):
            ids, mask = self.processor.build_text_batch(instructions[s:s + self.batch_size])
            pix = self.processor.preprocess_images(images[s:s + self.batch_size].to(self.device))
            out = self.policy(ids.to(self.device), mask.to(self.device), pix, temperature=self.temperature)
            tok, lp = sample_action_tokens(out["logits"], greedy=greedy, generator=self.generator)
            tokens.append(tok)
            logps.append(lp)
        tokens = torch.cat(tokens)
        return tokens, torch.cat(logps), self.processor.tokens_to_env_actions(tokens)

    @torch.no_grad()
    def collect(
        self,
        env: WorldModelEnv,
        inits: list[InitState],
        greedy: bool = False,
        max_chunks: Optional[int] = None,
    ) -> EmbodiedTrajectoryBatch:
        obs = env.reset(inits)
        n = env.num_envs
        instructions = sorted(set(obs["instructions"]))
        inst_index = torch.tensor([instructions.index(s) for s in obs["instructions"]])
        limit = max_chunks or -(-env.max_episode_steps // env.chunk)

        rec: dict[str, list[torch.Tensor]] = {k: [] for k in ("img", "tok", "lp", "env", "step")}
        chunks = torch.zeros(n, dtype=torch.long)
        policy_s = env_s = 0.0
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        for step in range(limit):
            active = (~env.done).nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                break
            idx = active.cpu()
            t0.record()
            images = obs["images"][active]
            tok, lp, env_act = self.act(images, [obs["instructions"][i] for i in idx.tolist()], greedy)
            t1.record()
            actions = torch.zeros((n, env.chunk, env_act.shape[-1]), device=env.device)
            actions[active] = env_act.to(env.device)
            obs, _, _, _ = env.step(actions)
            t2.record()
            torch.cuda.synchronize()
            policy_s += t0.elapsed_time(t1) / 1e3
            env_s += t1.elapsed_time(t2) / 1e3

            rec["img"].append(images.cpu())
            rec["tok"].append(tok.cpu())
            rec["lp"].append(lp.float().cpu())
            rec["env"].append(idx)
            rec["step"].append(torch.full_like(idx, step))
            chunks[idx] += 1

        env_ids = torch.cat(rec["env"])
        success = env.success.cpu()
        returns = None
        if getattr(env, "use_rel_reward", False):
            returns = env.episode_return.detach().float().cpu()
        metrics = {
            "rollout/success_rate": success.float().mean().item(),
            "rollout/episode_chunks": chunks.float().mean().item(),
            "rollout/num_samples": float(env_ids.numel()),
            "timing/rollout_policy_s": policy_s,
            "timing/rollout_env_s": env_s,
        }
        if returns is not None:
            metrics["rollout/episode_return"] = returns.mean().item()
        env_timing = getattr(env, "timing", None)
        if env_timing:
            metrics["timing/env_reset_s"] = float(env_timing["reset_s"])
            metrics["timing/env_sim_s"] = float(env_timing["sim_s"])
            metrics["rollout/env_sim_steps"] = float(env_timing["sim_steps"])
        return EmbodiedTrajectoryBatch(
            images=torch.cat(rec["img"]),
            instruction_ids=inst_index[env_ids],
            action_tokens=torch.cat(rec["tok"]),
            rollout_log_probs=torch.cat(rec["lp"]),
            env_ids=env_ids,
            step_ids=torch.cat(rec["step"]),
            instructions=instructions,
            success=success,
            episode_chunks=chunks,
            metrics=metrics,
            returns=returns,
        )


__all__ = ["EmbodiedHFRollout", "EmbodiedTrajectoryBatch", "sample_action_tokens"]
