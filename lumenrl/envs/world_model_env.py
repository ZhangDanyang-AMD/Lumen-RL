"""Batched world-model environment: reset from recorded states, step by imagination.

Each environment keeps a short window of past frames and the actions that
produced them. A step applies one action chunk: the world model imagines the
next ``chunk_size`` frames, the reward model scores every imagined frame, and
the episode ends on the first frame classified as task success or when the
step budget runs out.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class InitState:
    frames: np.ndarray        # uint8 [T, H, W, 3], T = 1 + k * context_frame_step
    actions: np.ndarray       # float32 [T, 7]; actions[i] produced frames[i]
    instruction: str
    source: str


def _bucket(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _valid_context_len(n: int, limit: int, step: int) -> int:
    best = 1
    k = 1
    while k <= min(n, limit):
        best = k
        k += step
    return best


class WorldModelInitDataset:
    """Initial states from the ``.npy`` episode files shipped with the world model.

    Every file is a list of per-step dicts with ``image``, ``delta_action`` and
    ``instruction`` (optionally wrapped in a length-1 object array). Only the
    first ``context_frames`` steps are kept, trimmed to a length the world
    model accepts (``1 + k * context_frame_step``).
    """

    def __init__(self, data_dir: str, context_frames: int = 5, context_frame_step: int = 1) -> None:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not files:
            raise FileNotFoundError(f"no init-state .npy files under {data_dir}")
        self.states: list[InitState] = []
        for f in files:
            episode = np.load(f, allow_pickle=True)
            t = _valid_context_len(len(episode), context_frames, context_frame_step)
            steps = episode[:t]
            self.states.append(InitState(
                frames=np.stack([s["image"] for s in steps]).astype(np.uint8),
                actions=np.stack([np.asarray(s["delta_action"], dtype=np.float32) for s in steps]),
                instruction=str(steps[0]["instruction"]),
                source=os.path.basename(f),
            ))
        self.instructions = sorted({s.instruction for s in self.states})
        logger.info(
            "World-model init states: %d episodes, %d tasks, context lengths %s",
            len(self.states), len(self.instructions),
            sorted({len(s.frames) for s in self.states}),
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> InitState:
        return self.states[i]


class WorldModelEnv:
    """``num_envs`` imagined environments sharing one world model on one GPU."""

    def __init__(
        self,
        world_model: Any,
        reward_model: torch.nn.Module,
        max_episode_steps: int,
        context_frames: int = 5,
        success_logit_threshold: float = 0.0,
        wm_batch_size: int = 32,
        seed: int = 0,
        record_frames: bool = False,
    ) -> None:
        self.ctx_step = int(getattr(world_model, "context_frame_step", 1))
        if (context_frames - 1) % self.ctx_step != 0:
            raise ValueError(f"context_frames must be 1 + a multiple of {self.ctx_step}")
        self.wm = world_model
        self.rm = reward_model
        self.chunk = world_model.cfg.chunk_size
        self.max_episode_steps = max_episode_steps
        self.context_frames = context_frames
        self.threshold = success_logit_threshold
        self.wm_batch_size = wm_batch_size
        self.device = world_model.device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        # Keeps every frame of every episode on the CPU (for videos).
        self.record_frames = record_frames
        self.history: list[list[torch.Tensor]] = []

        self.num_envs = 0
        self.frames: Optional[torch.Tensor] = None   # uint8 [N, C, H, W, 3], right-aligned
        self.actions: Optional[torch.Tensor] = None  # float [N, C, 7], aligned with frames
        self.ctx_len: Optional[torch.Tensor] = None  # [N] number of valid trailing frames
        self.elapsed: Optional[torch.Tensor] = None
        self.done: Optional[torch.Tensor] = None
        self.success: Optional[torch.Tensor] = None
        self.instructions: list[str] = []

    def reset(self, inits: list[InitState]) -> dict:
        n = len(inits)
        h, w = inits[0].frames.shape[1:3]
        c = self.context_frames
        self.num_envs = n
        self.frames = torch.zeros((n, c, h, w, 3), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros((n, c, 7), dtype=torch.float32, device=self.device)
        self.ctx_len = torch.zeros(n, dtype=torch.long, device=self.device)
        for i, s in enumerate(inits):
            t = min(len(s.frames), c)
            self.frames[i, c - t:] = torch.from_numpy(s.frames[-t:]).to(self.device)
            self.actions[i, c - t:] = torch.from_numpy(s.actions[-t:]).to(self.device)
            self.ctx_len[i] = t
        self.elapsed = torch.zeros(n, dtype=torch.long, device=self.device)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.instructions = [s.instruction for s in inits]
        if self.record_frames:
            self.history = [[torch.from_numpy(s.frames[-1:])] for s in inits]
        return self.observation()

    def observation(self) -> dict:
        return {"images": self.frames[:, -1], "instructions": self.instructions}

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """Apply ``actions`` ``[N, chunk, 7]`` to every env that is not done."""
        assert self.frames is not None, "call reset() first"
        actions = actions.to(self.device, torch.float32)
        active = (~self.done).nonzero().squeeze(-1)
        reward = torch.zeros(self.num_envs, device=self.device)
        frame_logits = torch.full((self.num_envs, self.chunk), float("-inf"), device=self.device)
        c = self.context_frames

        for t in self.ctx_len[active].unique().tolist():
            ids = active[self.ctx_len[active] == t]
            for s in range(0, len(ids), self.wm_batch_size):
                b = ids[s:s + self.wm_batch_size]
                # Pad to a power of two: every new batch size triggers a slow
                # MIOpen algorithm search for the VAE's 3D convolutions.
                padded = torch.cat([b, b[-1:].expand(_bucket(len(b)) - len(b))])
                ctx_actions = self.actions[padded, c - t + 1:] if t > 1 else None
                new = self.wm.predict_frames(
                    self.frames[padded, c - t:], ctx_actions, actions[padded], self.generator
                )[: len(b)]
                frame_logits[b] = self._score(new)
                if self.record_frames:
                    for i, f in zip(b.tolist(), new.cpu()):
                        self.history[i].append(f)
                self.frames[b] = torch.cat([self.frames[b], new], dim=1)[:, -c:]
                self.actions[b] = torch.cat([self.actions[b], actions[b]], dim=1)[:, -c:]
        self.ctx_len[active] = c

        self.elapsed[active] += self.chunk
        hit = (frame_logits > self.threshold).any(dim=1) & ~self.done
        reward[hit] = 1.0
        self.success |= hit
        self.done |= hit | (self.elapsed >= self.max_episode_steps)
        info = {"frame_logits": frame_logits, "active": active}
        return self.observation(), reward, self.done.clone(), info

    def _score(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        return self.rm(x).view(b, t)
