"""Value-function worker for critic-based RL algorithms."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumenrl.core.protocol import DataProto
from lumenrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class _ValueCore(nn.Module):
    """Small GRU value network."""

    def __init__(self, vocab_size: int = 32000, dim: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.v_head = nn.Linear(dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        out, _ = self.rnn(x)
        return self.v_head(out).squeeze(-1)


class CriticWorker(BaseWorker):
    """Computes V(s_t) along prompt/response tokens and supports a train step."""

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(rank, world_size, config)
        self._critic: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Load or construct the critic architecture."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        vocab = int(policy.get("critic_vocab_size", 32000))
        dim = int(policy.get("critic_hidden_size", 256))
        self._critic = _ValueCore(vocab_size=vocab, dim=dim).to(self._device)
        lr = float(policy.get("critic_lr", 1e-4))
        self._optimizer = torch.optim.AdamW(self._critic.parameters(), lr=lr)
        self._log.info("CriticWorker: value network ready (vocab=%d, dim=%d).", vocab, dim)

    def compute_values(self, batch: DataProto) -> DataProto:
        """Return per-token value estimates."""
        if self._critic is None:
            raise RuntimeError("init_model() must be called before compute_values().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        self._critic.eval()
        input_ids = batch["input_ids"].to(self._device)
        with torch.no_grad():
            values = self._critic(input_ids)
        return DataProto(
            tensors={"values": values.cpu(), "input_ids": batch["input_ids"]},
            meta=dict(batch.meta),
        )

    def train_step(self, batch: DataProto) -> dict[str, float]:
        """TD-style regression to ``returns`` when provided, else zeros target."""
        if self._critic is None or self._optimizer is None:
            raise RuntimeError("init_model() must be called before train_step().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        self._critic.train()
        input_ids = batch["input_ids"].to(self._device)
        preds = self._critic(input_ids)
        returns = batch.tensors.get("returns")
        if returns is None:
            returns = torch.zeros_like(preds)
        else:
            returns = returns.to(self._device)
            if returns.shape != preds.shape:
                raise ValueError("returns tensor must match critic output shape")

        loss = F.mse_loss(preds, returns)
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()
        return {"critic_loss": float(loss.detach().cpu())}

    def cleanup(self) -> None:
        self._critic = None
        self._optimizer = None
        super().cleanup()
