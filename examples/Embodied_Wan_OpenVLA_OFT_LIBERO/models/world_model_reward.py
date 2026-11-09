"""Frame-level task-success classifier used as the world-model reward.

ResNet-18 layout with biased convolutions: a 7x7 stem, four stages of two
residual blocks (a 1x1 projection shortcut where the shape changes), global
average pooling and a single-logit head. Inputs are RGB frames scaled to
``[-1, 1]``, the same range the world model decodes to.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Residual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (stride != 1 or in_ch != out_ch) else None
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return F.relu(y + x)


def _stage(in_ch: int, out_ch: int, first: bool) -> nn.Sequential:
    stride = 1 if first else 2
    return nn.Sequential(_Residual(in_ch, out_ch, stride), _Residual(out_ch, out_ch, 1))


class FrameSuccessClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            ),
            _stage(64, 64, first=True),
            _stage(64, 128, first=False),
            _stage(128, 256, first=False),
            _stage(256, 512, first=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` ``[N, 3, H, W]`` in ``[-1, 1]`` -> success logits ``[N]``."""
        return self.net(frames).squeeze(-1)


def load_reward_model(path: str, device: str | torch.device = "cpu") -> FrameSuccessClassifier:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model = FrameSuccessClassifier()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
