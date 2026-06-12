from __future__ import annotations

import torch
from torch import nn


class PPOPolicyNetwork(nn.Module):
    """Reference network matching the manuscript architecture description."""

    def __init__(self, input_channels: int = 4, action_dim: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
