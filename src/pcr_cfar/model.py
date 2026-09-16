"""Tiny fully convolutional residual CFAR head."""

from __future__ import annotations

import torch
from torch import nn


class ResidualCfarNet(nn.Module):
    """Predict a heterogeneity-gated log-threshold residual on top of CA-CFAR.

    Channels: [log I, log T_ca, local CV].  Output δ such that
        T = T_ca * exp(gate * residual),  gate ∈ (0, 1).
    Homogeneous speckle (low gate) recovers classical CA-CFAR.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.residual = nn.Conv2d(hidden, 1, 1)
        self.gate = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(x)
        residual = 1.2 * torch.tanh(self.residual(h))
        gate = torch.sigmoid(self.gate(h))
        log_t_ca = x[:, 1:2]
        log_t = log_t_ca + gate * residual
        return log_t, gate, residual

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlackBoxCfarNet(nn.Module):
    """Same capacity, predicts log-threshold from the image with no CFAR prior."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, log_img: torch.Tensor) -> torch.Tensor:
        return self.net(log_img)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
