from __future__ import annotations

import torch
import torch.nn as nn


class ScalarProjection(nn.Module):
    """Project a scalar frame-wise feature to a hidden dimension."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.proj(x)
