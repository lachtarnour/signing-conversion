from __future__ import annotations

import torch
import torch.nn as nn

from svc.models.acoustic.conditioning import ConditioningFusion
from svc.models.modules.prenet import PreNet


class ConditionedEncoder(nn.Module):
    """Map conditioned content frames to 512-dim context at 2x frame rate."""

    def __init__(
        self,
        content_dim: int = 256,
        hidden_dim: int = 256,
        num_speakers: int = 1,
        speaker_dim: int = 256,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        self.conditioning = ConditioningFusion(
            content_dim=content_dim,
            hidden_dim=hidden_dim,
            num_speakers=num_speakers,
            speaker_dim=speaker_dim,
        )
        self.prenet = PreNet(hidden_dim, 256, 256)
        self.convs = nn.Sequential(
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
        )

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        volume: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conditioning(content, f0, voiced, volume, speaker_id)
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)
