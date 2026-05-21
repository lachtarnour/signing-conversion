from __future__ import annotations

import torch
import torch.nn as nn

from svc.models.acoustic.conditioning import ConditioningFusion
from svc.models.modules.norm import MaskedInstanceNorm1d, apply_sequence_mask
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
            MaskedInstanceNorm1d(),
            nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            MaskedInstanceNorm1d(),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            MaskedInstanceNorm1d(),
        )
        self.upsample = upsample

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        volume: torch.Tensor,
        speaker_id: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conditioning(content, f0, voiced, volume, speaker_id)
        x = apply_sequence_mask(x, lengths)
        x = self.prenet(x)
        x = apply_sequence_mask(x, lengths)
        x = x.transpose(1, 2)

        x = self.convs[0](x)
        x = self.convs[1](x)
        x = self.convs[2](x, lengths)

        x = self.convs[3](x)
        if lengths is not None and self.upsample:
            lengths = lengths * 2
            x = apply_sequence_mask(x.transpose(1, 2), lengths).transpose(1, 2)
        x = self.convs[4](x)
        x = self.convs[5](x)
        x = self.convs[6](x, lengths)

        x = self.convs[7](x)
        x = self.convs[8](x)
        x = self.convs[9](x, lengths)
        return x.transpose(1, 2)
