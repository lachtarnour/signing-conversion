from __future__ import annotations

import torch
import torch.nn as nn

from svc.models.modules.embeddings import ScalarProjection


class ConditioningFusion(nn.Module):
    """Fuse content, F0, voiced flag, volume, and speaker id at content frame rate."""

    def __init__(
        self,
        content_dim: int,
        hidden_dim: int = 256,
        num_speakers: int = 1,
        speaker_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.f0_proj = ScalarProjection(hidden_dim)
        self.volume_proj = ScalarProjection(hidden_dim)
        self.voiced_embed = nn.Embedding(2, hidden_dim)
        self.speaker_embed = nn.Embedding(num_speakers, speaker_dim)
        self.speaker_proj = nn.Linear(speaker_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        volume: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused conditioning shaped `(B, T, hidden_dim)`."""
        if content.dim() != 3:
            raise ValueError(f"content must be (B, T, D), got {tuple(content.shape)}")
        batch, frames, _ = content.shape
        f0 = _match_length(f0, frames)
        voiced = _match_length(voiced, frames)
        volume = _match_length(volume, frames)
        if speaker_id.dim() == 0:
            speaker_id = speaker_id.unsqueeze(0)
        speaker_id = speaker_id.to(device=content.device, dtype=torch.long)

        content_h = self.content_proj(content)
        f0_h = self.f0_proj(f0.to(content.device))
        volume_h = self.volume_proj(torch.log1p(volume.to(content.device).clamp_min(0.0)))
        voiced_idx = (voiced.to(content.device) > 0.5).long()
        voiced_h = self.voiced_embed(voiced_idx)
        speaker_h = self.speaker_proj(self.speaker_embed(speaker_id)).unsqueeze(1)
        speaker_h = speaker_h.expand(batch, frames, -1)
        return self.dropout(self.norm(content_h + f0_h + volume_h + voiced_h + speaker_h))


def _match_length(x: torch.Tensor, frames: int) -> torch.Tensor:
    if x.dim() == 3 and x.size(-1) == 1:
        x = x.squeeze(-1)
    if x.dim() != 2:
        raise ValueError(f"conditioning feature must be (B, T), got {tuple(x.shape)}")
    if x.size(1) == frames:
        return x
    if x.size(1) > frames:
        return x[:, :frames]
    pad = torch.zeros(x.size(0), frames - x.size(1), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)
