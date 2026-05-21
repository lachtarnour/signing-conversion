from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from svc.models.acoustic.decoder import AutoregressiveMelDecoder
from svc.models.acoustic.encoder import ConditionedEncoder


class SoftVCF0AcousticModel(nn.Module):
    """SoftVC acoustic model extended with F0, voiced, volume, and speaker conditioning."""

    def __init__(
        self,
        content_dim: int = 256,
        num_speakers: int = 1,
        speaker_dim: int = 256,
        upsample: bool = True,
        mel_dim: int = 128,
    ) -> None:
        super().__init__()
        self.config = {
            "content_dim": content_dim,
            "num_speakers": num_speakers,
            "speaker_dim": speaker_dim,
            "upsample": upsample,
            "mel_dim": mel_dim,
        }
        self.encoder = ConditionedEncoder(
            content_dim=content_dim,
            num_speakers=num_speakers,
            speaker_dim=speaker_dim,
            upsample=upsample,
        )
        self.decoder = AutoregressiveMelDecoder(mel_dim=mel_dim)

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        volume: torch.Tensor,
        speaker_id: torch.Tensor,
        mel_context: torch.Tensor,
        content_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encoder(content, f0, voiced, volume, speaker_id, lengths=content_lengths)
        if context.size(1) != mel_context.size(1):
            frames = min(context.size(1), mel_context.size(1))
            context = context[:, :frames]
            mel_context = mel_context[:, :frames]
        return self.decoder(context, mel_context)

    @torch.inference_mode()
    def generate(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        volume: torch.Tensor,
        speaker_id: torch.Tensor,
        content_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encoder(content, f0, voiced, volume, speaker_id, lengths=content_lengths)
        return self.decoder.generate(context)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def save_checkpoint(self, path: str, extra: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "config": self.config,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> SoftVCF0AcousticModel:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model
