from __future__ import annotations

from typing import Literal

import torch

from svc.models.vocoders.base import Vocoder

Variant = Literal["soft", "discrete", "base"]


class HiFiGANVocoder(Vocoder):
    """SoftVC-compatible HiFi-GAN wrapper."""

    def __init__(self, variant: Variant = "soft", pretrained: bool = True) -> None:
        super().__init__()
        entry_point = {
            "soft": "hifigan_hubert_soft",
            "discrete": "hifigan_hubert_discrete",
            "base": "hifigan",
        }[variant]
        self.model = torch.hub.load(
            "bshall/hifigan",
            entry_point,
            pretrained=pretrained,
            progress=False,
            map_location="cpu",
            trust_repo=True,
            verbose=False,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.variant = variant

    @torch.inference_mode()
    def synthesize(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.dim() != 3:
            raise ValueError(f"mel must be (B, 128, T) or (128, T), got {tuple(mel.shape)}")
        wav = self.model(mel)
        return wav, int(getattr(self.model, "sample_rate", 16000))

    def train(self, mode: bool = True) -> HiFiGANVocoder:
        super().train(False)
        self.model.eval()
        return self
