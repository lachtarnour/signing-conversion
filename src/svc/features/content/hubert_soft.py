from __future__ import annotations

import torch

from svc.features.content.base import ContentEncoder, ContentEncoderConfig
from svc.utils.device import resolve_device


class HuBERTSoftEncoder(ContentEncoder):
    config = ContentEncoderConfig(
        name="hubert_soft",
        sample_rate=16000,
        frame_rate_hz=50,
        output_dim=256,
    )

    def __init__(self, pretrained: bool = True, device: str | None = None) -> None:
        super().__init__()
        self.device = resolve_device(device or "auto", backend="torch")
        self.model = torch.hub.load(
            "bshall/hubert",
            "hubert_soft",
            pretrained=pretrained,
            trust_repo=True,
            verbose=False,
        )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if wav.dim() != 3 or wav.size(1) != 1:
            raise ValueError(f"wav must be shaped (B, 1, samples), got {tuple(wav.shape)}")
        return self.model.units(wav)
