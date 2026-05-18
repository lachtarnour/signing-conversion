from __future__ import annotations

from svc.models.vocoders.base import Vocoder
from svc.models.vocoders.hifigan import HiFiGANVocoder

__all__ = ["Vocoder", "HiFiGANVocoder", "build_vocoder"]


def build_vocoder(name: str = "hifigan_soft", pretrained: bool = True) -> Vocoder:
    normalized = name.replace("-", "_").lower()
    if normalized in {"hifigan", "hifigan_soft"}:
        return HiFiGANVocoder(variant="soft", pretrained=pretrained)
    raise ValueError(f"Unknown vocoder: {name}")
