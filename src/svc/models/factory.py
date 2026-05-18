from __future__ import annotations

from typing import Any

from svc.models.acoustic.softvc_f0 import SoftVCF0AcousticModel


def build_acoustic_model(cfg: dict[str, Any]) -> SoftVCF0AcousticModel:
    """Build the configured acoustic model."""
    name = cfg.get("name", "softvc_f0")
    if name != "softvc_f0":
        raise ValueError(f"Unsupported acoustic model: {name}")
    return SoftVCF0AcousticModel(
        content_dim=int(cfg.get("content_dim", 256)),
        num_speakers=int(cfg.get("num_speakers", 1)),
        speaker_dim=int(cfg.get("speaker_dim", 256)),
        upsample=bool(cfg.get("upsample", True)),
        mel_dim=int(cfg.get("mel_dim", 128)),
    )
