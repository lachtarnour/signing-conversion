from __future__ import annotations

from svc.features.pitch.base import PitchExtractor, PitchTrack
from svc.features.pitch.pyin import PYINPitchExtractor
from svc.features.pitch.rmvpe import RMVPEPitchExtractor

__all__ = [
    "PitchExtractor",
    "PitchTrack",
    "PYINPitchExtractor",
    "RMVPEPitchExtractor",
    "build_pitch_extractor",
]


def build_pitch_extractor(
    name: str,
    sample_rate: int = 16000,
    hop_length: int = 320,
    device: str | None = None,
    backend: str = "auto",
) -> PitchExtractor:
    normalized = name.replace("-", "_").lower()
    if normalized == "rmvpe":
        return RMVPEPitchExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            device=device,
            backend=backend,
        )
    if normalized == "pyin":
        return PYINPitchExtractor(sample_rate=sample_rate, hop_length=hop_length)
    raise ValueError(f"Unknown pitch extractor: {name}")
