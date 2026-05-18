from __future__ import annotations

import librosa
import numpy as np

from svc.features.pitch.base import PitchExtractor, PitchTrack


class PYINPitchExtractor(PitchExtractor):
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        fmin: float | None = None,
        fmax: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = float(librosa.note_to_hz("C2")) if fmin is None else fmin
        self.fmax = float(librosa.note_to_hz("C7")) if fmax is None else fmax

    def extract(self, waveform: np.ndarray) -> PitchTrack:
        f0, voiced, _ = librosa.pyin(
            waveform.astype(np.float32, copy=False),
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            frame_length=1024,
            center=True,
        )
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        return PitchTrack(f0=f0, voiced=voiced.astype(np.float32))
