from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchTrack:
    f0: np.ndarray
    voiced: np.ndarray

    def as_log_f0(self) -> np.ndarray:
        out = np.zeros_like(self.f0, dtype=np.float32)
        mask = self.voiced.astype(bool) & (self.f0 > 0)
        out[mask] = np.log(self.f0[mask]).astype(np.float32)
        return out


class PitchExtractor:
    sample_rate: int
    hop_length: int

    def extract(self, waveform: np.ndarray) -> PitchTrack:
        raise NotImplementedError
