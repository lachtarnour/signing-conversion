from __future__ import annotations

import librosa
import numpy as np


def extract_rms_volume(
    waveform: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 320,
) -> np.ndarray:
    wav = waveform.astype(np.float32, copy=False)
    rms = librosa.feature.rms(
        y=wav,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
    )
    return rms.squeeze(0).astype(np.float32, copy=False)
