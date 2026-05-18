from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load an audio file as mono float32 at `sample_rate`."""
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=sample_rate,
            res_type="soxr_hq",
        )
    return waveform.astype(np.float32, copy=False)


def save_audio(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a mono waveform as 16-bit PCM WAV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), waveform.astype(np.float32, copy=False), sample_rate, subtype="PCM_16")

