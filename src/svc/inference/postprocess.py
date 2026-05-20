from __future__ import annotations

import numpy as np


def normalize_peak(waveform: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(waveform)))
    if max_abs < 1e-8:
        return waveform.astype(np.float32, copy=False)
    return (waveform * (peak / max_abs)).astype(np.float32, copy=False)
