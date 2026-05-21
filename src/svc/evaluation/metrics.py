from __future__ import annotations

import numpy as np


def mel_l1(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        frames = min(pred.shape[0], target.shape[0])
        pred = pred[:frames]
        target = target[:frames]
    return float(np.mean(np.abs(pred - target)))
