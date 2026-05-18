from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


def _frame_rms(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if waveform.shape[0] < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.shape[0]))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_length)[::hop_length]
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


@dataclass(frozen=True)
class Slicer:
    sample_rate: int = 16000
    threshold_db: float = -42.0
    min_length_ms: int = 1500
    min_interval_ms: int = 400
    hop_ms: int = 15
    max_silence_kept_ms: int = 500

    def slice(self, waveform: np.ndarray) -> Iterator[np.ndarray]:
        """Yield non-silent chunks from `waveform`."""
        hop = max(1, int(self.sample_rate * self.hop_ms / 1000))
        frame = max(hop, int(self.sample_rate * 0.04))
        min_interval = max(1, int(self.min_interval_ms / self.hop_ms))
        min_samples = int(self.sample_rate * self.min_length_ms / 1000)
        keep_samples = int(self.sample_rate * self.max_silence_kept_ms / 1000)

        rms = _frame_rms(waveform, frame, hop)
        db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        silent = db < self.threshold_db

        starts: list[int] = []
        ends: list[int] = []
        in_speech = False
        start = 0
        silence_run = 0

        for idx, is_silent in enumerate(silent):
            if not in_speech and not is_silent:
                in_speech = True
                start = max(0, idx * hop - keep_samples)
                silence_run = 0
            elif in_speech and is_silent:
                silence_run += 1
                if silence_run >= min_interval:
                    end = min(waveform.shape[0], idx * hop + keep_samples)
                    if end - start >= min_samples:
                        starts.append(start)
                        ends.append(end)
                    in_speech = False
                    silence_run = 0
            elif in_speech:
                silence_run = 0

        if in_speech:
            end = waveform.shape[0]
            if end - start >= min_samples:
                starts.append(start)
                ends.append(end)

        if not starts:
            yield waveform.astype(np.float32, copy=False)
            return

        for start, end in zip(starts, ends, strict=True):
            yield waveform[start:end].astype(np.float32, copy=False)
