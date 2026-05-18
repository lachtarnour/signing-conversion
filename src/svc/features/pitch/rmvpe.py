from __future__ import annotations

import numpy as np

from svc.features.pitch.base import PitchExtractor, PitchTrack
from svc.utils.device import resolve_device


class RMVPEPitchExtractor(PitchExtractor):
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        device: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self._rmvpe = None
        try:
            import mlx.core as mx
            from mlx_rmvpe import RMVPE  # type: ignore[import-not-found]

            mx.set_default_device(resolve_device(device or "auto", backend="mlx"))
            self._rmvpe = RMVPE.from_pretrained()
        except Exception as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError(
                "RMVPE dependency is required. Install with `pip install -e .[rmvpe]`."
            ) from exc

    def extract(self, waveform: np.ndarray) -> PitchTrack:
        if self._rmvpe is None:
            raise RuntimeError("RMVPE model is not initialized.")
        f0 = self._rmvpe.infer_from_audio(  # pragma: no cover - optional dependency
            waveform.astype(np.float32, copy=False),
            sample_rate=self.sample_rate,
        )
        f0 = np.asarray(f0, dtype=np.float32)
        voiced = (f0 > 0).astype(np.float32)
        return PitchTrack(f0=f0, voiced=voiced)
