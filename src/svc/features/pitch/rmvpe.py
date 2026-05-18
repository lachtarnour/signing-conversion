from __future__ import annotations

import platform

import numpy as np

from svc.features.pitch.base import PitchExtractor, PitchTrack
from svc.utils.device import resolve_device


class RMVPEPitchExtractor(PitchExtractor):
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        device: str | None = None,
        backend: str = "auto",
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.backend = _resolve_backend(backend)
        self._rmvpe = None
        self._load(device or "auto")

    def _load(self, device: str) -> None:
        if self.backend == "mlx":
            try:
                import mlx.core as mx
                from mlx_rmvpe import RMVPE  # type: ignore[import-not-found]

                mx.set_default_device(resolve_device(device, backend="mlx"))
                self._rmvpe = RMVPE.from_pretrained()
            except Exception as exc:  # pragma: no cover - optional dependency boundary
                raise RuntimeError(
                    "RMVPE MLX backend is not available. On macOS install with "
                    "`python -m pip install -e '.[rmvpe-mlx]'`. On Linux/Colab use "
                    "`pitch.backend: onnx` and install `.[rmvpe-onnx]`."
                ) from exc
            return

        if self.backend == "onnx":
            try:
                from rmvpe_onnx import RMVPE  # type: ignore[import-not-found]

                self._rmvpe = RMVPE(device=_onnx_device(device))
            except Exception as exc:  # pragma: no cover - optional dependency boundary
                raise RuntimeError(
                    "RMVPE ONNX backend is not available. Install with "
                    "`python -m pip install -e '.[rmvpe-onnx]'`. For CUDA on Colab, "
                    "also install `onnxruntime-gpu`."
                ) from exc
            return

        raise ValueError(f"Unknown RMVPE backend: {self.backend}")

    def extract(self, waveform: np.ndarray) -> PitchTrack:
        if self._rmvpe is None:
            raise RuntimeError("RMVPE model is not initialized.")

        audio = waveform.astype(np.float32, copy=False)
        if self.backend == "mlx":
            f0 = self._rmvpe.infer_from_audio(  # pragma: no cover - optional dependency
                audio,
                sample_rate=self.sample_rate,
            )
            confidence = None
        else:
            _, f0, confidence, _ = self._rmvpe.predict(audio=audio, sr=self.sample_rate)

        f0 = np.asarray(f0, dtype=np.float32)
        f0 = _fit_to_hop(f0, audio.shape[0], self.hop_length)
        if confidence is None:
            voiced = (f0 > 0).astype(np.float32)
        else:
            confidence = _fit_to_hop(np.asarray(confidence, dtype=np.float32), audio.shape[0], self.hop_length)
            voiced = (confidence >= 0.03).astype(np.float32)
            f0 = np.where(voiced > 0, f0, 0.0).astype(np.float32)
        return PitchTrack(f0=f0, voiced=voiced)


def _resolve_backend(backend: str) -> str:
    normalized = backend.replace("-", "_").lower()
    if normalized != "auto":
        return normalized
    return "mlx" if platform.system() == "Darwin" else "onnx"


def _onnx_device(device: str) -> str | None:
    normalized = device.lower()
    if normalized == "auto":
        return None
    if normalized.startswith("cuda"):
        return normalized
    return "cpu"


def _fit_to_hop(values: np.ndarray, samples: int, hop_length: int) -> np.ndarray:
    target = max(1, int(round(samples / hop_length)))
    if values.shape[0] == target:
        return values.astype(np.float32, copy=False)
    if values.shape[0] == 0:
        return np.zeros(target, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=values.shape[0], endpoint=True)
    target_x = np.linspace(0.0, 1.0, num=target, endpoint=True)
    return np.interp(target_x, source_x, values).astype(np.float32)
