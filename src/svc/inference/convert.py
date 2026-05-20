from __future__ import annotations

from pathlib import Path

from svc.data.audio import load_audio, save_audio
from svc.inference.pipeline import Stage1ConversionPipeline
from svc.inference.postprocess import normalize_peak


def convert_file(
    pipeline: Stage1ConversionPipeline,
    input_path: str | Path,
    output_path: str | Path,
    speaker_id: int = 0,
    sample_rate: int = 16000,
) -> Path:
    waveform = load_audio(input_path, sample_rate=sample_rate)
    result = pipeline.convert(waveform, speaker_id=speaker_id)
    output = Path(output_path)
    save_audio(output, normalize_peak(result.waveform), result.sample_rate)
    return output
