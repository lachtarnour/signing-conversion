from __future__ import annotations

from pathlib import Path

from svc.inference.convert import convert_file
from svc.inference.pipeline import Stage1ConversionPipeline


def batch_convert(
    pipeline: Stage1ConversionPipeline,
    inputs: list[str | Path],
    output_dir: str | Path,
    speaker_id: int = 0,
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for input_path in inputs:
        input_path = Path(input_path)
        outputs.append(
            convert_file(
                pipeline,
                input_path,
                out_dir / f"{input_path.stem}_converted.wav",
                speaker_id=speaker_id,
            )
        )
    return outputs
