from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from svc.features.content import build_content_encoder
from svc.features.pitch import build_pitch_extractor
from svc.inference.convert import convert_file
from svc.inference.pipeline import Stage1ConversionPipeline
from svc.models.acoustic.softvc_f0 import SoftVCF0AcousticModel
from svc.models.vocoders import build_vocoder
from svc.utils.config import load_config, resolve_path, section
from svc.utils.device import resolve_device
from svc.utils.logging import configure_logging, setup_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert singing audio with a Stage 1 checkpoint.")
    parser.add_argument("--config", default="configs/inference/softvc_f0.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/softvc_f0_best.pt")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--speaker-id", type=int, default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--speaker-map", default="data/processed/speaker_map.json")
    return parser.parse_args()


def main() -> int:
    setup_warnings()
    configure_logging()
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    input_path = resolve_path(args.input)
    speaker_map = _load_speaker_map(args.speaker_map)
    speaker_id = _resolve_speaker_id(args.speaker_id, args.speaker, speaker_map)
    output_path = _output_path(args.output, input_path, checkpoint_path, speaker_id)
    original_path = _copy_original(input_path)

    cfg = load_config(config_path)
    audio_cfg = section(cfg, "audio")
    encoder_cfg = section(cfg, "encoder")
    pitch_cfg = section(cfg, "pitch")
    vocoder_cfg = section(cfg, "vocoder")
    device_cfg = section(cfg, "device")

    device_name = device_cfg.get("torch", "auto")
    device = resolve_device(device_name, backend="torch")
    model = SoftVCF0AcousticModel.load_checkpoint(checkpoint_path, map_location=device)
    pitch_hop = int(
        int(audio_cfg["sample_rate"]) / int(encoder_cfg.get("frame_rate_hz", 50))
    )
    pipeline = Stage1ConversionPipeline(
        content_encoder=build_content_encoder(
            encoder_cfg["name"],
            pretrained=True,
            device=device_name,
        ),
        pitch_extractor=build_pitch_extractor(
            pitch_cfg["name"],
            sample_rate=int(audio_cfg["sample_rate"]),
            hop_length=pitch_hop,
            device=device_name,
            backend=pitch_cfg.get("backend", "auto"),
        ),
        acoustic_model=model,
        vocoder=build_vocoder(
            vocoder_cfg["name"],
            pretrained=bool(vocoder_cfg.get("pretrained", True)),
        ),
        device=device_name,
    )
    output = convert_file(
        pipeline,
        input_path,
        output_path,
        speaker_id=speaker_id,
        sample_rate=int(audio_cfg["sample_rate"]),
    )
    print(f"original: {original_path}")
    print(f"generated: {output}")
    return 0


def _load_speaker_map(path: str) -> dict[str, int]:
    speaker_map_path = resolve_path(path)
    if not speaker_map_path.is_file():
        return {}
    data = json.loads(speaker_map_path.read_text(encoding="utf-8"))
    return {str(name): int(idx) for name, idx in data.items()}


def _resolve_speaker_id(
    speaker_id: int | None,
    speaker: str | None,
    speaker_map: dict[str, int],
) -> int:
    if speaker_id is not None and speaker is not None:
        raise ValueError("Use either --speaker-id or --speaker, not both.")
    if speaker_id is not None:
        return speaker_id
    if speaker is not None:
        return _speaker_name_to_id(speaker, speaker_map)
    if speaker_map:
        _print_speakers(speaker_map)
        choice = input("Choose target singer: ").strip()
        return _speaker_choice_to_id(choice, speaker_map)
    choice = input("Choose target singer id: ").strip()
    return int(choice)


def _speaker_name_to_id(speaker: str, speaker_map: dict[str, int]) -> int:
    if speaker not in speaker_map:
        choices = ", ".join(sorted(speaker_map))
        raise ValueError(f"Unknown speaker '{speaker}'. Available speakers: {choices}")
    return speaker_map[speaker]


def _speaker_choice_to_id(choice: str, speaker_map: dict[str, int]) -> int:
    if choice in speaker_map:
        return speaker_map[choice]
    try:
        speaker_id = int(choice)
    except ValueError as exc:
        choices = ", ".join(sorted(speaker_map))
        raise ValueError(f"Unknown speaker '{choice}'. Available speakers: {choices}") from exc
    if speaker_id not in set(speaker_map.values()):
        choices = ", ".join(f"{idx}:{name}" for name, idx in sorted(speaker_map.items()))
        raise ValueError(f"Unknown speaker id '{speaker_id}'. Available speakers: {choices}")
    return speaker_id


def _print_speakers(speaker_map: dict[str, int]) -> None:
    print("Available target singers:")
    for name, speaker_id in sorted(speaker_map.items(), key=lambda item: item[1]):
        print(f"  {speaker_id:2d}  {name}")


def _output_path(
    output: str | None,
    input_path: Path,
    checkpoint_path: Path,
    speaker_id: int,
) -> Path:
    if output is not None:
        return resolve_path(output)
    name = f"{input_path.stem}_spk{speaker_id}_{checkpoint_path.stem}.wav"
    return resolve_path(Path("outputs/inference") / name)


def _copy_original(input_path: Path) -> Path:
    output = resolve_path(Path("outputs/original") / input_path.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() != output.resolve():
        shutil.copy2(input_path, output)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
