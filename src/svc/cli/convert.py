from __future__ import annotations

import argparse

from svc.features.content import build_content_encoder
from svc.features.pitch import build_pitch_extractor
from svc.inference.convert import convert_file
from svc.inference.pipeline import Stage1ConversionPipeline
from svc.models.acoustic.softvc_f0 import SoftVCF0AcousticModel
from svc.models.vocoders import build_vocoder
from svc.utils.config import load_config, section
from svc.utils.device import resolve_device
from svc.utils.logging import configure_logging, setup_warnings


def main() -> int:
    setup_warnings()
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speaker-id", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    audio_cfg = section(cfg, "audio")
    encoder_cfg = section(cfg, "encoder")
    pitch_cfg = section(cfg, "pitch")
    vocoder_cfg = section(cfg, "vocoder")
    device_cfg = section(cfg, "device")

    device_name = device_cfg.get("torch", "auto")
    device = resolve_device(device_name, backend="torch")
    model = SoftVCF0AcousticModel.load_checkpoint(args.checkpoint, map_location=device)
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
    convert_file(
        pipeline,
        args.input,
        args.output,
        speaker_id=args.speaker_id,
        sample_rate=int(audio_cfg["sample_rate"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
