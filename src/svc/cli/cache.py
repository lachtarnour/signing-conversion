from __future__ import annotations

import argparse

from svc.features.content import build_content_encoder
from svc.features.pitch import build_pitch_extractor
from svc.utils.config import load_config, section
from svc.utils.logging import configure_logging, setup_warnings
from svc.utils.runtime import check_pitch_dependency, check_python_version, resolve_rmvpe_backend
from svc.utils.seed import seed_everything


def _pitch_device(pitch_name: str, pitch_backend: str, torch_device: str, mlx_device: str) -> str:
    if pitch_name.replace("-", "_").lower() == "rmvpe" and resolve_rmvpe_backend(pitch_backend) == "mlx":
        return mlx_device
    return torch_device


def main() -> int:
    check_python_version()
    setup_warnings()
    configure_logging()
    parser = argparse.ArgumentParser(description="Cache external pretrained models.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-pitch",
        action="store_true",
        help="Only cache the content encoder.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_cfg = section(cfg, "project")
    audio_cfg = section(cfg, "audio")
    encoder_cfg = section(cfg, "encoder")
    pitch_cfg = section(cfg, "pitch")
    device_cfg = section(cfg, "device")

    seed_everything(int(project_cfg.get("seed", 1234)))
    encoder_name = encoder_cfg["name"]
    pitch_name = pitch_cfg["name"]
    pitch_backend = pitch_cfg.get("backend", "auto")
    torch_device = device_cfg.get("torch", "auto")
    mlx_device = device_cfg.get("mlx", "auto")

    print(f"Caching content encoder: {encoder_name}")
    content_encoder = build_content_encoder(
        encoder_name,
        pretrained=True,
        device=torch_device,
    )

    if not args.skip_pitch:
        check_pitch_dependency(pitch_name, pitch_backend)
        hop_length = pitch_cfg.get("hop_length")
        if hop_length is None:
            hop_length = int(int(audio_cfg["sample_rate"]) / content_encoder.frame_rate_hz)
        print(f"Caching pitch extractor: {pitch_name} ({resolve_rmvpe_backend(pitch_backend)})")
        build_pitch_extractor(
            pitch_name,
            sample_rate=int(audio_cfg["sample_rate"]),
            hop_length=hop_length,
            device=_pitch_device(pitch_name, pitch_backend, torch_device, mlx_device),
            backend=pitch_backend,
        )

    print("Cache ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
