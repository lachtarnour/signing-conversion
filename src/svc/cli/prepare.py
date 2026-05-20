from __future__ import annotations

import argparse

from svc.data.preprocess import PreprocessConfig, preprocess_dataset_parallel
from svc.utils.config import load_config, resolve_path, section
from svc.utils.logging import configure_logging, setup_warnings
from svc.utils.runtime import check_pitch_dependency, check_python_version
from svc.utils.seed import seed_everything


def main() -> int:
    check_python_version()
    setup_warnings()
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for preprocessing. Overrides preprocess.num_workers.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_cfg = section(cfg, "project")
    paths_cfg = section(cfg, "paths")
    audio_cfg = section(cfg, "audio")
    encoder_cfg = section(cfg, "encoder")
    pitch_cfg = section(cfg, "pitch")
    device_cfg = section(cfg, "device")
    options_cfg = section(cfg, "preprocess")
    base_dir = resolve_path(paths_cfg.get("base_dir", "."))

    seed = int(project_cfg.get("seed", 1234))
    seed_everything(seed)
    encoder_name = encoder_cfg["name"]
    pitch_name = pitch_cfg["name"]
    pitch_backend = pitch_cfg.get("backend", "auto")
    check_pitch_dependency(pitch_name, pitch_backend)
    sample_rate = int(audio_cfg["sample_rate"])
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(options_cfg.get("num_workers", 1))
    )
    preprocess_cfg = PreprocessConfig(
        base_dir=base_dir,
        raw_dir=resolve_path(paths_cfg["raw_dir"], base_dir=base_dir),
        processed_dir=resolve_path(paths_cfg["processed_dir"], base_dir=base_dir),
        manifest_dir=resolve_path(paths_cfg["manifest_dir"], base_dir=base_dir),
        encoder_name=encoder_name,
        pitch_name=pitch_name,
        sample_rate=sample_rate,
        min_seconds=float(options_cfg.get("min_seconds", 1.0)),
        extension=str(options_cfg.get("extension", ".wav")),
        slice_on_silence=bool(options_cfg.get("slice_on_silence", False)),
        num_workers=max(1, num_workers),
        torch_threads_per_worker=int(options_cfg.get("torch_threads_per_worker", 1)),
        torch_device=device_cfg.get("torch", "auto"),
        mlx_device=device_cfg.get("mlx", "auto"),
        pitch_backend=pitch_backend,
        pitch_hop_length=pitch_cfg.get("hop_length"),
        seed=seed,
    )
    preprocess_dataset_parallel(
        preprocess_cfg,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
