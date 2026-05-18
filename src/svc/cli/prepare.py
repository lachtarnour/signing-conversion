from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from svc.data.preprocess import PreprocessConfig, preprocess_dataset_parallel
from svc.utils.config import load_config, resolve_path
from svc.utils.logging import configure_logging, setup_warnings


def _check_python_version() -> None:
    if sys.version_info >= (3, 11):
        return
    raise SystemExit(
        "This project requires Python 3.11+.\n"
        f"Current Python: {sys.version.split()[0]}\n"
        "Run: source .venv/bin/activate"
    )


def _check_pitch_dependency(pitch_name: str) -> None:
    if pitch_name.replace("-", "_").lower() != "rmvpe":
        return
    if importlib.util.find_spec("mlx_rmvpe") is not None:
        return
    raise SystemExit(
        "RMVPE is selected but mlx_rmvpe is not installed.\n"
        "Use Python 3.11+ and run: python -m pip install -e '.[rmvpe]'\n"
        "Or set pitch.name: pyin in the config."
    )


def main() -> int:
    _check_python_version()
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
    encoder_name = cfg["encoder"]["name"]
    pitch_name = cfg["pitch"]["name"]
    _check_pitch_dependency(pitch_name)
    sample_rate = int(cfg["audio"]["sample_rate"])
    device_cfg = cfg.get("device", {})
    preprocess_cfg = cfg.get("preprocess", {})
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(preprocess_cfg.get("num_workers", 1))
    )
    preprocess_cfg = PreprocessConfig(
        raw_dir=resolve_path(cfg["paths"]["raw_dir"]),
        processed_dir=resolve_path(cfg["paths"]["processed_dir"]),
        manifest_dir=resolve_path(cfg["paths"]["manifest_dir"]),
        encoder_name=encoder_name,
        sample_rate=sample_rate,
        min_seconds=float(preprocess_cfg.get("min_seconds", 1.0)),
        slice_on_silence=bool(preprocess_cfg.get("slice_on_silence", False)),
        num_workers=max(1, num_workers),
        torch_threads_per_worker=int(preprocess_cfg.get("torch_threads_per_worker", 1)),
        torch_device=device_cfg.get("torch", "auto"),
        mlx_device=device_cfg.get("mlx", "auto"),
    )
    preprocess_dataset_parallel(
        preprocess_cfg,
        encoder_name=encoder_name,
        pitch_name=pitch_name,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
