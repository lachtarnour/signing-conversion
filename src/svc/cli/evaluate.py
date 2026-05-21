from __future__ import annotations

import argparse
import json
from pathlib import Path

from svc.evaluation import evaluate_manifest
from svc.utils.config import load_config, resolve_path, section
from svc.utils.device import resolve_device
from svc.utils.logging import configure_logging, setup_warnings


def main() -> int:
    setup_warnings()
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--num-clips", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths_cfg = section(cfg, "paths")
    device_cfg = section(cfg, "device")
    base_dir = resolve_path(paths_cfg.get("base_dir", "."))
    device = resolve_device(device_cfg.get("torch", "auto"), backend="torch")

    manifest_path = _manifest_path(args.manifest, args.split, paths_cfg, base_dir)
    result = evaluate_manifest(
        manifest_path=manifest_path,
        checkpoint_path=resolve_path(args.checkpoint),
        base_dir=base_dir,
        num_clips=args.num_clips,
        device=device,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _manifest_path(
    manifest: str | None,
    split: str,
    paths_cfg: dict,
    base_dir: Path,
) -> Path:
    if manifest is not None:
        return resolve_path(manifest, base_dir=base_dir)
    key = f"{split}_manifest"
    default = f"manifests/manifest_{split}.jsonl"
    return resolve_path(paths_cfg.get(key, default), base_dir=base_dir)


if __name__ == "__main__":
    raise SystemExit(main())
