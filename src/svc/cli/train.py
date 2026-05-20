from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from svc.data.dataset import SVCDataset, pad_collate
from svc.models.factory import build_acoustic_model
from svc.training.trainer import SoftVCTrainer, TrainerConfig
from svc.training.wandb import build_wandb_logger
from svc.utils.config import load_config, resolve_path, section
from svc.utils.device import resolve_device
from svc.utils.logging import configure_logging, setup_warnings
from svc.utils.seed import seed_everything


def _num_speakers(processed_dir: str) -> int:
    path = Path(processed_dir) / "speaker_map.json"
    if not path.is_file():
        return 1
    return max(1, len(json.loads(path.read_text(encoding="utf-8"))))


def build_trainer(cfg: dict, max_steps: int | None = None) -> SoftVCTrainer:
    project_cfg = section(cfg, "project")
    paths_cfg = section(cfg, "paths")
    train_cfg = section(cfg, "train")
    model_cfg = section(cfg, "model")
    device_cfg = section(cfg, "device")
    logging_cfg = section(cfg, "logging")
    base_dir = resolve_path(paths_cfg.get("base_dir", "."))

    seed_everything(int(project_cfg.get("seed", 1234)))
    train_manifest = resolve_path(paths_cfg["train_manifest"], base_dir=base_dir)
    dev_manifest = paths_cfg.get("dev_manifest")
    dev_manifest_path = resolve_path(dev_manifest, base_dir=base_dir) if dev_manifest else None

    train_ds = SVCDataset(train_manifest, base_dir=base_dir)
    val_ds = (
        SVCDataset(dev_manifest_path, base_dir=base_dir)
        if dev_manifest_path is not None and dev_manifest_path.is_file()
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=pad_collate,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            collate_fn=pad_collate,
        )
        if val_ds is not None
        else None
    )
    num_speakers = int(
        model_cfg.get(
            "num_speakers",
            _num_speakers(str(resolve_path(paths_cfg["processed_dir"], base_dir=base_dir))),
        )
    )
    model = build_acoustic_model({**model_cfg, "num_speakers": num_speakers})
    trainer_cfg = TrainerConfig(
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=tuple(train_cfg.get("betas", (0.8, 0.99))),
        max_steps=int(max_steps if max_steps is not None else train_cfg["max_steps"]),
        log_every=int(train_cfg["log_every"]),
        eval_every=int(train_cfg["eval_every"]),
        save_every=int(train_cfg["save_every"]),
        checkpoints_dir=str(resolve_path(paths_cfg["checkpoints_dir"])),
        device=str(resolve_device(device_cfg.get("torch", "auto"), backend="torch")),
    )
    logger = build_wandb_logger(logging_cfg, run_config=cfg)
    return SoftVCTrainer(model, train_loader, val_loader, trainer_cfg, logger=logger)


def main() -> int:
    setup_warnings()
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    trainer = build_trainer(load_config(args.config), max_steps=args.max_steps)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
