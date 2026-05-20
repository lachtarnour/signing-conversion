from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from svc.models.acoustic.softvc_f0 import SoftVCF0AcousticModel
from svc.training.checkpoint import save_training_checkpoint
from svc.training.losses import length_normalized_l1
from svc.training.optimizer import build_adamw
from svc.training.wandb import WandbLogger

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 4e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.8, 0.99)
    max_steps: int = 80000
    log_every: int = 1
    eval_every: int = 1000
    save_every: int = 1000
    checkpoints_dir: str = "checkpoints"
    device: str = "cpu"


class SoftVCTrainer:
    def __init__(
        self,
        model: SoftVCF0AcousticModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: TrainerConfig,
        logger: WandbLogger | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.optimizer = build_adamw(
            self.model.parameters(),
            learning_rate=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        self.global_step = 0
        self.best_val_loss = float("inf")
        Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: dict) -> dict:
        out = dict(batch)
        for key in ("mel", "mel_lengths", "content", "f0", "voiced", "volume", "speaker_id"):
            out[key] = out[key].to(self.device, non_blocking=True)
        return out

    def step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        batch = self._move_batch(batch)
        pred = self.model(
            content=batch["content"],
            f0=batch["f0"],
            voiced=batch["voiced"],
            volume=batch["volume"],
            speaker_id=batch["speaker_id"],
            mel_context=batch["mel"][:, :-1, :],
        )
        target = batch["mel"][:, 1 : pred.size(1) + 1, :]
        loss = length_normalized_l1(pred, target, batch["mel_lengths"])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def evaluate(self) -> float | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        total = 0.0
        count = 0
        for batch in self.val_loader:
            batch = self._move_batch(batch)
            pred = self.model(
                content=batch["content"],
                f0=batch["f0"],
                voiced=batch["voiced"],
                volume=batch["volume"],
                speaker_id=batch["speaker_id"],
                mel_context=batch["mel"][:, :-1, :],
            )
            target = batch["mel"][:, 1 : pred.size(1) + 1, :]
            total += float(length_normalized_l1(pred, target, batch["mel_lengths"]).cpu())
            count += 1
        return total / max(1, count)

    def save(self, name: str) -> Path:
        path = Path(self.cfg.checkpoints_dir) / name
        save_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            extra={"global_step": self.global_step, "best_val_loss": self.best_val_loss},
        )
        if self.logger is not None:
            self.logger.log_checkpoint(path, self.global_step)
        return path

    def _epoch_iterator(self) -> Iterable:
        while True:
            yield from self.train_loader

    def fit(self) -> None:
        start = time.time()
        running = 0.0
        running_count = 0
        LOG.info("Training params=%d device=%s", self.model.num_parameters, self.device)
        if self.logger is not None:
            self.logger.log_training_info(self.optimizer, self.cfg, self.train_loader.batch_size)
        try:
            for batch in self._epoch_iterator():
                metrics = self.step(batch)
                running += metrics["loss"]
                running_count += 1
                if self.global_step % self.cfg.log_every == 0:
                    train_loss = running / max(1, running_count)
                    elapsed = time.time() - start
                    LOG.info(
                        "step=%d loss=%.4f elapsed=%.1fs",
                        self.global_step,
                        train_loss,
                        elapsed,
                    )
                    if self.logger is not None:
                        wandb_metrics = {"train/loss": train_loss}
                        wandb_metrics.update(self.logger.gpu_metrics())
                        self.logger.log(wandb_metrics, self.global_step)
                    running = 0.0
                    running_count = 0
                if self.val_loader is not None and self.global_step % self.cfg.eval_every == 0:
                    val_loss = self.evaluate()
                    LOG.info("step=%d val_loss=%.4f", self.global_step, val_loss)
                    if self.logger is not None and val_loss is not None:
                        self.logger.log({"val/loss": val_loss}, self.global_step)
                    if val_loss is not None and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save("softvc_f0_best.pt")
                if self.global_step % self.cfg.save_every == 0:
                    self.save(f"softvc_f0_step_{self.global_step}.pt")
                if self.global_step >= self.cfg.max_steps:
                    self.save("softvc_f0_last.pt")
                    return
        finally:
            if self.logger is not None:
                self.logger.finish()
