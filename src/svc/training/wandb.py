from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "signing-conversion"
    entity: str | None = None
    run_name: str | None = None
    mode: str | None = None
    tags: tuple[str, ...] = ()
    log_checkpoints: bool = False


class WandbLogger:
    def __init__(self, cfg: WandbConfig, config: dict[str, Any]) -> None:
        self.cfg = cfg
        self._wandb = None
        self._run = None
        if not cfg.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "Weights & Biases logging is enabled but wandb is not installed. "
                "Install it with `python -m pip install -e '.[tracking]'` "
                "or set logging.wandb.enabled: false."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            mode=cfg.mode,
            tags=list(cfg.tags),
            config=config,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def watch(self, model: Any) -> None:
        if self._wandb is not None:
            self._wandb.watch(model, log="gradients", log_freq=100)

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_checkpoint(self, path: str | Path, step: int) -> None:
        if self._wandb is None or not self.cfg.log_checkpoints:
            return
        artifact = self._wandb.Artifact(Path(path).stem, type="checkpoint")
        artifact.add_file(str(path))
        self._wandb.log_artifact(artifact, aliases=[f"step-{step}"])

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def build_wandb_logger(cfg: dict[str, Any], run_config: dict[str, Any]) -> WandbLogger:
    wandb_cfg = cfg.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        raise TypeError("logging.wandb must be a mapping.")
    return WandbLogger(
        WandbConfig(
            enabled=bool(wandb_cfg.get("enabled", False)),
            project=str(wandb_cfg.get("project", "signing-conversion")),
            entity=wandb_cfg.get("entity"),
            run_name=wandb_cfg.get("run_name"),
            mode=wandb_cfg.get("mode"),
            tags=tuple(wandb_cfg.get("tags", ())),
            log_checkpoints=bool(wandb_cfg.get("log_checkpoints", False)),
        ),
        config=run_config,
    )
