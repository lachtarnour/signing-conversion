from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    return value


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(value)
    if str(p).startswith("~"):
        return p.expanduser()
    if p.is_absolute():
        return p
    if base_dir is not None:
        return resolve_path(base_dir) / p
    return project_root() / p
