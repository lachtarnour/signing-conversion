from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def project_root() -> Path:
    # src/svc/utils/config.py -> project root is three levels up.
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path: '~' expands, absolute stays, relative is anchored at project root."""
    p = Path(value)
    if str(p).startswith("~"):
        return p.expanduser()
    if p.is_absolute():
        return p
    return project_root() / p

