from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureStore:
    def __init__(self, root: str | Path, encoder_name: str = "hubert_soft") -> None:
        self.root = Path(root)
        self.encoder_name = encoder_name

    def path(self, kind: str, split: str, speaker: str, stem: str) -> Path:
        if kind == "content":
            return self.root / "content" / self.encoder_name / split / speaker / f"{stem}.npy"
        return self.root / kind / split / speaker / f"{stem}.npy"

    def save(self, kind: str, split: str, speaker: str, stem: str, array: np.ndarray) -> Path:
        path = self.path(kind, split, speaker, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        return path

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        return np.load(path)
