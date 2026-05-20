from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from svc.data.manifest import read_manifest


class SVCDataset(Dataset):
    def __init__(self, manifest_path: str | Path, base_dir: str | Path | None = None) -> None:
        self.entries = read_manifest(manifest_path)
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir is not None else None
        if not self.entries:
            raise ValueError(f"No samples in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _load(path: str) -> np.ndarray:
        return np.load(path)

    def _path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute() or self.base_dir is None:
            return p
        return self.base_dir / p

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        mel = self._load(str(self._path(entry.mel_path)))
        content = self._load(str(self._path(entry.content_path)))
        f0 = self._load(str(self._path(entry.f0_path)))
        voiced = self._load(str(self._path(entry.voiced_path)))
        volume = self._load(str(self._path(entry.volume_path)))

        length = min(content.shape[0], f0.shape[0], voiced.shape[0], volume.shape[0])
        mel_length = min(mel.shape[0], 2 * length)
        length = mel_length // 2
        mel_length = 2 * length

        mel_t = torch.from_numpy(mel[:mel_length].astype(np.float32))
        mel_t = F.pad(mel_t, (0, 0, 1, 0))

        return {
            "utt_id": entry.utt_id,
            "mel": mel_t,
            "mel_length": mel_length,
            "content": torch.from_numpy(content[:length].astype(np.float32)),
            "content_length": length,
            "f0": torch.from_numpy(f0[:length].astype(np.float32)),
            "voiced": torch.from_numpy(voiced[:length].astype(np.float32)),
            "volume": torch.from_numpy(volume[:length].astype(np.float32)),
            "speaker_id": int(entry.speaker_id),
        }


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    mels = [item["mel"] for item in batch]
    content = [item["content"] for item in batch]
    f0 = [item["f0"] for item in batch]
    voiced = [item["voiced"] for item in batch]
    volume = [item["volume"] for item in batch]

    return {
        "utt_id": [item["utt_id"] for item in batch],
        "mel": pad_sequence(mels, batch_first=True),
        "mel_lengths": torch.tensor([item["mel_length"] for item in batch], dtype=torch.long),
        "content": pad_sequence(content, batch_first=True),
        "content_lengths": torch.tensor(
            [item["content_length"] for item in batch],
            dtype=torch.long,
        ),
        "f0": pad_sequence(f0, batch_first=True),
        "voiced": pad_sequence(voiced, batch_first=True),
        "volume": pad_sequence(volume, batch_first=True),
        "speaker_id": torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long),
    }
