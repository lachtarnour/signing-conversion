from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from svc.data.dataset import SVCDataset, pad_collate
from svc.models.acoustic.softvc_f0 import SoftVCF0AcousticModel
from svc.training.losses import per_sample_length_normalized_l1


@torch.inference_mode()
def evaluate_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    base_dir: str | Path | None = None,
    num_clips: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    dataset = SVCDataset(
        manifest_path,
        base_dir=base_dir,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    model = SoftVCF0AcousticModel.load_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    for idx, batch in enumerate(loader):
        if idx >= num_clips:
            break
        batch = _move_batch(batch, device)
        pred = model(
            content=batch["content"],
            f0=batch["f0"],
            voiced=batch["voiced"],
            volume=batch["volume"],
            speaker_id=batch["speaker_id"],
            mel_context=batch["mel"][:, :-1, :],
            content_lengths=batch["content_lengths"],
        )
        target = batch["mel"][:, 1 : pred.size(1) + 1, :]
        loss = per_sample_length_normalized_l1(pred, target, batch["mel_lengths"])
        rows.append(
            {
                "idx": idx,
                "utt_id": batch["utt_id"][0],
                "mel_l1": float(loss[0].item()),
            }
        )
        del batch, pred, target, loss

    values = [row["mel_l1"] for row in rows]
    return {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "num_clips": len(rows),
        "mel_l1": float(np.mean(values)) if values else float("nan"),
        "per_clip": rows,
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in (
        "mel",
        "mel_lengths",
        "content",
        "content_lengths",
        "f0",
        "voiced",
        "volume",
        "speaker_id",
    ):
        out[key] = out[key].to(device, non_blocking=True)
    return out
