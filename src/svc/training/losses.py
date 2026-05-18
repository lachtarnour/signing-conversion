from __future__ import annotations

import torch
import torch.nn.functional as F


def length_normalized_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> torch.Tensor:
    """SoftVC length-normalized L1 mel loss."""
    if pred.shape != target.shape:
        frames = min(pred.size(1), target.size(1))
        pred = pred[:, :frames]
        target = target[:, :frames]
        mel_lengths = mel_lengths.clamp(max=frames)
    loss = F.l1_loss(pred, target, reduction="none")
    loss = loss.sum(dim=(1, 2)) / (pred.size(-1) * mel_lengths.float().clamp_min(1.0))
    return loss.mean()
