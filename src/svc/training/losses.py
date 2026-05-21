from __future__ import annotations

import torch
import torch.nn.functional as F


def length_normalized_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> torch.Tensor:
    return per_sample_length_normalized_l1(pred, target, mel_lengths).mean()


def per_sample_length_normalized_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> torch.Tensor:
    if pred.shape != target.shape:
        frames = min(pred.size(1), target.size(1))
        pred = pred[:, :frames]
        target = target[:, :frames]
        mel_lengths = mel_lengths.clamp(max=frames)

    frames = pred.size(1)
    lengths = mel_lengths.to(pred.device).clamp(min=1, max=frames)
    mask = torch.arange(frames, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).to(pred.dtype)

    loss = F.l1_loss(pred, target, reduction="none")
    return (loss * mask).sum(dim=(1, 2)) / (pred.size(-1) * lengths.to(pred.dtype))
