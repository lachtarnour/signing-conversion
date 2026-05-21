from __future__ import annotations

import torch
import torch.nn as nn


def layer_norm(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim)


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long).clamp(min=0, max=max_length)
    steps = torch.arange(max_length, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


def apply_sequence_mask(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if lengths is None:
        return x
    if x.dim() != 3:
        raise ValueError(f"x must be 3D, got {tuple(x.shape)}")
    mask = lengths_to_mask(lengths.to(x.device), x.size(1)).unsqueeze(-1)
    return x * mask.to(x.dtype)


class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, C, T), got {tuple(x.shape)}")
        if lengths is None:
            mean = x.mean(dim=2, keepdim=True)
            var = (x - mean).pow(2).mean(dim=2, keepdim=True)
            return (x - mean) * torch.rsqrt(var + self.eps)

        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.size(2))
        mask = lengths_to_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        denom = lengths.to(x.dtype).view(-1, 1, 1)
        mean = (x * mask).sum(dim=2, keepdim=True) / denom
        var = ((x - mean).pow(2) * mask).sum(dim=2, keepdim=True) / denom
        return (x - mean) * torch.rsqrt(var + self.eps) * mask
