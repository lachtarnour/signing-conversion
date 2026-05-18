from __future__ import annotations

import torch


def align_tensor_length(tensor: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
    if tensor.size(1) == length:
        return tensor
    if tensor.size(1) > length:
        return tensor[:, :length]
    pad_shape = (tensor.size(0), length - tensor.size(1), *tensor.shape[2:])
    pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1)
