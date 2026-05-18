from __future__ import annotations

from collections.abc import Iterable

import torch


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float = 4e-4,
    betas: tuple[float, float] = (0.8, 0.99),
    weight_decay: float = 1e-5,
) -> torch.optim.AdamW:
    """Official SoftVC AdamW recipe."""
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
