from __future__ import annotations

import torch.nn as nn


def layer_norm(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim)
