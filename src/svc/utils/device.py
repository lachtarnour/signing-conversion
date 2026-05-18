from __future__ import annotations

from typing import Any

import torch


def resolve_device(name: str = "auto", backend: str = "torch") -> Any:
    if backend == "torch":
        return _resolve_torch_device(name)
    if backend == "mlx":
        return _resolve_mlx_device(name)
    raise ValueError(f"Unknown device backend: {backend}")


def _resolve_torch_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_mlx_device(name: str) -> Any:
    import mlx.core as mx

    normalized = name.lower()
    if normalized in ("auto", "gpu", "mps"):
        return mx.gpu
    if normalized == "cpu":
        return mx.cpu
    raise ValueError("MLX device must be one of: auto, gpu, mps, cpu")
