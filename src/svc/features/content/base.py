from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ContentEncoderConfig:
    name: str
    sample_rate: int
    frame_rate_hz: int
    output_dim: int


class ContentEncoder(nn.Module):
    config: ContentEncoderConfig

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate_hz(self) -> int:
        return self.config.frame_rate_hz

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train(self, mode: bool = True) -> ContentEncoder:
        # Encoders are feature extractors here, not trainable modules.
        super().train(False)
        return self
