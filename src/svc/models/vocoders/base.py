from __future__ import annotations

import torch
import torch.nn as nn


class Vocoder(nn.Module):
    sample_rate: int = 16000

    @torch.inference_mode()
    def synthesize(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        raise NotImplementedError
