from __future__ import annotations

import torch
import torch.nn as nn

from svc.models.modules.prenet import PreNet


class AutoregressiveMelDecoder(nn.Module):
    """Three-layer residual LSTM decoder from the public SoftVC acoustic model."""

    def __init__(self, mel_dim: int = 128) -> None:
        super().__init__()
        self.mel_dim = mel_dim
        self.prenet = PreNet(mel_dim, 256, 256)
        self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
        self.lstm2 = nn.LSTM(768, 768, batch_first=True)
        self.lstm3 = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, mel_dim, bias=False)

    def forward(self, context: torch.Tensor, mel_context: torch.Tensor) -> torch.Tensor:
        mels = self.prenet(mel_context)
        x, _ = self.lstm1(torch.cat((context, mels), dim=-1))
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, context: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(context.size(0), self.mel_dim, device=context.device)
        h1 = torch.zeros(1, context.size(0), 768, device=context.device)
        c1 = torch.zeros(1, context.size(0), 768, device=context.device)
        h2 = torch.zeros(1, context.size(0), 768, device=context.device)
        c2 = torch.zeros(1, context.size(0), 768, device=context.device)
        h3 = torch.zeros(1, context.size(0), 768, device=context.device)
        c3 = torch.zeros(1, context.size(0), 768, device=context.device)

        mel = []
        for frame in torch.unbind(context, dim=1):
            m = self.prenet(m)
            x = torch.cat((frame, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)
