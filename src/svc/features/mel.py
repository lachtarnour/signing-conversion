from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms


class LogMelSpectrogram(nn.Module):
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 160
    n_mels: int = 128
    pad: int = (1024 - 160) // 2

    def __init__(self) -> None:
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze = True
        wav = F.pad(wav, (self.pad, self.pad), mode="reflect")
        mel = self.melspectrogram(wav)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.squeeze(0) if squeeze else log_mel
