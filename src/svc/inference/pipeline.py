from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from svc.features.alignment import align_tensor_length
from svc.features.content.base import ContentEncoder
from svc.features.pitch.base import PitchExtractor
from svc.features.volume import extract_rms_volume
from svc.models.acoustic.softvc_f0 import SoftVCF0AcousticModel
from svc.models.vocoders.base import Vocoder
from svc.utils.device import resolve_device


@dataclass(frozen=True)
class ConversionResult:
    waveform: np.ndarray
    sample_rate: int
    mel: np.ndarray


class Stage1ConversionPipeline:
    def __init__(
        self,
        content_encoder: ContentEncoder,
        pitch_extractor: PitchExtractor,
        acoustic_model: SoftVCF0AcousticModel,
        vocoder: Vocoder,
        device: str = "auto",
    ) -> None:
        self.content_encoder = content_encoder
        self.pitch_extractor = pitch_extractor
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder
        self.device = resolve_device(device, backend="torch")
        self.content_encoder.to(self.device)
        self.acoustic_model.to(self.device).eval()
        self.vocoder.to(self.device).eval()

    @torch.inference_mode()
    def convert(self, waveform: np.ndarray, speaker_id: int = 0) -> ConversionResult:
        wav_t = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0).to(self.device)
        content = self.content_encoder.units(wav_t)

        frames = content.size(1)
        pitch = self.pitch_extractor.extract(waveform)
        f0 = pitch.as_log_f0()
        voiced = pitch.voiced.astype(np.float32)
        content_hop = int(self.content_encoder.sample_rate / self.content_encoder.frame_rate_hz)
        volume = extract_rms_volume(waveform, hop_length=content_hop)

        f0_t = align_tensor_length(
            torch.from_numpy(f0).float().unsqueeze(0).to(self.device),
            frames,
        )
        voiced_t = align_tensor_length(
            torch.from_numpy(voiced).float().unsqueeze(0).to(self.device),
            frames,
        )
        volume_t = align_tensor_length(
            torch.from_numpy(volume).float().unsqueeze(0).to(self.device),
            frames,
        )
        speaker_t = torch.tensor([speaker_id], dtype=torch.long, device=self.device)
        content_lengths = torch.tensor([frames], dtype=torch.long, device=self.device)
        mel = self.acoustic_model.generate(
            content,
            f0_t,
            voiced_t,
            volume_t,
            speaker_t,
            content_lengths=content_lengths,
        )
        audio, sample_rate = self.vocoder.synthesize(mel.transpose(1, 2), f0=f0_t)
        audio_np = audio.squeeze(0).detach().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np[0]
        return ConversionResult(
            waveform=audio_np.astype(np.float32, copy=False),
            sample_rate=sample_rate,
            mel=mel.squeeze(0).detach().cpu().numpy(),
        )
