from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from svc.data.audio import load_audio
from svc.data.feature_store import FeatureStore
from svc.data.manifest import ManifestEntry, iter_raw_audio, write_manifest
from svc.data.slicing import Slicer
from svc.data.speaker_map import SpeakerMap
from svc.features.content.base import ContentEncoder
from svc.features.mel import LogMelSpectrogram
from svc.features.pitch.base import PitchExtractor
from svc.features.volume import extract_rms_volume
from svc.utils.device import resolve_device
from svc.utils.logging import setup_warnings
from svc.utils.seed import seed_everything


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    processed_dir: Path
    manifest_dir: Path
    encoder_name: str = "hubert_soft"
    pitch_name: str = "pyin"
    sample_rate: int = 16000
    min_seconds: float = 1.0
    extension: str = ".wav"
    slice_on_silence: bool = False
    num_workers: int = 1
    torch_threads_per_worker: int = 1
    torch_device: str = "auto"
    mlx_device: str = "auto"
    pitch_backend: str = "auto"
    pitch_hop_length: int | None = None
    seed: int = 1234


_WORKER: dict[str, Any] = {}


def compute_mel(waveform: np.ndarray, mel_module: LogMelSpectrogram) -> np.ndarray:
    device = _module_device(mel_module)
    wav = torch.from_numpy(waveform).float().to(device)
    mel = mel_module(wav).cpu().numpy()
    return mel.T.astype(np.float32)


def compute_content(waveform: np.ndarray, encoder: ContentEncoder) -> np.ndarray:
    device = _module_device(encoder)
    wav = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0).to(device)
    units = encoder.units(wav)
    return units.squeeze(0).cpu().numpy().astype(np.float32)


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in chain(module.parameters(), module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _align_array_length(array: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    if array.shape[0] == length:
        return array
    if array.shape[0] > length:
        return array[:length]
    pad_shape = (length - array.shape[0], *array.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


def _feature_stem(
    raw_dir: Path,
    split: str,
    speaker: str,
    audio_path: Path,
    chunk_idx: int,
    chunk_count: int,
) -> str:
    speaker_root = raw_dir / split / speaker
    try:
        relative = audio_path.relative_to(speaker_root)
    except ValueError:
        relative = Path(audio_path.name)

    path = Path(relative).with_suffix("")
    stem = "__".join(part.replace(" ", "_") for part in path.parts)
    if chunk_count > 1:
        stem = f"{stem}__chunk{chunk_idx:04d}"
    return stem


def extract_features(
    waveform: np.ndarray,
    content_encoder: ContentEncoder,
    pitch_extractor: PitchExtractor,
    mel_module: LogMelSpectrogram,
) -> dict[str, np.ndarray]:
    content = compute_content(waveform, content_encoder)
    length = content.shape[0]

    mel = compute_mel(waveform, mel_module)
    pitch = pitch_extractor.extract(waveform)
    f0 = _align_array_length(pitch.as_log_f0(), length)
    voiced = _align_array_length(pitch.voiced.astype(np.float32), length)
    content_hop = int(content_encoder.sample_rate / content_encoder.frame_rate_hz)
    volume = _align_array_length(extract_rms_volume(waveform, hop_length=content_hop), length)

    expected_mel = 2 * length
    if mel.shape[0] >= expected_mel:
        mel = mel[:expected_mel]
    else:
        keep = mel.shape[0] // 2
        content = content[:keep]
        f0 = f0[:keep]
        voiced = voiced[:keep]
        volume = volume[:keep]
        mel = mel[: 2 * keep]

    return {
        "content": content.astype(np.float32),
        "mel": mel.astype(np.float32),
        "f0": f0.astype(np.float32),
        "voiced": voiced.astype(np.float32),
        "volume": volume.astype(np.float32),
    }


def _speaker_ids(raw_items: list[tuple[str, str, Path]]) -> dict[str, int]:
    return {speaker: idx for idx, speaker in enumerate(sorted({item[1] for item in raw_items}))}


def _entry_sort_key(entry: ManifestEntry) -> tuple[str, str, str]:
    return entry.split, entry.speaker, entry.utt_id


def _process_audio_item(
    item: tuple[str, str, Path],
    cfg: PreprocessConfig,
    store: FeatureStore,
    speaker_ids: dict[str, int],
    content_encoder: ContentEncoder,
    pitch_extractor: PitchExtractor,
    mel_module: LogMelSpectrogram,
) -> list[ManifestEntry]:
    split, speaker, audio_path = item
    waveform = load_audio(audio_path, sample_rate=cfg.sample_rate)
    slicer = Slicer(sample_rate=cfg.sample_rate) if cfg.slice_on_silence else None
    chunks = list(slicer.slice(waveform)) if slicer is not None else [waveform]
    entries: list[ManifestEntry] = []

    for chunk_idx, chunk in enumerate(chunks):
        if chunk.shape[0] < int(cfg.sample_rate * cfg.min_seconds):
            continue
        stem = _feature_stem(cfg.raw_dir, split, speaker, audio_path, chunk_idx, len(chunks))
        features = extract_features(chunk, content_encoder, pitch_extractor, mel_module)
        paths = {
            "mel": store.save("mels", split, speaker, stem, features["mel"]),
            "content": store.save("content", split, speaker, stem, features["content"]),
            "f0": store.save("f0", split, speaker, stem, features["f0"]),
            "voiced": store.save("voiced", split, speaker, stem, features["voiced"]),
            "volume": store.save("volume", split, speaker, stem, features["volume"]),
        }
        entries.append(
            ManifestEntry(
                utt_id=f"{split}_{speaker}_{stem}",
                split=split,
                speaker=speaker,
                speaker_id=speaker_ids[speaker],
                audio_path=str(audio_path),
                mel_path=str(paths["mel"]),
                content_path=str(paths["content"]),
                f0_path=str(paths["f0"]),
                voiced_path=str(paths["voiced"]),
                volume_path=str(paths["volume"]),
            )
        )
    return entries


def _init_worker(
    cfg: PreprocessConfig,
    speaker_ids: dict[str, int],
) -> None:
    from svc.features.content import build_content_encoder
    from svc.features.pitch import build_pitch_extractor

    setup_warnings()
    seed_everything(cfg.seed + (os.getpid() % 10_000))
    torch.set_num_threads(max(1, cfg.torch_threads_per_worker))
    content_encoder = build_content_encoder(
        cfg.encoder_name,
        pretrained=True,
        device=cfg.torch_device,
    )
    pitch_extractor = build_pitch_extractor(
        cfg.pitch_name,
        sample_rate=cfg.sample_rate,
        hop_length=_pitch_hop_length(cfg, content_encoder),
        device=_pitch_device(cfg),
        backend=cfg.pitch_backend,
    )
    _WORKER.update(
        cfg=cfg,
        store=FeatureStore(cfg.processed_dir, cfg.encoder_name),
        speaker_ids=speaker_ids,
        content_encoder=content_encoder,
        pitch_extractor=pitch_extractor,
        mel_module=_build_mel_module(cfg),
    )


def _process_audio_item_in_worker(item: tuple[str, str, Path]) -> list[ManifestEntry]:
    return _process_audio_item(
        item,
        _WORKER["cfg"],
        _WORKER["store"],
        _WORKER["speaker_ids"],
        _WORKER["content_encoder"],
        _WORKER["pitch_extractor"],
        _WORKER["mel_module"],
    )


def preprocess_dataset(
    cfg: PreprocessConfig,
    content_encoder: ContentEncoder,
    pitch_extractor: PitchExtractor,
    limit: int | None = None,
) -> dict[str, list[ManifestEntry]]:
    store = FeatureStore(cfg.processed_dir, cfg.encoder_name)
    mel_module = _build_mel_module(cfg)
    manifests: dict[str, list[ManifestEntry]] = {}

    raw_items = list(iter_raw_audio(cfg.raw_dir, cfg.extension))
    if limit is not None:
        raw_items = raw_items[:limit]
    speaker_ids = _speaker_ids(raw_items)

    progress = tqdm(raw_items, desc="Processing audio", unit="file")
    for split, speaker, audio_path in progress:
        progress.set_postfix(split=split, speaker=speaker)
        entries = _process_audio_item(
            (split, speaker, audio_path),
            cfg,
            store,
            speaker_ids,
            content_encoder,
            pitch_extractor,
            mel_module,
        )
        for entry in entries:
            manifests.setdefault(entry.split, []).append(entry)

    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    SpeakerMap(speaker_ids).save(cfg.processed_dir / "speaker_map.json")
    for split, entries in manifests.items():
        entries = sorted(entries, key=_entry_sort_key)
        manifests[split] = entries
        write_manifest(cfg.manifest_dir / f"manifest_{split}.jsonl", entries)
    return manifests


def preprocess_dataset_parallel(
    cfg: PreprocessConfig,
    limit: int | None = None,
) -> dict[str, list[ManifestEntry]]:
    raw_items = list(iter_raw_audio(cfg.raw_dir, cfg.extension))
    if limit is not None:
        raw_items = raw_items[:limit]

    if not raw_items:
        cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
        SpeakerMap({}).save(cfg.processed_dir / "speaker_map.json")
        return {}

    if cfg.num_workers <= 1:
        from svc.features.content import build_content_encoder
        from svc.features.pitch import build_pitch_extractor

        seed_everything(cfg.seed)
        content_encoder = build_content_encoder(
            cfg.encoder_name,
            pretrained=True,
            device=cfg.torch_device,
        )
        pitch_extractor = build_pitch_extractor(
            cfg.pitch_name,
            sample_rate=cfg.sample_rate,
            hop_length=_pitch_hop_length(cfg, content_encoder),
            device=_pitch_device(cfg),
            backend=cfg.pitch_backend,
        )
        return preprocess_dataset(cfg, content_encoder, pitch_extractor, limit=limit)

    speaker_ids = _speaker_ids(raw_items)
    manifests: dict[str, list[ManifestEntry]] = {}
    max_workers = min(cfg.num_workers, len(raw_items), os.cpu_count() or cfg.num_workers)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cfg, speaker_ids),
    ) as executor:
        progress = tqdm(
            executor.map(_process_audio_item_in_worker, raw_items, chunksize=1),
            total=len(raw_items),
            desc=f"Processing audio ({max_workers} workers)",
            unit="file",
        )
        for entries in progress:
            for entry in entries:
                manifests.setdefault(entry.split, []).append(entry)

    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    SpeakerMap(speaker_ids).save(cfg.processed_dir / "speaker_map.json")
    for split, entries in manifests.items():
        entries = sorted(entries, key=_entry_sort_key)
        manifests[split] = entries
        write_manifest(cfg.manifest_dir / f"manifest_{split}.jsonl", entries)
    return manifests


def _pitch_hop_length(cfg: PreprocessConfig, content_encoder: ContentEncoder) -> int:
    if cfg.pitch_hop_length is not None:
        return cfg.pitch_hop_length
    return int(cfg.sample_rate / content_encoder.frame_rate_hz)


def _pitch_device(cfg: PreprocessConfig) -> str:
    if cfg.pitch_name.replace("-", "_").lower() == "rmvpe" and cfg.pitch_backend == "mlx":
        return cfg.mlx_device
    return cfg.torch_device


def _build_mel_module(cfg: PreprocessConfig) -> LogMelSpectrogram:
    return LogMelSpectrogram().to(resolve_device(cfg.torch_device, backend="torch")).eval()
