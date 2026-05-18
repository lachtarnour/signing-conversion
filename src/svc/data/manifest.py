from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    split: str
    speaker: str
    speaker_id: int
    audio_path: str
    mel_path: str
    content_path: str
    f0_path: str
    voiced_path: str
    volume_path: str

    @classmethod
    def from_json(cls, line: str) -> ManifestEntry:
        payload = json.loads(line)
        fields = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in fields})

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    entries = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(ManifestEntry.from_json(line))
    return entries


def write_manifest(path: str | Path, entries: list[ManifestEntry]) -> None:
    manifest = Path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(entry.to_json() for entry in entries)
    manifest.write_text(text + ("\n" if text else ""), encoding="utf-8")


def iter_raw_audio(
    raw_root: str | Path,
    extension: str = ".wav",
) -> Iterator[tuple[str, str, Path]]:
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing raw audio directory: {root}")
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for speaker_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for audio_path in sorted(speaker_dir.rglob(f"*{extension}")):
                yield split_dir.name, speaker_dir.name, audio_path
