from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
import json

def iter_raw_audio(
        raw_root:str | Path,
        extension:str = ".wav",
    ) -> Iterator[tuple[str, str, Path]]:
    """Yield ('split','speaker','path') from raw structrue '<split>/<speaker>/*/<file_name.wav>"""
    root = Path(raw_root)
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for speaker_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for audio_path in sorted(speaker_dir.rglob(f"*{extension}")):
                yield split_dir.name, speaker_dir.name, audio_path


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    split: str 
    speaker: str
    speaker_id: int
    audio_path: Path
    audio_type: str = "isolated_vocal"
    instrumental_path: Path | None = None   

    @classmethod
    def from_json(cls,line:str) -> "ManifestEntry":
        payload = json.loads(line)
        payload.setdefault("audio_type","isolated_vocal")
        payload.setdefault("instrumental_path",None)
        return cls(**payload)
    
    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "split": self.split,
            "speaker": self.speaker,
            "speaker_id": self.speaker_id,
            "audio_path": str(self.audio_path),
            "audio_type": self.audio_type,
            "instrumental_path": str(self.instrumental_path) if self.instrumental_path else None
        })
    
def read_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read a JSONL manifest."""
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
    """Write a JSONL manifest."""
    manifest = Path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(entry.to_json() for entry in entries)
    manifest.write_text(text + ("\n" if text else ""), encoding="utf-8")

