from __future__ import annotations

import json
from pathlib import Path


class SpeakerMap:
    def __init__(self, speakers: dict[str, int] | None = None) -> None:
        self.speakers = dict(speakers or {})

    def get_or_add(self, speaker: str) -> int:
        if speaker not in self.speakers:
            self.speakers[speaker] = len(self.speakers)
        return self.speakers[speaker]

    def __len__(self) -> int:
        return len(self.speakers)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.speakers, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SpeakerMap:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
