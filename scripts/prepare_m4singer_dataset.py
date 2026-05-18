from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from svc.utils.config import load_config, resolve_path

ROOT = Path(__file__).resolve().parent.parent


def discover_songs(src: Path) -> dict[str, list[tuple[str, Path]]]:
    songs: dict[str, list[tuple[str, Path]]] = {}
    for folder in sorted(src.iterdir()):
        if not folder.is_dir() or "#" not in folder.name:
            continue
        if not any(folder.glob("*.wav")):
            continue
        speaker, song = folder.name.split("#", 1)
        songs.setdefault(song.strip(), []).append((speaker.strip(), folder))
    if not songs:
        raise RuntimeError(f"No valid M4Singer folders found in: {src}")
    return songs


def split_by_song(
    songs: dict[str, list[tuple[str, Path]]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    random.seed(seed)
    names = sorted(songs.keys())
    random.shuffle(names)
    n_train = int(len(names) * train_ratio) 
    n_dev = int(len(names) * dev_ratio)
    return {
        "train": set(names[:n_train]),
        "dev": set(names[n_train:n_train + n_dev]),
        "test": set(names[n_train + n_dev:]),
    }


def ensure_speakers_in_train(
    split_map: dict[str, set[str]],
    songs: dict[str, list[tuple[str, Path]]],
) -> None:
    speakers = lambda group: {s for song in group for s, _ in songs[song]}

    # Move one song at a time so every speaker has training examples.
    for _ in range(len(songs)):
        missing = (speakers(split_map["dev"]) | speakers(split_map["test"])) - speakers(
            split_map["train"]
        )

        if not missing:
            return

        moved = False

        for spk in missing:
            for src in ("dev", "test"):
                song = None
                for candidate_song in split_map[src]:
                    for speaker, _ in songs[candidate_song]:
                        if speaker == spk:
                            song = candidate_song
                            break
                    if song is not None:
                        break
                if song:
                    split_map[src].remove(song)
                    split_map["train"].add(song)
                    moved = True
                    break
            if moved:
                break
        if not moved:
            raise RuntimeError(f"Cannot place these speakers in train: {sorted(missing)}")


def link_file(file_path: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(file_path.resolve(), destination.parent.resolve()))


def materialize_split(
    split_map: dict[str, set[str]],
    songs: dict[str, list[tuple[str, Path]]],
    destination: Path,
    suffixes: set[str],
) -> None:
    for split, split_songs in split_map.items():
        speakers = set()
        folders = 0
        for song in split_songs:
            for speaker, song_dir in songs[song]:
                speakers.add(speaker)
                folders += 1
                for file in song_dir.iterdir():
                    if file.suffix in suffixes:
                        link_file(file, destination / split / speaker / song / file.name)
        print(
            f"{split:5s}  singers={len(speakers):3d}  "
            f"songs={len(split_songs):3d}  folders={folders}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dataset/m4singer.yaml",
        help="Dataset config YAML.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    src = resolve_path(cfg["source_dir"])
    dst = resolve_path(cfg["raw_dir"])
    split_cfg = cfg.get("split", {})
    suffixes = set(split_cfg.get("suffixes", [".wav"]))

    if split_cfg.get("clean", False) and dst.exists():
        shutil.rmtree(dst)

    songs = discover_songs(src)
    split_map = split_by_song(
        songs,
        train_ratio=split_cfg.get("train_ratio", 0.8),
        dev_ratio=split_cfg.get("dev_ratio", 0.1),
        seed=split_cfg.get("seed", 42),
    )
    ensure_speakers_in_train(split_map, songs)
    materialize_split(split_map, songs, dst, suffixes)

    print(f"\nDataset prepared.\nOutput: {dst}")


if __name__ == "__main__":
    main()
