#!/usr/bin/env python3
"""
Sammelt zufällig ausgewählte Frame-Dateien aus dem Kameraverzeichnis und kopiert
sie in einen Zielordner. Entspricht dem bisherigen Shell-Snippet, ist aber als
gültiges Python-Skript nutzbar.
"""

import argparse
import random
import shutil
from pathlib import Path


def find_frame_files(source: Path):
    """Alle Dateien zurückgeben, die in Ordnern mit Suffix '_frames' liegen."""
    files = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if any(parent.name.endswith("_frames") for parent in path.parents):
            files.append(path)
    return files


def copy_files(files, destination: Path, limit: int):
    destination.mkdir(parents=True, exist_ok=True)
    selected = files if limit is None else files[:limit]
    copied = 0
    for src in selected:
        dest = destination / src.name
        # Bei Namenskonflikten Suffix anhängen
        counter = 1
        final_dest = dest
        while final_dest.exists():
            final_dest = destination / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        shutil.copy2(src, final_dest)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Zufällige Frames nach potential_train_2022 kopieren."
    )
    parser.add_argument(
        "--source",
        default="/Volumes/WD Elements/KasankaCameras",
        help="Wurzelordner, der *_frames-Unterordner enthält",
    )
    parser.add_argument(
        "--destination",
        default="/Users/cara/Desktop/BA/potential_train_2022",
        help="Zielordner für kopierte Frames",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximale Anzahl zu kopierender Dateien (None = alle)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionaler Zufalls-Seed für reproduzierbare Auswahl",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Quellordner nicht gefunden: {source}")

    if args.seed is not None:
        random.seed(args.seed)

    files = find_frame_files(source)
    if not files:
        raise SystemExit("Keine passenden Frame-Dateien gefunden.")

    random.shuffle(files)
    limit = None if args.limit is None or args.limit <= 0 else args.limit
    destination = Path(args.destination).expanduser().resolve()
    copied = copy_files(files, destination, limit)

    print(
        f"{copied} Dateien nach '{destination}' kopiert "
        f"(aus {len(files)} gefundenen Frames)."
    )


if __name__ == "__main__":
    main()

