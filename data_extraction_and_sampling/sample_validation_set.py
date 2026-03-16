#!/usr/bin/env python3
import argparse, math, os, random, shutil, uuid
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_image_files(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]

def proportional_allocation(counts, target):
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    # proportional via floor + Largest Remainder
    base, remainder = {}, {}
    for k, n in counts.items():
        q = (n / total) * target
        b = min(n, int(q))
        base[k] = b
        remainder[k] = q - b
    allocated = sum(base.values())
    need = min(target, total) - allocated
    while need > 0:
        # nur Ordner mit Restkapazität
        cands = [k for k, n in counts.items() if base[k] < n]
        if not cands: break
        cands.sort(key=lambda k: (remainder[k], counts[k] - base[k]), reverse=True)
        for k in cands:
            if need == 0: break
            if base[k] < counts[k]:
                base[k] += 1
                need -= 1
    return base

def main():
    ap = argparse.ArgumentParser(description="Stratifizierte, flache Zufallsauswahl von Bildern.")
    ap.add_argument("--root", default="/Users/cara/Desktop/BA/YOLO/Images", help="Wurzelordner mit den 17 Unterordnern")
    ap.add_argument("--out",  default="/Volumes/WD Elements/KasankaCameras", help="Zielordner (flach)")
    ap.add_argument("--target", type=int, default=572, help="Gesamtzahl der zu kopierenden Bilder")
    ap.add_argument("--seed", type=int, default=None, help="Optional: Zufalls-Seed für Reproduzierbarkeit")
    ap.add_argument("--clean", action="store_true", help="Zielordner vorab leeren (vorsichtig!)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    out  = Path(args.out).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root nicht gefunden: {root}")

    # Nur echte Unterordner, Train ausschließen
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != out.name]
    if not subdirs:
        raise SystemExit("Keine Unterordner gefunden.")

    # Dateien je Ordner
    files_by_dir, counts = {}, {}
    for d in sorted(subdirs, key=lambda p: p.name):
        files = list_image_files(d)
        # Durchmischen, damit random.sample (falls später ersetzt) unabhängig ist
        random.shuffle(files)
        files_by_dir[d] = files
        counts[d] = len(files)

    total_imgs = sum(counts.values())
    if total_imgs == 0:
        raise SystemExit("Keine Bilddateien gefunden.")

    target = min(args.target, total_imgs)
    alloc = proportional_allocation(counts, target)

    print(f"Gefundene Bilder gesamt: {total_imgs}")
    print(f"Geplante Auswahl gesamt: {sum(alloc.values())} (Ziel: {target})")
    for d in sorted(counts.keys(), key=lambda p: p.name):
        print(f"{d.name:25s}  verfügbar: {counts[d]:5d}  -> Auswahl: {alloc[d]:5d}")

    # Auswahl je Ordner -> globaler Pool, dann global shuffle für bunten Mix
    chosen = []
    for d, k in alloc.items():
        if k <= 0: continue
        # sample ohne zeitliche Nachbarschaft bevorzugen: random.sample erfüllt das bereits
        pick = random.sample(files_by_dir[d], k)
        chosen.extend([(d.name, p) for p in pick])

    random.shuffle(chosen)  # globaler Mix

    # Zielordner vorbereiten
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        # WARNUNG: löscht nur Dateien, keine Unterordner (sollten nicht existieren)
        removed = 0
        for f in out.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        print(f"[clean] {removed} Dateien in '{out}' entfernt.")

    # Flaches Kopieren mit eindeutigen Namen
    copied = 0
    for folder_name, src in chosen:
        stem, ext = src.stem, src.suffix.lower()
        # eindeutiger Zielsuffix, vermeidet Kollisionen und erhält Ordner-Herkunft
        tag = uuid.uuid4().hex[:6]
        dest = out / f"{folder_name}__{stem}__{tag}{ext}"
        shutil.copy2(src, dest)
        copied += 1

    print(f"\nFertig. {copied} Bilder flach nach '{out}' kopiert.")

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
import argparse, math, random, shutil, re, uuid
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_NAME_RE = re.compile(r"^(?P<folder>[^/\\]+)__(?P<stem>.*)__[0-9a-fA-F]{6}\.[^.]+$")

def list_image_files(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]

def proportional_allocation(counts, target):
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    base, remainder = {}, {}
    for k, n in counts.items():
        q = (n / total) * target
        b = min(n, int(q))
        base[k] = b
        remainder[k] = q - b
    allocated = sum(base.values())
    need = min(target, total) - allocated
    while need > 0:
        cands = [k for k, n in counts.items() if base[k] < n]
        if not cands: break
        cands.sort(key=lambda k: (remainder[k], counts[k] - base[k]), reverse=True)
        for k in cands:
            if need == 0: break
            if base[k] < counts[k]:
                base[k] += 1
                need -= 1
    return base

def parse_train_sources(train_dir: Path):
    """Liest Train-Dateien (flach) und extrahiert (folder, stem) aus 'folder__stem__tag.ext'."""
    used = set()
    if not train_dir.exists():
        return used
    for f in train_dir.iterdir():
        if not f.is_file(): continue
        m = TRAIN_NAME_RE.match(f.name)
        if m:
            folder = m.group("folder")
            stem = m.group("stem")
            used.add((folder, stem.lower()))
    return used

def main():
    ap = argparse.ArgumentParser(description="Validierungsset: 572 Bilder, flach, ohne Überschneidung mit Train.")
    ap.add_argument("--root", default="/Users/cara/Desktop/BA/YOLO/Images", help="Wurzel mit den 17 Unterordnern")
    ap.add_argument("--train", default="/Users/cara/Desktop/BA/YOLO/Images/Train", help="Train-Ordner (flach, Quelle ausschließen)")
    ap.add_argument("--out",  default="/Volumes/WD Elements/KasankaCameras", help="Zielordner (flach)")
    ap.add_argument("--target", type=int, default=572, help="Zielgröße Valid")
    ap.add_argument("--seed", type=int, default=None, help="Zufalls-Seed für Reproduzierbarkeit")
    ap.add_argument("--clean", action="store_true", help="Zielordner vorab leeren (nur Dateien)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root  = Path(args.root).expanduser().resolve()
    train = Path(args.train).expanduser().resolve()
    out   = Path(args.out).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Root nicht gefunden: {root}")

    # Train-Quellen erkennen (Ordner, Stem)
    used_src = parse_train_sources(train)
    print(f"Train-Sperrliste: {len(used_src)} Einträge erkannt.")

    # Unterordner (Train und Valid-Struktur ausschließen)
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name not in {train.name, Path(args.out).name}]
    if not subdirs:
        raise SystemExit("Keine Unterordner gefunden.")

    # Verfügbare Dateien je Ordner (ohne Train-Quellen)
    files_by_dir, counts = {}, {}
    for d in sorted(subdirs, key=lambda p: p.name):
        candidates = []
        for p in list_image_files(d):
            key = (d.name, p.stem.lower())
            if key in used_src:
                continue
            candidates.append(p)
        random.shuffle(candidates)  # interne Durchmischung
        files_by_dir[d] = candidates
        counts[d] = len(candidates)

    total_avail = sum(counts.values())
    if total_avail == 0:
        raise SystemExit("Keine verfügbaren Bilddateien für Valid (nach Ausschluss von Train).")

    target = min(args.target, total_avail)
    alloc = proportional_allocation(counts, target)

    print(f"Verfügbar nach Ausschluss: {total_avail}")
    print(f"Geplante Auswahl gesamt: {sum(alloc.values())} (Ziel: {target})")
    for d in sorted(counts.keys(), key=lambda p: p.name):
        print(f"{d.name:25s}  verfügbar: {counts[d]:5d}  -> Auswahl: {alloc[d]:5d}")

    # Auswahl je Ordner -> global mischen
    chosen = []
    for d, k in alloc.items():
        if k <= 0: continue
        pick = random.sample(files_by_dir[d], k)
        chosen.extend([(d.name, p) for p in pick])
    random.shuffle(chosen)

    # Zielordner vorbereiten
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for f in out.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        print(f"[clean] {removed} Dateien in '{out}' entfernt.")

    # Flaches Kopieren mit eindeutigem Namen
    copied = 0
    for folder_name, src in chosen:
        stem, ext = src.stem, src.suffix.lower()
        tag = uuid.uuid4().hex[:6]
        dest = out / f"{folder_name}__{stem}__{tag}{ext}"
        shutil.copy2(src, dest)
        copied += 1

    print(f"\nFertig. {copied} Valid-Bilder flach nach '{out}' kopiert.")
    if copied < args.target:
        print(f"Hinweis: Nur {copied} verfügbar/zugewiesen (weniger als Ziel {args.target}).")

if __name__ == "__main__":
    main()
