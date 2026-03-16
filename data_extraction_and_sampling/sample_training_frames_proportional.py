#!/usr/bin/env python3
import argparse, math, os, random, shutil
from pathlib import Path

# Unterstützte Bild-Erweiterungen
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_image_files(folder: Path):
    """
    Listet alle Bilddateien in einem Ordner und dessen UNTERORDNERN (rekursiv) auf.
    Die Änderung: von .iterdir() zu .rglob('*').
    """
    # **FIX**: Rekursive Suche in allen Unterordnern
    return [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT]

def proportional_allocation(counts, target):
    """
    Berechnet die proportionale Zuweisung (Quoten) der Bilder pro Ordner.
    Erlaubt nun auch Zielwerte > Gesamtanzahl (Oversampling).
    """
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    
    base, remainder = {}, {}
    for k, n in counts.items():
        q = (n / total) * target
        # Kein 'min(n, ...)' mehr, da wir Oversampling erlauben wollen
        base[k] = int(q)
        remainder[k] = q - int(q)
        
    allocated = sum(base.values())
    need = target - allocated
    
    # Largest Remainder Method: Verteile den Rest an die mit dem größten Bruchteil
    cands = sorted(counts.keys(), key=lambda k: remainder[k], reverse=True)
    
    for k in cands:
        if need <= 0: break
        base[k] += 1
        need -= 1
                
    return base

def main():
    ap = argparse.ArgumentParser(description="Stratifizierte, flache Zufallsauswahl von Bildern.")
    
    # Ihre spezifischen Standardwerte:
    ap.add_argument("--root", default="/Volumes/WD Elements/KasankaCameras", help="Wurzelordner mit den Unterordnern")
    ap.add_argument("--out",  default="/Users/cara/Desktop/Test_v5/train/images/Kasanka22", help="Zielordner (flach)")
    ap.add_argument("--target", type=int, default=100, help="Gesamtzahl der zu kopierenden Bilder (Standard: 75)")
    
    # Optionale Argumente
    ap.add_argument("--seed", type=int, default=None, help="Optional: Zufalls-Seed für Reproduzierbarkeit")
    ap.add_argument("--clean", action="store_true", help="Zielordner vorab leeren (vorsichtig!)")
    
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    out  = Path(args.out).expanduser().resolve()
    
    if not root.exists():
        raise SystemExit(f"Root nicht gefunden: {root}")

    # Unterordner finden (diese dienen als Stratifikationseinheit, z.B. Kamera_1, Kamera_2, etc.)
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != out.name]
    if not subdirs:
        raise SystemExit("Keine Unterordner gefunden.")

    # Dateien je Ordner (jetzt rekursiv) zählen
    files_by_dir, counts = {}, {}
    for d in sorted(subdirs, key=lambda p: p.name):
        # Ruft die rekursiv angepasste Funktion auf
        files = list_image_files(d) 
        random.shuffle(files)
        files_by_dir[d] = files
        counts[d] = len(files)

    total_imgs = sum(counts.values())
    if total_imgs == 0:
        raise SystemExit("Keine Bilddateien gefunden.") # Dieser Fehler sollte jetzt behoben sein

    target = args.target
    if target > total_imgs:
        print(f"Info: Ziel ({target}) > Verfügbar ({total_imgs}). Es wird mit Zurücklegen (Oversampling) gezogen.")

    alloc = proportional_allocation(counts, target)

    print(f"Gefundene Bilder gesamt: {total_imgs}")
    print(f"Geplante Auswahl gesamt: {sum(alloc.values())} (Ziel: {target})")
    print("-" * 50)
    for d in sorted(counts.keys(), key=lambda p: p.name):
        print(f"{d.name:25s}  verfügbar: {counts[d]:5d}  -> Auswahl: {alloc[d]:5d}")
    print("-" * 50)

    # Auswahl der Bilder
    chosen = []
    for d, k in alloc.items():
        if k <= 0: continue
        if k > len(files_by_dir[d]):
            pick = random.choices(files_by_dir[d], k=k)
        else:
            pick = random.sample(files_by_dir[d], k)
        chosen.extend([(d.name, p) for p in pick])

    random.shuffle(chosen)  # Globaler Mix

    # Zielordner vorbereiten
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for f in out.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        print(f"[clean] {removed} alte Dateien in '{out}' entfernt.")

    # Flaches Kopieren
    copied = 0
    for folder_name, src in chosen:
        stem, ext = src.stem, src.suffix.lower()
        # Der Dateiname folgt dem Schema: <video name>_<date>_frame
        # Wir nehmen an, dass 'stem' bereits das Format <date>_<frame> oder ähnlich enthält.
        # Fallback für Duplikate (wegen Oversampling): Counter anhängen.
        
        base_name = f"{folder_name}_{stem}"
        dest = out / f"{base_name}{ext}"
        
        counter = 1
        while dest.exists():
            dest = out / f"{base_name}_{counter}{ext}"
            counter += 1
            
        shutil.copy2(src, dest)
        copied += 1


    print(f"\nFertig. {copied} Bilder flach nach '{out}' kopiert.")

if __name__ == "__main__":
    main()