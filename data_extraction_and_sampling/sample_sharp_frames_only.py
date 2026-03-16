#!/usr/bin/env python3
import argparse, math, os, random, shutil
import cv2  # WICHTIG: pip install opencv-python
from pathlib import Path

# Unterstützte Bild-Erweiterungen
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_image_files(folder: Path):
    """
    Listet alle Bilddateien in einem Ordner und dessen UNTERORDNERN (rekursiv) auf.
    """
    return [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT]

def get_blur_score(image_path):
    """
    Berechnet die Varianz des Laplace-Filters.
    Hoher Wert = Scharf. Niedriger Wert = Unscharf.
    Gibt 0 zurück, wenn das Bild defekt ist.
    """
    try:
        # cv2.imread mag keine Path-Objekte unter manchen OS, daher str()
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception as e:
        print(f"Warnung: Konnte {image_path.name} nicht lesen: {e}")
        return 0.0

def proportional_allocation(counts, target):
    """
    Berechnet die proportionale Zuweisung (Quoten) der Bilder pro Ordner.
    """
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    
    base, remainder = {}, {}
    for k, n in counts.items():
        q = (n / total) * target
        base[k] = int(q)
        remainder[k] = q - int(q)
        
    allocated = sum(base.values())
    need = target - allocated
    
    # Largest Remainder Method
    cands = sorted(counts.keys(), key=lambda k: remainder[k], reverse=True)
    
    for k in cands:
        if need <= 0: break
        base[k] += 1
        need -= 1
                
    return base

def main():
    ap = argparse.ArgumentParser(description="Stratifizierte Zufallsauswahl mit Blur-Filter.")
    
    ap.add_argument("--root", default="/Volumes/Kasanka21/KasankaCameras", help="Wurzelordner")
    ap.add_argument("--out",  default="/Users/cara/Desktop/Test_v5/train/images/Kasanka21", help="Zielordner")
    ap.add_argument("--target", type=int, default=500, help="Gesamtzahl der SCHARFEN Bilder")
    ap.add_argument("--blur-threshold", type=float, default=50.0, help="Grenzwert für Schärfe (Var of Laplacian). Standard gesenkt auf 50.0.")
    
    ap.add_argument("--seed", type=int, default=None, help="Zufalls-Seed")
    ap.add_argument("--clean", action="store_true", help="Zielordner vorab leeren")
    
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    out  = Path(args.out).expanduser().resolve()
    
    if not root.exists():
        raise SystemExit(f"Root nicht gefunden: {root}")

    print("Scanne Verzeichnisse...")
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != out.name]
    if not subdirs:
        raise SystemExit("Keine Unterordner gefunden.")

    files_by_dir, counts = {}, {}
    for d in sorted(subdirs, key=lambda p: p.name):
        files = list_image_files(d) 
        # Wir shuffeln hier schon, damit wir später beim Durchsuchen zufällige Bilder greifen
        random.shuffle(files)
        files_by_dir[d] = files
        counts[d] = len(files)

    total_imgs = sum(counts.values())
    if total_imgs == 0:
        raise SystemExit("Keine Bilddateien gefunden.")

    target = args.target
    print(f"Gefundene Bilder gesamt (ungefiltert): {total_imgs}")

    # Berechnung der Quoten (basierend auf Dateianzahl, noch ohne Schärfe-Wissen)
    alloc = proportional_allocation(counts, target)
    
    print("-" * 60)
    print(f"{'Ordner':<25} | {'Vorhanden':<10} | {'Ziel (Scharf)':<15}")
    print("-" * 60)
    for d in sorted(counts.keys(), key=lambda p: p.name):
        print(f"{d.name:<25} | {counts[d]:<10} | {alloc[d]:<15}")
    print("-" * 60)

    # Zielordner vorbereiten
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for f in out.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        print(f"[clean] {removed} alte Dateien entfernt.")

    # --- HAUPTSCHLEIFE: Auswahl & Prüfung ---
    print(f"\nStarte Auswahl und Schärfeprüfung (Threshold: {args.blur_threshold})...")
    
    copied_total = 0
    skipped_blur = 0
    
    for d, needed_count in alloc.items():
        if needed_count <= 0: continue
        
        candidates = files_by_dir[d] # Ist bereits gemischt
        selected_for_folder = []
        
        # Wir iterieren durch die Kandidaten in diesem Ordner
        # und nehmen nur die scharfen, bis wir 'needed_count' erreicht haben.
        idx = 0
        while len(selected_for_folder) < needed_count and idx < len(candidates):
            current_img = candidates[idx]
            idx += 1
            
            # BLUR CHECK
            score = get_blur_score(current_img)
            
            if score >= args.blur_threshold:
                selected_for_folder.append(current_img)
            else:
                skipped_blur += 1
                # Optional: Debug-Ausgabe, falls du sehen willst, was fliegt
                # print(f"Unscharf ({score:.1f}): {current_img.name}")

        # Fallback: Wenn wir nicht genug scharfe Bilder gefunden haben
        # (z.B. Ordner hat 100 Bilder, wir brauchen 50, aber 80 sind unscharf -> wir haben nur 20)
        # Hier könntest du entscheiden: Auffüllen mit unscharfen? Oder einfach weniger nehmen?
        # Aktuelle Logik: Wir nehmen einfach weniger und melden das.
        
        if len(selected_for_folder) < needed_count:
            print(f"WARNUNG: In '{d.name}' wurden nur {len(selected_for_folder)} scharfe Bilder gefunden (Ziel war {needed_count}).")

        # Kopieren
        for src in selected_for_folder:
            stem, ext = src.stem, src.suffix.lower()
            base_name = f"{d.name}_{stem}"
            dest = out / f"{base_name}{ext}"
            
            counter = 1
            while dest.exists():
                dest = out / f"{base_name}_{counter}{ext}"
                counter += 1
                
            shutil.copy2(src, dest)
            copied_total += 1
            
            # Kleiner Fortschrittsanzeiger alle 50 Bilder
            if copied_total % 50 == 0:
                print(f"... {copied_total} kopiert (Aussortiert: {skipped_blur})")

    print("\n" + "="*50)
    print(f"FERTIG.")
    print(f"Bilder kopiert: {copied_total}")
    print(f"Wegen Unschärfe übersprungen: {skipped_blur}")
    print(f"Zielordner: {out}")
    print("="*50)

if __name__ == "__main__":
    main()