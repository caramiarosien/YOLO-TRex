#!/usr/bin/env python3
import argparse, re, shutil, sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAME_RE = re.compile(r"^(?P<folder>[^/\\]+)__(?P<stem>.+?)__([0-9a-fA-F]{6})\.[^.]+$")

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXT

def parse_origin(fname: str):
    m = NAME_RE.match(fname)
    if not m:
        return None, Path(fname).stem
    return m.group("folder"), m.group("stem")

def find_label(stem: str, origin_folder: str|None, label_roots: list[Path]) -> Path|None:
    cands = []
    for r in label_roots:
        if origin_folder:
            cands += [
                r / origin_folder / f"{stem}.txt",
                r / "labels" / origin_folder / f"{stem}.txt",
                r / "Labels" / origin_folder / f"{stem}.txt",
            ]
        cands += [
            r / f"{stem}.txt",
            r / "labels" / f"{stem}.txt",
            r / "Labels" / f"{stem}.txt",
        ]
    for c in cands:
        if c.exists() and c.is_file():
            return c
    return None

def copy_pair(img_src: Path, lbl_src: Path|None, img_dst: Path, lbl_dst: Path, create_empty: bool):
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    # Kollisionen vermeiden
    target = img_dst
    i = 1
    while target.exists():
        target = img_dst.with_stem(f"{img_dst.stem}({i})")
        i += 1
    shutil.copy2(img_src, target)

    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if lbl_src and lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    elif create_empty:
        lbl_dst.write_text("")

def stage(split_src: Path, split_name: str, ds_dir: Path, label_roots: list[Path], create_empty: bool):
    if not split_src.exists():
        print(f"[WARN] Split-Ordner fehlt: {split_src}")
        return 0, 0
    imgs = [p for p in split_src.iterdir() if is_image(p)]
    print(f"[INFO] {split_name}: gefunden {len(imgs)} Bilddateien in {split_src}")
    if len(imgs) == 0:
        # kleine Hilfe: zeige 5 Dateien (falls falsche Endungen)
        others = list(split_src.iterdir())
        if others:
            print("[DEBUG] Beispiel-Dateien (erste 5):")
            for x in others[:5]:
                print("       ", x.name)
    out_imgs = ds_dir / split_name / "images"
    out_lbls = ds_dir / split_name / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, 0
    for img in imgs:
        origin_folder, stem = parse_origin(img.name)
        lbl_src = find_label(stem, origin_folder, label_roots)
        img_dst = out_imgs / f"{stem}{img.suffix.lower()}"
        lbl_dst = out_lbls / f"{stem}.txt"
        had_lbl = lbl_src is not None and lbl_src.exists()
        copy_pair(img, lbl_src, img_dst, lbl_dst, create_empty)
        if not had_lbl and not create_empty:
            missing += 1
        copied += 1
    return copied, missing

def write_data_yaml(ds_dir: Path, nc: int, names: list[str]):
    y = ds_dir / "data.yaml"
    lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
    ]
    if (ds_dir/"test/images").exists() and any((ds_dir/"test/images").glob("*")):
        lines.append("test: test/images")
    names_line = "names: [" + ", ".join(f"'{n}'" for n in names) + "]"  # <-- fix ohne Backslashes
    lines += [f"nc: {nc}", names_line]
    y.write_text("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Baut YOLO-Struktur aus Images/Detection/{Train,Valid,Test}.")
    ap.add_argument("--detection-dir", default="/Users/cara/Desktop/BA/YOLO/Images/Detection")
    ap.add_argument("--dataset-dir",   default="/Users/cara/Desktop/BA/YOLO/YOLO_Dataset")
    ap.add_argument("--label-roots",   nargs="*", default=[
        "/Users/cara/Desktop/BA/YOLO/Images"  # hier liegen evtl. Original-Labeldateien
    ])
    ap.add_argument("--create-empty-labels", action="store_true", help="fehlende Labels als leere .txt anlegen")
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--names", nargs="*", default=["object"])
    args = ap.parse_args()

    det = Path(args.detection_dir).expanduser().resolve()
    ds  = Path(args.dataset_dir).expanduser().resolve()
    if not det.exists():
        sys.exit(f"Detection-Ordner nicht gefunden: {det}")
    train_src = det / "Train"
    valid_src = det / "Valid"
    test_src  = det / "Test"

    label_roots = [Path(p).expanduser().resolve() for p in args.label_roots]
    ds.mkdir(parents=True, exist_ok=True)

    c_tr, m_tr = stage(train_src, "train", ds, label_roots, args.create_empty_labels)
    c_va, m_va = stage(valid_src, "valid", ds, label_roots, args.create_empty_labels)
    c_te = m_te = 0
    if test_src.exists():
        c_te, m_te = stage(test_src, "test", ds, label_roots, args.create_empty_labels)

    if c_tr == 0 and c_va == 0 and c_te == 0:
        sys.exit("Keine Bilddateien gefunden. Prüfe Pfade und Dateiendungen.")

    write_data_yaml(ds, args.nc, args.names)
    print("\n== Zusammenfassung ==")
    print(f"train: {c_tr} Bilder | fehlende Labels (nicht erzeugt): {m_tr}")
    print(f"valid: {c_va} Bilder | fehlende Labels (nicht erzeugt): {m_va}")
    if c_te or test_src.exists():
        print(f"test : {c_te} Bilder | fehlende Labels (nicht erzeugt): {m_te}")
    print(f"data.yaml -> {ds/'data.yaml'}")
    print("Fertig.")

if __name__ == "__main__":
    main()
