import argparse
from pathlib import Path
import cv2
import numpy as np
from yolo_segmentation_utils import masks_to_yolo_seg_lines, write_yolo_seg_file

def main():
    parser = argparse.ArgumentParser(description="Convert a folder of masks into YOLO format labels.")
    parser.add_argument("--img-dir", required=True, help="Directory containing original images")
    parser.add_argument("--mask-dir", required=True, help="Directory containing masks")
    parser.add_argument("--label-dir", required=True, help="Directory to save YOLO labels to")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID for YOLO labels")
    parser.add_argument("--approx-eps", type=float, default=0.5, help="Epsilon for polygon simplification (default 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    parser.add_argument("--verbose", action="store_true", help="Show detailed info about each image and missing masks")
    
    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    label_dir = Path(args.label_dir)
    
    if not img_dir.exists():
        print(f"❌ Error: Image Directory not found: {img_dir}")
        return
    if not mask_dir.exists():
        print(f"❌ Error: Mask Directory not found: {mask_dir}")
        return

    if not args.dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions])

    print(f"🚀 {'[DRY RUN] ' if args.dry_run else ''}Starting batch conversion...")
    print(f"📂 Images: {img_dir} ({len(image_paths)} found)")
    print(f"📂 Masks:  {mask_dir}")
    print(f"📂 Labels: {label_dir}")
    print("-" * 60)

    written_count = 0
    missing_masks = []

    for i, img_path in enumerate(image_paths):
        if not args.verbose and i % 100 == 0 and i > 0:
            print(f"   Processed {i}/{len(image_paths)} images...")

        # Flexible mask matching logic
        candidates = [
            mask_dir / f"{img_path.stem}_mask.png",
            mask_dir / f"{img_path.stem}_mask_cleaned.png",
            mask_dir / f"{img_path.stem}.png",
            mask_dir / f"{img_path.stem}.jpg"
        ]
        
        mask_file = None
        for cand in candidates:
            if cand.exists():
                mask_file = cand
                break
        
        if mask_file is None:
            missing_masks.append(img_path.name)
            if args.verbose:
                print(f"❓ Missing mask for: {img_path.name}")
            continue

        if args.verbose:
            print(f"🔄 Processing {img_path.name} using {mask_file.name}...")

        if args.dry_run:
            written_count += 1
            continue

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️  [ERR] Could not read mask: {mask_file.name}")
            missing_masks.append(img_path.name)
            continue

        try:
            lines = masks_to_yolo_seg_lines(mask, class_id=args.class_id, approx_eps=args.approx_eps)
            label_out = label_dir / f"{img_path.stem}.txt"
            write_yolo_seg_file(lines, label_out)
            written_count += 1
        except Exception as e:
            print(f"⚠️  [ERR] Failed processing {img_path.name}: {e}")

    print("-" * 60)
    print(f"✅ Finished.")
    print(f"📝 Labels {'simulated' if args.dry_run else 'created'}: {written_count}")
    print(f"⚠️  Missing or invalid masks: {len(missing_masks)}")
    
    if args.dry_run:
        print("💡 This was a dry run. No files were written.")
    
    if missing_masks and not args.verbose:
        print(f"💡 Run with --verbose to see which specific masks are missing.")
    elif missing_masks and args.verbose:
        print("📋 Full list of images with missing masks:")
        for m in missing_masks[:10]:
            print(f"   - {m}")
        if len(missing_masks) > 10:
            print(f"   ... and {len(missing_masks) - 10} more.")

if __name__ == "__main__":
    main()

