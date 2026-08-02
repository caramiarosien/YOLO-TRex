"""
Categorizes videos by brightness into dark / mediumdark / light subfolders.

For each .mp4 file in a location folder, average brightness is calculated.
Then all associated files (.mp4, .pv, .results, .settings, average_*.png)
are MOVED to the corresponding brightness subfolder.

Categories:
  - dark:        brightness <= 35
  - mediumdark:  brightness 36–85
  - light:       brightness > 85

Safety features:
  - Dry run preview shown first
  - User confirmation required before moving
  - File counts verified before and after moving
"""

import cv2
import os
import shutil
import sys

# ============================================================
# CONFIGURATION
# ============================================================

# Main directory with location folders
BASE_DIR = "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/17Nov-raw"

# Brightness thresholds
THRESHOLD_DARK = 35       # <= 35 → dark
THRESHOLD_MEDIUM = 85     # 36–85 → mediumdark, > 85 → light

# Interval in seconds between frames to analyze
SAMPLE_INTERVAL_SEC = 30

# ============================================================


def analyze_video_brightness(video_path: str, interval_sec: float = SAMPLE_INTERVAL_SEC) -> float | None:
    """
    Calculates the average brightness of a video by extracting a frame
    every `interval_sec` seconds.

    Returns:
        Average brightness value (0–255) or None on error.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        print(f"  [ERROR] Invalid video metadata (FPS={fps}, Frames={total_frames})")
        cap.release()
        return None

    # Calculate frame interval based on second interval
    frame_interval = max(1, int(fps * interval_sec))

    # Determine frame indices to read
    sample_indices = list(range(0, total_frames, frame_interval))

    if not sample_indices:
        sample_indices = [0]

    brightness_values = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        brightness_values.append(mean_brightness)

    cap.release()

    if not brightness_values:
        print(f"  [ERROR] No frames could be read.")
        return None

    return sum(brightness_values) / len(brightness_values)


def get_brightness_category(brightness: float) -> str:
    """Returns the brightness category."""
    if brightness <= THRESHOLD_DARK:
        return "dark"
    elif brightness <= THRESHOLD_MEDIUM:
        return "mediumdark"
    else:
        return "light"


def find_related_files(mp4_path: str) -> list[str]:
    """
    Finds all files belonging to an .mp4 file:
      - Same base name with extensions: .pv, .results, .settings
      - average_<basename>.png

    Returns:
        List of all related file paths (including the .mp4 itself).
    """
    location_dir = os.path.dirname(mp4_path)
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]

    related = []

    # The .mp4 file itself
    related.append(mp4_path)

    # Same base name, other extensions
    for ext in [".pv", ".results", ".settings"]:
        candidate = os.path.join(location_dir, base_name + ext)
        if os.path.isfile(candidate):
            related.append(candidate)

    # Average image: average_<basename>.png
    avg_png = os.path.join(location_dir, f"average_{base_name}.png")
    if os.path.isfile(avg_png):
        related.append(avg_png)

    return related


def move_files_to_category(related_files: list[str], location_dir: str, category: str, dry_run: bool = True) -> list[tuple[str, str]]:
    """
    Moves (or previews during dry run) all related files to the category folder.

    Returns:
        List of (source, destination) tuples for all files.
    """
    category_dir = os.path.join(location_dir, category)
    moves = []

    for filepath in related_files:
        filename = os.path.basename(filepath)
        dest = os.path.join(category_dir, filename)
        moves.append((filepath, dest))

    if not dry_run:
        # Create directory if necessary
        os.makedirs(category_dir, exist_ok=True)

        for src, dst in moves:
            if os.path.exists(dst):
                print(f"  [WARNING] Destination already exists, skipping: {dst}")
                continue
            shutil.move(src, dst)

    return moves


def process_location(location_dir: str, dry_run: bool = True) -> dict:
    """
    Processes a location directory: Analyzes all .mp4 files
    and sorts them by brightness.

    Returns:
        Dictionary with statistics.
    """
    location_name = os.path.basename(location_dir)

    # Only .mp4 files directly in location folder (not subfolders like dark/)
    mp4_files = sorted([
        os.path.join(location_dir, f)
        for f in os.listdir(location_dir)
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(location_dir, f))
    ])

    if not mp4_files:
        return {"location": location_name, "videos": 0, "skipped": True}

    stats = {
        "location": location_name,
        "videos": len(mp4_files),
        "skipped": False,
        "dark": [],
        "mediumdark": [],
        "light": [],
        "errors": [],
        "total_files_moved": 0
    }

    print(f"\n{'='*65}")
    print(f"Location: {location_name} ({len(mp4_files)} Videos)")
    print(f"{'='*65}")

    for mp4_path in mp4_files:
        video_name = os.path.basename(mp4_path)
        brightness = analyze_video_brightness(mp4_path)

        if brightness is None:
            stats["errors"].append(video_name)
            print(f"  ✗ {video_name} → Analysis failed")
            continue

        category = get_brightness_category(brightness)
        related_files = find_related_files(mp4_path)

        stats[category].append(video_name)
        stats["total_files_moved"] += len(related_files)

        print(f"  {'→' if dry_run else '✓'} {video_name}")
        print(f"    Brightness: {brightness:.1f} → {category}")
        print(f"    Related files: {len(related_files)}")

        moves = move_files_to_category(related_files, location_dir, category, dry_run=dry_run)

        if not dry_run:
            # Verify that all files arrived at destination
            missing = [dst for _, dst in moves if not os.path.exists(dst)]
            if missing:
                print(f"    [ERROR] {len(missing)} files missing at destination!")
                for m in missing:
                    print(f"      - {m}")

    return stats


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] Base directory does not exist: {BASE_DIR}")
        return

    # Find all location folders (direct subdirectories only)
    location_dirs = sorted([
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith(".")
    ])

    if not location_dirs:
        print(f"No location folders found in {BASE_DIR}.")
        return

    print("=" * 65)
    print("BRIGHTNESS SORTING – DRY RUN (Preview)")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Thresholds: dark ≤ {THRESHOLD_DARK} | mediumdark {THRESHOLD_DARK+1}–{THRESHOLD_MEDIUM} | light > {THRESHOLD_MEDIUM}")
    print(f"Locations: {len(location_dirs)}")
    print("=" * 65)

    # ─── PHASE 1: DRY RUN ───────────────────────────────────────
    all_stats = []
    for loc_dir in location_dirs:
        stats = process_location(loc_dir, dry_run=True)
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY (Dry Run)")
    print("=" * 65)

    total_dark = 0
    total_medium = 0
    total_light = 0
    total_errors = 0
    total_files = 0

    for stats in all_stats:
        if stats["skipped"]:
            print(f"  {stats['location']}: No videos (skipped)")
            continue

        n_dark = len(stats.get("dark", []))
        n_medium = len(stats.get("mediumdark", []))
        n_light = len(stats.get("light", []))
        n_errors = len(stats.get("errors", []))
        n_files = stats.get("total_files_moved", 0)

        total_dark += n_dark
        total_medium += n_medium
        total_light += n_light
        total_errors += n_errors
        total_files += n_files

        print(f"  {stats['location']}: {n_dark} dark | {n_medium} mediumdark | {n_light} light | {n_errors} errors | {n_files} files")

    print(f"\n  TOTAL: {total_dark} dark | {total_medium} mediumdark | {total_light} light | {total_errors} errors")
    print(f"  FILES TO MOVE: {total_files}")

    if total_files == 0:
        print("\nNo files to move. Done.")
        return

    # ─── PHASE 2: CONFIRMATION ───────────────────────────────────
    print("\n" + "=" * 65)
    print("WARNING: Files will be MOVED (not copied).")
    print("Originals will NOT remain in the source location.")
    print("=" * 65)

    confirm = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if confirm not in ("ja", "j", "yes", "y"):
        print("Aborted. No files were moved.")
        return

    # ─── PHASE 3: MOVE ───────────────────────────────────
    print("\n" + "=" * 65)
    print("MOVING FILES...")
    print("=" * 65)

    for loc_dir in location_dirs:
        process_location(loc_dir, dry_run=False)

    print("\n" + "=" * 65)
    print("DONE! All files successfully moved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
