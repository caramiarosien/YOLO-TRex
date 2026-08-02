"""
Brightness analysis for .mp4 files in a directory.

Analyzes all .mp4 files, computes the average brightness
over evenly spaced frames, categorizes the videos, and saves results to CSV.
"""

import cv2
import os
import csv
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Path to the directory containing all .mp4 files
VIDEO_DIR = "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/17Nov-raw"

# Path to the CSV output file
OUTPUT_CSV = "/Users/cara/Desktop/BA/Data/2019_brightness/video_brightness-17Nov19.csv"

# Interval in seconds between frames to analyze
SAMPLE_INTERVAL_SEC = 30

# Brightness threshold values for categorization
THRESHOLD_DARK = 35       # <= 35 → dark
THRESHOLD_MEDIUM = 85     # 36–85 → medium, > 85 → light

# ============================================================


def get_brightness_category(brightness: float) -> str:
    """
    Returns the brightness category based on thresholds:
      - dark:    brightness <= 35
      - medium:  brightness 36–85
      - light:   brightness > 85
    """
    if brightness <= THRESHOLD_DARK:
        return "dark"
    elif brightness <= THRESHOLD_MEDIUM:
        return "medium"
    else:
        return "light"


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

    # Calculate frame interval based on the second interval
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


def main():
    video_path_obj = Path(VIDEO_DIR)
    if not video_path_obj.is_dir():
        print(f"[ERROR] Directory does not exist: {VIDEO_DIR}")
        return

    # Recursive search for all files filtered by .mp4 (case-insensitive)
    # Filter out hidden files
    mp4_files = sorted([
        f for f in video_path_obj.rglob("*")
        if f.is_file() and f.suffix.lower() == ".mp4" and not f.name.startswith(".")
    ])

    if not mp4_files:
        print(f"No .mp4 files found in {VIDEO_DIR} (or subdirectories).")
        return

    print(f"Directory: {VIDEO_DIR} (including subdirectories)")
    print(f"Found videos: {len(mp4_files)}")
    print(f"Sampling interval: every {SAMPLE_INTERVAL_SEC} seconds")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("=" * 65)

    results = []

    for filepath in mp4_files:
        video_name = filepath.name

        # Relative path from VIDEO_DIR to parent folder of the file
        # Useful for deep subdirectory structures
        try:
            rel_folder = filepath.parent.relative_to(video_path_obj)
            location = str(rel_folder) if str(rel_folder) != "." else "Root"
        except ValueError:
            location = "Unknown"

        # Site is the name of the main directory
        standort = video_path_obj.name

        brightness = analyze_video_brightness(str(filepath))

        if brightness is not None:
            category = get_brightness_category(brightness)
            print(f"Video: {video_name} | Location: {location} | Brightness: {brightness:.1f} | Category: {category}")
            results.append([VIDEO_DIR, standort, location, video_name, round(brightness, 2), category])
        else:
            print(f"Video: {video_name} | Analysis failed")

        print("-" * 65)

    # Ensure target directory for CSV exists
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['Pfad (Video_dir)', 'Standort', 'Location', 'Videoname', 'Helligkeit', 'Kategorie'])
            writer.writerows(results)
        print(f"Results successfully saved to {OUTPUT_CSV}.")
    except Exception as e:
        print(f"[ERROR] Could not write to CSV file: {e}")

    print("=" * 65)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
