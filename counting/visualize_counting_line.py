#!/usr/bin/env python3
"""
visualize_counting_line.py — Draw the counting line on a sample video frame.
=========================================================================

Extracts a frame from the bat video and overlays the horizontal counting
line at Y = frame_height / 2 for visual verification.

Dependencies: matplotlib, numpy, ffmpeg (CLI tool — no cv2 required).

Usage:
    python3 visualize_counting_line.py
    python3 visualize_counting_line.py --video path/to/video.mp4 --frame_nr 5000
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def extract_frame(video_path: str, frame_nr: int) -> np.ndarray:
    """
    Extract a single frame from *video_path* at index *frame_nr*
    using ffmpeg (no cv2 dependency).
    """
    # Use ffmpeg to extract the frame as a temporary PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_nr})",
        "-frames:v", "1",
        tmp_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise IOError(
            f"ffmpeg failed to extract frame {frame_nr}:\n"
            f"{result.stderr.decode()}"
        )

    frame = mpimg.imread(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)  # clean up temp file
    return frame


def visualize(frame: np.ndarray, line_y: float, output_path: str, frame_nr: int):
    """Draw the counting line on *frame* and save the result."""
    h, w = frame.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame)

    # --- Draw the counting line ---
    ax.axhline(
        y=line_y,
        color="#00FF88",
        linewidth=2.5,
        linestyle="--",
        label=f"Counting Line  (Y = {line_y:.0f} px)",
    )

    # --- Annotate the two zones ---
    ax.text(
        w * 0.02, line_y - 30,
        "↑  ABOVE line  (low Y)",
        color="#00FF88", fontsize=12, fontweight="bold",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
    )
    ax.text(
        w * 0.02, line_y + 30,
        "↓  BELOW line  (high Y)",
        color="#FF6666", fontsize=12, fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
    )

    # --- Title & legend ---
    ax.set_title(
        f"Counting Line Visualisation  —  Frame #{frame_nr}   "
        f"({w}×{h} px)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
    ax.set_xlabel("X  (pixels)")
    ax.set_ylabel("Y  (pixels)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualisation to: {output_path}")
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Visualise the counting line on a video frame.",
    )
    parser.add_argument(
        "--video", type=str,
        default=str(script_dir / "Videos" / "GH069860.MP4"),
        help="Path to the video file.",
    )
    parser.add_argument(
        "--frame_nr", type=int, default=5000,
        help="Frame number to extract (default: 5000).",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(script_dir / "countline_visualisation_GH069860.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    # Extract frame
    print(f"Extracting frame #{args.frame_nr} from: {args.video}")
    frame = extract_frame(args.video, args.frame_nr)
    h, w = frame.shape[:2]

    # Counting line at vertical midpoint
    line_y = h / 2.0
    print(f"Video size: {w}×{h} px  →  Counting line at Y = {line_y:.0f} px")

    # Visualise
    visualize(frame, line_y, args.output, args.frame_nr)


if __name__ == "__main__":
    main()
