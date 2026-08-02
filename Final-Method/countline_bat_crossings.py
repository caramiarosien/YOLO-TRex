#!/usr/bin/env python3
"""
countline_bat_crossings.py — Virtual Counting Line for Bat Population Analysis
===============================================================================

Processes T-ReX tracking files (.npz or .csv) and counts bats crossing
a horizontal midline using frame-accurate trajectory analysis. Includes
spatial and temporal filtering to remove noise (dust, insects).

Logic:
    - Loads all tracks, respecting the per-tracklet segment table
      (``tracklets`` key in .npz) so that each sub-segment is treated
      as an independent trajectory.
    - Filters out inf values (frames where the animal was not detected).
    - Detects crossings of the midline per segment.
    - Counting logic follows Koger et al. (2023): a track is counted
      as one crossing if its first and last Y positions are on opposite
      sides of the countline. The line is placed at int(video_h / 2).
    - Returns at most 1 crossing per segment (forward or backward).
    - Output is a CSV file containing Video_Name, Tracklet_ID, and Frame_Number.

Direction convention (screen coordinates, Y increases downward):
    +1  (forward / away from colony):  moves from high-Y (below) to low-Y (above)
    -1  (backward / toward colony):    moves from low-Y (above) to high-Y (below)

Date:    2026-06-13
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# ===========================================================================
# ⚙️  CONFIGURATION — edit default paths here
# ===========================================================================
DEFAULT_DATA_DIR = "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/17Nov-brightness-processed/"
DEFAULT_PREFIX = ""
DEFAULT_OUTPUT_DIR = "/Users/cara/Desktop/BA/Method_YOLO/Final-Output"



# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_all_tracks(data_dir: str, file_prefix: str = None, split_segments: bool = False) -> list[dict]:
    """Load per-tracklet .npz files and return one dict per track/segment.

    If ``split_segments`` is True and the .npz contains a ``tracklets``
    key, each row becomes a separate dict so that downstream counting
    treats each sub-segment as an independent trajectory.

    Otherwise, or if the ``tracklets`` key is absent, all detections
    in the file are returned as a single dict (matching Ben Koger's
    original track representation).

    ``inf`` values in Y/X (frames where the animal was not detected)
    are dropped.

    Returns
    -------
    list[dict]
        Each dict has keys: id, y, x, frame, video_size, video_name.
    """
    if file_prefix:
        pattern = os.path.join(data_dir, f"{file_prefix}_id*.npz")
    else:
        pattern = os.path.join(data_dir, "*_id*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        sub_data = os.path.join(data_dir, "data")
        if os.path.isdir(sub_data):
            if file_prefix:
                pattern = os.path.join(sub_data, f"{file_prefix}_id*.npz")
            else:
                pattern = os.path.join(sub_data, "*_id*.npz")
            files = sorted(glob.glob(pattern))
            if files:
                data_dir = sub_data

    if not files:
        raise FileNotFoundError(
            f"No .npz files matching '{pattern}' found. "
            f"Check --data_dir and --prefix arguments."
        )

    tracks = []
    for fpath in files:
        basename = Path(fpath).stem
        parts = basename.rsplit("_id", maxsplit=1)
        if len(parts) < 2:
            print(f"[WARN] Skipping file with unexpected name: {fpath}")
            continue
        vname, id_str = parts[0], parts[1]
        try:
            track_id = int(id_str)
        except ValueError:
            print(f"[WARN] Skipping file with non-numeric ID: {fpath}")
            continue

        data = np.load(fpath, allow_pickle=True)

        # Centroid keys:
        y_all = data["Y#wcentroid"].astype(float)
        x_all = data["X#wcentroid"].astype(float)
        frame_all = data["frame"].astype(float)
        video_size = data["video_size"]

        # --- Segment-aware loading ------------------------------------
        # Segment-aware splitting:
        if split_segments and "tracklets" in data:
            segments = data["tracklets"]  # shape (K, 2), uint32
            for seg_idx, (start_idx, end_idx) in enumerate(segments):
                # start_idx and end_idx are indices into the arrays
                # (inclusive on both ends)
                seg_y = y_all[start_idx: end_idx + 1]
                seg_x = x_all[start_idx: end_idx + 1]
                seg_f = frame_all[start_idx: end_idx + 1]

                # Drop inf values (frames where the animal was not detected)
                finite_mask = np.isfinite(seg_y) & np.isfinite(seg_x)
                seg_y = seg_y[finite_mask]
                seg_x = seg_x[finite_mask]
                seg_f = seg_f[finite_mask]

                if len(seg_y) == 0:
                    continue  # skip empty segments

                tracks.append({
                    "id":         f"{track_id}_seg{seg_idx}",
                    "y":          seg_y,
                    "x":          seg_x,
                    "frame":      seg_f,
                    "video_size": video_size,
                    "video_name": vname,
                })
        else:
            # Fallback: no segment table — treat whole file as one trajectory
            finite_mask = np.isfinite(y_all) & np.isfinite(x_all)
            tracks.append({
                "id":         track_id,
                "y":          y_all[finite_mask],
                "x":          x_all[finite_mask],
                "frame":      frame_all[finite_mask],
                "video_size": video_size,
                "video_name": vname,
            })

    return tracks


def load_all_tracks_csv(data_dir: str, file_prefix: str = None) -> list[dict]:
    if file_prefix:
        pattern = os.path.join(data_dir, f"{file_prefix}_id*.csv")
    else:
        pattern = os.path.join(data_dir, "*_id*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        sub_data = os.path.join(data_dir, "data")
        if os.path.isdir(sub_data):
            if file_prefix:
                pattern = os.path.join(sub_data, f"{file_prefix}_id*.csv")
            else:
                pattern = os.path.join(sub_data, "*_id*.csv")
            files = sorted(glob.glob(pattern))
            if files:
                data_dir = sub_data

    if not files:
        raise FileNotFoundError(
            f"No CSV files matching '{pattern}' found. "
            f"Check --data_dir and --prefix arguments."
        )

    tracks = []
    for fpath in files:
        basename = Path(fpath).stem
        parts = basename.rsplit("_id", maxsplit=1)
        if len(parts) < 2:
            print(f"[WARN] Skipping file with unexpected name: {fpath}")
            continue
        vname, id_str = parts[0], parts[1]
        try:
            track_id = int(id_str)
        except ValueError:
            print(f"[WARN] Skipping file with non-numeric ID: {fpath}")
            continue

        df = pd.read_csv(fpath)
        x_col = [c for c in df.columns if c.startswith("X#wcentroid")][0]
        y_col = [c for c in df.columns if c.startswith("Y#wcentroid")][0]

        y_arr = df[y_col].values.astype(float)
        x_arr = df[x_col].values.astype(float)
        frame_arr = df["frame"].iloc[:, 0].values.astype(float) if isinstance(df["frame"], pd.DataFrame) else df["frame"].values.astype(float)

        tracks.append({
            "id":    track_id,
            "y":     y_arr,
            "x":     x_arr,
            "frame": frame_arr,
            "video_name": vname,
        })

    return tracks


def load_npy_results(npy_path: str) -> pd.DataFrame:
    """
    Loads a pre-processed .npy crossing file (Ben Koger pipeline format).
    The file contains a dict with keys: date, camera, frames, ids,
    direction, mean_wing, darkness, track_length.
    Returns a DataFrame with one row per crossing event.
    """
    data = np.load(npy_path, allow_pickle=True).item()

    df = pd.DataFrame({
        "Video_Name":    f"{data['date']}-{data['camera']}",
        "Tracklet_ID":   data["ids"],
        "Frame_Number":  data["frames"],
        "Direction":     data["direction"],
        "Mean_Wing":     data["mean_wing"],
        "Darkness":      data["darkness"],
        "Track_Length":  data["track_length"],
    })
    return df


# ---------------------------------------------------------------------------
# 2. Counting Line Definition
# ---------------------------------------------------------------------------

def get_counting_line_y(video_height: float) -> float:
    """Counting line at the vertical centre of the frame.

    Matches Koger et al. (2023): int(frame_height / 2).
    """
    return int(video_height) / 2.0


# ---------------------------------------------------------------------------
# 3. Track Filtering (Quality Control)
# ---------------------------------------------------------------------------

def is_valid_track(track: dict, min_frames: int = 5, min_distance_px: float = 50.0) -> bool:
    """
    Evaluates whether a tracklet belongs to a real bat or is false-positive
    (e.g., dust, insects) based on its lifespan and spatial movement.
    """
    valid_mask = np.isfinite(track["x"]) & np.isfinite(track["y"])

    # We need at least 2 points to calculate a distance
    if np.sum(valid_mask) < 2:
        return False

    valid_frames = track["frame"][valid_mask]
    lifespan = valid_frames[-1] - valid_frames[0]

    valid_x = track["x"][valid_mask]
    valid_y = track["y"][valid_mask]

    # Calculate straight-line distance from first to last detection
    distance = np.sqrt((valid_x[-1] - valid_x[0])**2 + (valid_y[-1] - valid_y[0])**2)

    # Track must exist for a soft minimum of frames AND have moved a minimum distance
    return (lifespan >= min_frames) and (distance >= min_distance_px)


# ---------------------------------------------------------------------------
# 4. Core Counting Logic (Koger's start-vs-end algorithm)
# ---------------------------------------------------------------------------

def detect_tracklet_crossings(
    track: dict,
    line_y: float,
    forward_is_upward: bool = True,
) -> list[dict]:
    """Ben Koger's start-vs-end crossing detection.

    A trajectory is counted as one crossing if its first and last
    Y positions are on opposite sides of ``line_y``.  Returns at
    most one crossing event per call.

    Parameters
    ----------
    track : dict
        Must contain keys ``y`` and ``frame``.  ``y`` and ``frame`` are
        1-D arrays with inf-filtered finite values (done by
        ``load_all_tracks``).
    line_y : float
        Y-pixel value of the countline.
    forward_is_upward : bool, default True
        If True, Y decreasing (moving up in screen) is +1 (away from
        colony).  If False, Y increasing (moving down) is +1.

    Returns
    -------
    list[dict]
        Zero or one ``{"frame": int, "direction": +1 | -1}``.
    """
    y = track["y"]

    # Safety filter (CSV paths may still contain inf)
    valid_mask = np.isfinite(y)
    valid_y = y[valid_mask]
    valid_frames = track["frame"][valid_mask]

    # Need at least two points
    if len(valid_y) < 2:
        return []

    y0 = float(valid_y[0])
    yN = float(valid_y[-1])

    # Direction sign: +1 = away from colony = Y decreasing (going up)
    sign = +1 if forward_is_upward else -1

    # Forward: starts on/below line, ends on/above line
    if y0 >= line_y and yN <= line_y:
        # np.argmax finds the first True in the boolean mask, i.e. the first
        # frame where the bat is actually on the target (upper) side of the
        # line.  The original np.argmin always returned 0 (the first frame of
        # the track) because argmin on a boolean array picks the first False,
        # which is index 0 when the track starts below the line.
        mask = valid_y <= line_y
        crossing_frame = int(np.argmax(mask)) + int(valid_frames[0])
        return [{"frame": crossing_frame, "direction": sign * +1}]

    # Backward: starts on/above line, ends on/below line
    if y0 <= line_y and yN >= line_y:
        # Same fix: argmax gives the first frame the bat is on the lower side.
        mask = valid_y >= line_y
        crossing_frame = int(np.argmax(mask)) + int(valid_frames[0])
        return [{"frame": crossing_frame, "direction": sign * -1}]

    return []


def count_crossings(
    tracks: list[dict],
    line_y: float,
    forward_is_upward: bool = True,
) -> pd.DataFrame:
    """Run crossing detection on all tracks and return a DataFrame.

    The output has one row per crossing event, with columns
    ``Video_Name``, ``Tracklet_ID``, ``Frame_Number``, ``Direction``.
    """
    all_events = []
    for track in tracks:
        crossings = detect_tracklet_crossings(
            track, line_y,
            forward_is_upward=forward_is_upward,
        )
        for c in crossings:
            all_events.append({
                "Video_Name":   track["video_name"],
                "Tracklet_ID":  track["id"],
                "Frame_Number": c["frame"],
                "Direction":    c["direction"],
            })
    return pd.DataFrame(all_events)


# ---------------------------------------------------------------------------
# 5. Batch Processing (Multi-Folder / Day Analysis)
# ---------------------------------------------------------------------------

def process_batch_directory(
    batch_dir: str,
    day_name: str = None,
    output_csv: str = None,
    split_segments: bool = True,
    min_frames: int = 5,
    min_distance: float = 50.0,
    no_quality_filter: bool = False,
    flip_direction: bool = False,
    save_individual_csvs: bool = True,
) -> pd.DataFrame:
    """Recursively finds all video data subdirectories (containing .npz files) inside
    batch_dir, analyzes all videos, and generates an aggregate day summary CSV.

    Columns in summary CSV:
    - Day (e.g. '16Nov' or '17Nov')
    - Video name
    - Counting forward
    - Counting backward
    - Nett count
    """
    batch_path = Path(batch_dir).resolve()
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    # Auto-detect Day label if not specified
    if not day_name:
        match = re.search(r"(16\s*Nov|17\s*Nov)", str(batch_path), re.IGNORECASE)
        if match:
            day_name = match.group(1).replace(" ", "")
        else:
            day_name = batch_path.name

    print(f"==================================================")
    print(f"🔍 BATCH PROCESSING: {batch_path}")
    print(f"📅 Day Label: {day_name}")
    print(f"==================================================")

    # Find all subdirectories containing .npz track files
    video_dirs = []
    for dirpath, dirnames, filenames in os.walk(batch_path):
        npz_files = [f for f in filenames if f.endswith(".npz") and "_id" in f]
        if npz_files:
            video_dirs.append(dirpath)

    video_dirs = sorted(video_dirs)
    if not video_dirs:
        print(f"[WARN] No video data directories containing *_id*.npz found in {batch_path}")
        return pd.DataFrame()

    print(f"Found {len(video_dirs)} video data directories to analyze.")

    summary_rows = []
    for idx, vdir in enumerate(video_dirs, 1):
        vdir_name = Path(vdir).name
        print(f"\n[{idx}/{len(video_dirs)}] Processing video directory: {vdir_name}")

        try:
            tracks = load_all_tracks(vdir, split_segments=split_segments)
            if not tracks:
                print(f"[WARN] No valid tracks loaded from {vdir}")
                continue

            initial_count = len(tracks)
            if not no_quality_filter:
                valid_tracks = [t for t in tracks if is_valid_track(t, min_frames=min_frames, min_distance_px=min_distance)]
            else:
                valid_tracks = tracks

            filtered_count = len(valid_tracks)

            # Determine video name
            vname = tracks[0].get("video_name")
            if not vname:
                vname = vdir_name.replace("data-", "").replace("data_", "").replace("data", "")

            # Determine video height & counting line
            if "video_size" in tracks[0] and tracks[0]["video_size"][1] > 0:
                video_h = tracks[0]["video_size"][1]
            else:
                all_y = np.concatenate([t["y"] for t in tracks])
                all_y = all_y[np.isfinite(all_y)]
                if len(all_y) == 0:
                    print(f"[WARN] Skipping {vname}: No finite Y values found.")
                    continue
                video_h = int(np.ceil((np.max(all_y) + 1) / 8) * 8)

            line_y = get_counting_line_y(video_h)
            df_results = count_crossings(valid_tracks, line_y, forward_is_upward=not flip_direction)

            if not df_results.empty:
                forward = len(df_results[df_results["Direction"] == 1])
                backward = len(df_results[df_results["Direction"] == -1])
            else:
                forward = backward = 0

            net_count = forward - backward

            print(f"   -> Video name: {vname}")
            print(f"   -> Tracks: {filtered_count} valid (of {initial_count} total)")
            print(f"   -> Counting forward: {forward} | Counting backward: {backward} | Nett count: {net_count}")

            # Optionally save individual video crossing details CSV
            if save_individual_csvs:
                out_dir = Path(vdir) / "counting"
                out_dir.mkdir(parents=True, exist_ok=True)
                df_results.to_csv(out_dir / f"crossings_{vname}.csv", index=False)

            summary_rows.append({
                "Day": day_name,
                "Video name": vname,
                "Counting forward": forward,
                "Counting backward": backward,
                "Nett count": net_count,
            })

        except Exception as e:
            print(f"[ERROR] Failed processing {vdir}: {e}")

    df_summary = pd.DataFrame(summary_rows)

    if output_csv is None:
        out_dir = Path(DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = str(out_dir / f"counting-{day_name}-summary.csv")

    df_summary.to_csv(output_csv, index=False)
    print(f"\n==================================================")
    print(f"✅ BATCH ANALYSIS COMPLETE")
    print(f"📊 Processed {len(df_summary)} videos for Day '{day_name}'.")
    print(f"💾 Saved day summary CSV to: {output_csv}")
    print(f"==================================================")

    return df_summary


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CountLine — Virtual counting line for T-ReX bat tracking data. "
                    "Crossing logic follows Koger et al. (2023): a track is counted "
                    "as one crossing if its first and last Y positions are on opposite "
                    "sides of the countline. The line is placed at int(video_h / 2).",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the per-individual .npz or .csv files.",
    )
    parser.add_argument(
        "--batch_dir", type=str, default=None,
        help="Root directory containing subdirectories (e.g. data<video_name>) to analyze in batch mode across all subfolders.",
    )
    parser.add_argument(
        "--day", type=str, default=None,
        help="Day label for summary CSV (e.g. '16Nov' or '17Nov'). Auto-detected from batch_dir if not specified.",
    )
    parser.add_argument(
        "--summary_output", type=str, default=None,
        help="Output CSV path for the batch day summary file (default: summary_<day>.csv in batch_dir).",
    )
    parser.add_argument(
        "--prefix", type=str,
        default=DEFAULT_PREFIX,
        help="Filename prefix for the tracking files (e.g. 'GH059860', 'thre_03').",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the crossing events CSV file.",
    )
    parser.add_argument(
        "--flip_direction", action="store_true",
        help="Flip the +1/-1 convention (default: upward = forward = +1).",
    )
    parser.add_argument(
        "--video_height", type=int, default=None,
        help="Video height in pixels (required for CSV input without video_size).",
    )
    parser.add_argument(
        "--format", type=str, choices=["npz", "csv", "auto"], default="auto",
        help="Input format: 'npz', 'csv', or 'auto' (default: auto-detect).",
    )
    parser.add_argument(
        "--npy", type=str, default=None,
        help="Path to a pre-processed .npy crossing file (Ben Koger format). "
             "When provided, --data_dir and --prefix are ignored.",
    )
    parser.add_argument(
        "--min_frames", type=int, default=5,
        help="Minimum lifespan in frames for a track to be considered valid (default: 5).",
    )
    parser.add_argument(
        "--min_distance", type=float, default=50.0,
        help="Minimum straight-line distance (in pixels) a track must move to be valid (default: 50.0).",
    )
    parser.add_argument(
        "--no_quality_filter", action="store_true",
        help="Disable the min_frames/min_distance quality filter (count every segment).",
    )
    parser.add_argument(
        "--split_segments", action="store_true", default=True,
        help="Split tracks into independent segments at tracking gaps using T-ReX tracklets "
             "(default: True, matching Ben Koger's segment-based benchmark counting). "
             "Pass --no_split_segments to disable and treat each full tracklet as one trajectory.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run the built-in self-tests.",
    )

    args = parser.parse_args()

    if args.test:
        run_self_test()
        return

    if args.batch_dir:
        process_batch_directory(
            batch_dir=args.batch_dir,
            day_name=args.day,
            output_csv=args.summary_output,
            split_segments=args.split_segments,
            min_frames=args.min_frames,
            min_distance=args.min_distance,
            no_quality_filter=args.no_quality_filter,
            flip_direction=args.flip_direction,
        )
        return

    prefix_filter = args.prefix if (args.prefix and args.prefix.strip() != "") else None
    if prefix_filter and prefix_filter.endswith("_"):
        prefix_filter = prefix_filter[:-1]

    if args.output is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix_str = prefix_filter if prefix_filter else "all"
        args.output = str(output_dir / f"crossings_{prefix_str}.csv")
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # === .npy shortcut: file already contains crossing results ===
    if args.npy:
        print(f"Loading pre-processed .npy file: {args.npy}")
        df_results = load_npy_results(args.npy)

        forward  = len(df_results[df_results["Direction"] ==  1])
        backward = len(df_results[df_results["Direction"] == -1])
        total    = len(df_results)

        print(f"Total crossing events: {total}")
        print(f"Forward (+1): {forward} | Backward (-1): {backward}")
        print(f"Net count: {forward - backward}")

        npy_stem = Path(args.npy).stem
        output_path = str(Path(args.output).parent / f"crossings_{npy_stem}.csv")
        df_results.to_csv(output_path, index=False)
        print(f"Saved {total} crossing events to {output_path}")
        return

    fmt = args.format
    prefix_pattern = f"{prefix_filter}_id*" if prefix_filter else "*_id*"
    if fmt == "auto":
        npz_files = glob.glob(os.path.join(args.data_dir, f"{prefix_pattern}.npz"))
        csv_files = glob.glob(os.path.join(args.data_dir, f"{prefix_pattern}.csv"))
        if not npz_files and not csv_files:
            sub_data = os.path.join(args.data_dir, "data")
            if os.path.isdir(sub_data):
                npz_sub = glob.glob(os.path.join(sub_data, f"{prefix_pattern}.npz"))
                csv_sub = glob.glob(os.path.join(sub_data, f"{prefix_pattern}.csv"))
                if npz_sub or csv_sub:
                    args.data_dir = sub_data
                    npz_files, csv_files = npz_sub, csv_sub

        # Auto-switch to batch mode if data_dir contains video subdirectories with .npz files
        if not npz_files and not csv_files:
            batch_subdirs = []
            for dirpath, dirnames, filenames in os.walk(args.data_dir):
                if any(f.endswith(".npz") and "_id" in f for f in filenames):
                    batch_subdirs.append(dirpath)
            if batch_subdirs:
                print(f"[INFO] No tracking files found directly in '{args.data_dir}', "
                      f"but found {len(batch_subdirs)} video subdirectories. "
                      f"Automatically switching to BATCH processing mode.\n")
                process_batch_directory(
                    batch_dir=args.data_dir,
                    day_name=args.day,
                    output_csv=args.summary_output,
                    split_segments=args.split_segments,
                    min_frames=args.min_frames,
                    min_distance=args.min_distance,
                    no_quality_filter=args.no_quality_filter,
                    flip_direction=args.flip_direction,
                )
                return

        if npz_files:
            fmt = "npz"
        elif csv_files:
            fmt = "csv"
        else:
            raise FileNotFoundError(
                f"No .npz or .csv files matching '{prefix_pattern}' "
                f"found in {args.data_dir}."
            )
        print(f"Auto-detected format: {fmt}")

    if fmt == "npz":
        print(f"Loading tracks from: {args.data_dir}/{prefix_pattern}.npz")
        tracks = load_all_tracks(args.data_dir, prefix_filter, split_segments=args.split_segments)
    else:
        print(f"Loading tracks from: {args.data_dir}/{prefix_pattern}.csv")
        tracks = load_all_tracks_csv(args.data_dir, prefix_filter)

    initial_tracks = tracks
    initial_track_count = len(initial_tracks)
    if not args.no_quality_filter:
        tracks = [t for t in tracks if is_valid_track(t, min_frames=args.min_frames, min_distance_px=args.min_distance)]
    filtered_track_count = len(tracks)
    dropped_tracks = initial_track_count - filtered_track_count

    if args.no_quality_filter:
        print(f"Quality filter: DISABLED (counting all {initial_track_count} segments)")
    else:
        print(f"Quality Filter: Dropped {dropped_tracks} short/static tracks (likely dust/noise).")
        print(f"Valid tracks remaining: {filtered_track_count}")

    # 1st Priority: Manually passed video height always takes precedence
    if args.video_height is not None:
        video_h = args.video_height
        video_w = "?"
    # 2nd Priority: Read from file metadata if valid (> 0)
    elif fmt == "npz" and initial_tracks and "video_size" in initial_tracks[0] and initial_tracks[0]["video_size"][1] > 0:
        video_size = initial_tracks[0]["video_size"]
        video_w, video_h = video_size[0], video_size[1]
    # 3rd Priority: Fallback to estimating video height from max track Y coordinate
    elif initial_tracks:
        all_y = np.concatenate([t["y"] for t in initial_tracks])
        all_y = all_y[np.isfinite(all_y)]
        if len(all_y) == 0:
            raise ValueError("No finite Y values found in tracks.")
        # Round up max Y to next multiple of 8 (common video dimension alignment)
        max_y = float(np.max(all_y))
        video_h = int(np.ceil((max_y + 1) / 8) * 8)
        video_w = "?"
        print(f"[INFO] video_size invalid (-1). Auto-estimated video height = {video_h} px "
              f"(from max Y = {max_y:.1f}). Use --video_height to override.")
    else:
        raise ValueError(
            "No tracks loaded and no --video_height provided. "
            "Cannot determine video dimensions."
        )
    line_y = get_counting_line_y(video_h)
    print(f"Counting line Y = {line_y:.1f} px  (video: {video_w} x {int(video_h)})")

    df_results = count_crossings(
        tracks, line_y,
        forward_is_upward=not args.flip_direction,
    )

    if not df_results.empty:
        forward = len(df_results[df_results["Direction"] == 1])
        backward = len(df_results[df_results["Direction"] == -1])
    else:
        forward = backward = 0

    print(f"Forward (+1): {forward} | Backward (-1): {backward}")
    print(f"Net count: {forward - backward}")

    output_path = args.output
    df_results.to_csv(output_path, index=False)
    print(f"Saved {len(df_results)} crossing events to {output_path}")


def run_self_test():
    import tempfile

    print("Running self-test…")
    d = tempfile.mkdtemp()

    # Test 1: load_all_tracks respects tracklets table + drops inf
    y = np.array(
        [100, 200,            # seg 0  (idx 0-1)
         np.inf,              # gap     (idx 2)
         50, 80, 90,          # seg 1  (idx 3-5)
         np.inf,              # gap     (idx 6)
         300,                 # seg 2  (idx 7)
         400, 500,            # seg 3  (idx 8-9)
         600, 700, 800, 900], # seg 4  (idx 10-13)
        dtype=np.float32,
    )
    x = np.arange(len(y), dtype=np.float32) * 10
    frame = np.arange(len(y), dtype=np.float32) + 100
    tracklets = np.array(
        [[0, 1], [3, 5], [7, 7], [8, 9], [10, 13]],
        dtype=np.uint32,
    )
    np.savez(
        os.path.join(d, "test_id402.npz"),
        **{
            "Y#wcentroid": y,
            "X#wcentroid": x,
            "frame": frame,
            "tracklets": tracklets,
            "cm_per_pixel": np.float64(0.0136),
            "frame_rate": np.float64(30.0),
            "video_size": np.array([1920, 1080]),
        },
    )

    # Test 1.1: load_all_tracks with split_segments=True (returns 5 segments)
    result = load_all_tracks(d, "test", split_segments=True)
    assert len(result) == 5, f"FAIL: got {len(result)} segments, expected 5"
    print(f"PASS: {len(result)} segments returned (expected 5)")

    total_points = sum(len(t["y"]) for t in result)
    assert total_points == 12, f"FAIL: expected 12 finite points, got {total_points}"
    print(f"PASS: {total_points} finite data points (2 inf dropped)")

    ids = [t["id"] for t in result]
    assert all(isinstance(i, str) and "_seg" in i for i in ids), \
        f"FAIL: segment IDs should contain '_seg', got {ids}"
    print(f"PASS: segment IDs look correct: {ids}")

    # Test 1.2: load_all_tracks with split_segments=False (default, returns 1 complete tracklet)
    result_fallback = load_all_tracks(d, "test", split_segments=False)
    assert len(result_fallback) == 1, f"FAIL: got {len(result_fallback)} segments, expected 1 when split_segments=False"
    print(f"PASS: fallback returns 1 tracklet (expected 1)")

    total_points_fallback = len(result_fallback[0]["y"])
    assert total_points_fallback == 12, f"FAIL: expected 12 finite points in fallback, got {total_points_fallback}"
    print(f"PASS: fallback contains {total_points_fallback} finite data points")

    # Test 2: Koger's start-vs-end algorithm
    # Case A: forward crossing (y goes 1000 -> 200, starts below line 712, ends above)
    track_a = {"y": np.array([1000.0, 200.0]), "frame": np.array([10.0, 11.0])}
    cr = detect_tracklet_crossings(track_a, line_y=712)
    assert len(cr) == 1 and cr[0]["direction"] == 1, \
        f"FAIL case A: expected 1 forward crossing, got {cr}"
    print(f"PASS case A (forward crossing 1000->200): {cr[0]}")

    # Case B: backward crossing (y goes 200 -> 1000)
    track_b = {"y": np.array([200.0, 1000.0]), "frame": np.array([10.0, 11.0])}
    cr = detect_tracklet_crossings(track_b, line_y=712)
    assert len(cr) == 1 and cr[0]["direction"] == -1, \
        f"FAIL case B: expected 1 backward crossing, got {cr}"
    print(f"PASS case B (backward crossing 200->1000): {cr[0]}")

    # Case C: same-side (y stays 800 -> 900)
    track_c = {"y": np.array([800.0, 900.0]), "frame": np.array([10.0, 11.0])}
    cr = detect_tracklet_crossings(track_c, line_y=712)
    assert cr == [], f"FAIL case C: expected no crossing, got {cr}"
    print(f"PASS case C (same-side 800->900): no crossing")

    # Case D: boundary on-the-line (y starts at line value 712, ends above)
    track_d = {"y": np.array([712.0, 100.0]), "frame": np.array([10.0, 11.0])}
    cr = detect_tracklet_crossings(track_d, line_y=712)
    assert len(cr) == 1 and cr[0]["direction"] == 1, \
        f"FAIL case D: expected 1 forward crossing (on-the-line inclusive), got {cr}"
    print(f"PASS case D (boundary 712->100): {cr[0]}")

    # Case E: verify that crossing_frame is the actual transition index, not
    # always the first frame (argmin regression test).
    # y crosses line_y=500 at index 2 (frame 12). argmin would wrongly return
    # frame 10 (index 0) because it picks the first False in the mask.
    track_e = {
        "y": np.array([900.0, 700.0, 400.0, 200.0]),
        "frame": np.array([10.0, 11.0, 12.0, 13.0]),
    }
    cr = detect_tracklet_crossings(track_e, line_y=500)
    assert len(cr) == 1, f"FAIL case E: expected 1 crossing, got {cr}"
    assert cr[0]["direction"] == 1, \
        f"FAIL case E: expected forward direction (+1), got {cr[0]['direction']}"
    assert cr[0]["frame"] == 12, (
        f"FAIL case E: crossing_frame should be 12 (first frame bat is above "
        f"line_y=500), got {cr[0]['frame']}. "
        f"This would be 10 if the argmin bug were still present."
    )
    print(f"PASS case E (crossing frame = {cr[0]['frame']}, expected 12 — argmin bug is fixed)")

    print("All self-tests passed.")


if __name__ == "__main__":
    main()
