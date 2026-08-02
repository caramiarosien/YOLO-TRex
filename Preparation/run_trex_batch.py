#!/usr/bin/env python3
import os
import subprocess

# Base directory containing videos to process in batch mode
BASE_DIR = "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/16Nov"

# Output fields
ofields = '[["SPEED",["WCENTROID"]],["X",["WCENTROID"]],["Y",["WCENTROID"]],["blob_height",["RAW"]],["blob_width",["RAW"]],["detection_class",["RAW"]],["detection_p",["RAW"]],["frame",["RAW"]],["midline_length",["RAW"]],["time",["RAW"]],["timestamp",["RAW"]],["tracklet_id",["RAW","WCENTROID"]]]'

MODEL = "/Users/cara/Desktop/BA/Method_YOLO/best-new-model-yolo26s-kasanka-v6-3-2.pt"

# Mapping: folder prefix → settings file
SETTINGS_MAP = {
    "mediumdark": "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/default-settings/mediumdark.settings",
    "light":      "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/default-settings/light-crowded-default.settings",
    "dark":       "/Users/cara/Desktop/BA/Data/Analysis_BA_videos/default-settings/dark-default.settings",
}

# Set CONDA_PREFIX so TRex can find its resources (fonts, scripts, etc.)
os.environ["CONDA_PREFIX"] = "/Users/cara/miniforge3/envs/track"
TREX_PATH = "/Users/cara/miniforge3/envs/track/bin/trex"

# Recursively find all .mp4 files in BASE_DIR that sit inside a subfolder
# whose name starts with one of the defined prefixes (case-insensitive).
mp4_files = []  # list of (filepath, settings_file)
for root, dirs, files in os.walk(BASE_DIR):
    path_parts = os.path.normpath(root).split(os.sep)
    # Check each path component against the known prefixes
    matched_settings = None
    for part in path_parts:
        for prefix, settings in SETTINGS_MAP.items():
            if part.lower().startswith(prefix):
                matched_settings = settings
                break
        if matched_settings is not None:
            break
    if matched_settings is None:
        continue
    for file in files:
        if file.lower().endswith('.mp4'):
            mp4_files.append((os.path.join(root, file), matched_settings))

for file, settings_file in mp4_files:
    folder = os.path.dirname(file)

    if settings_file is None:
        print(f"[SKIP] No settings file defined for: {file}")
        continue

    print(f"Processing file: {file} (in {folder})")
    print(f"  Using settings: {settings_file}")

    # base name without extension
    filename = os.path.basename(file)
    if filename.lower().endswith('.mp4'):
        base_no_ext = filename[:-4]
    else:
        base_no_ext = filename

    # drop the leading '#' ONLY for pv/settings
    base_nohash = base_no_ext.lstrip('#')

    # Threshold sweep
    for thr in [0.3]:
        # make prefix thre_03 / thre_04 / thre_05
        thr_tag = str(thr).replace('.', '')
        output_prefix = f"thre_{thr_tag}"

        cmd = [
            TREX_PATH,
            "-i", file,
            "-s", settings_file,
            "-task", "convert",
            "-detect_model", MODEL,
            "-output_min_frames", "10",
            "-meta_source_path", file,
            "-output_tracklet_images", "true",
            "-output_fields", ofields,
            "-track_conf_threshold", str(thr),
            "-output_prefix", output_prefix,
            "-auto_quit"
        ]

        subprocess.run(cmd)

        # Rename the output folder from "data" to "data-<videoname>"
        # TRex creates: folder/output_prefix/data/
        data_dir = os.path.join(folder, output_prefix, "data")
        renamed_dir = os.path.join(folder, output_prefix, f"data-{base_nohash}")
        if os.path.isdir(data_dir):
            os.rename(data_dir, renamed_dir)
            print(f"  Renamed '{data_dir}' → '{renamed_dir}'")
        else:
            print(f"  [WARNING] data folder not found: '{data_dir}'")
