# YOLO–T-ReX Bat Counting Pipeline

This repository contains a research pipeline for detecting, tracking, and counting flying bats in video recordings. A YOLO model provides detections inside T-ReX, T-ReX exports individual trajectories, and a virtual horizontal countline is used to estimate directional bat counts.

The current public scope is intentionally limited to the preparation and final counting stages. Training data, model evaluation experiments, raw videos, model weights, tracking outputs, and legacy methods are not part of this initial version.

## Pipeline overview

```text
MP4 videos
    │
    ├── optional frame extraction
    │
    ├── brightness analysis and categorization
    │
    ├── YOLO detection within T-ReX
    │       └── per-track .npz files
    │
    ├── countline crossing analysis
    │       ├── per-video crossing CSV files
    │       └── daily count summaries
    │
    └── comparison with manual and reference counts
```

## Included scripts

```text
Preparation/
├── analyze_video_brightness.py
├── categorize_videos_by_brightness.py
├── extract_frames_batch_parallel.py
├── run_trex_batch.py

Final-Method/
├── countline_bat_crossings.py
├── compare_counting_methods.py
└── visualize_counting_line.py
```

### Preparation

`extract_frames_batch_parallel.py`

Recursively finds MP4 files and uses FFmpeg to extract one high-quality JPEG frame per second. Videos are processed in parallel, and frames are written to a `<video_name>_frames` directory next to each source video.

`analyze_video_brightness.py`

Samples frames at a configurable interval, calculates mean grayscale brightness, assigns each video to a brightness category, and writes the results to CSV.

Default categories:

- `dark`: brightness ≤ 35
- `medium`: brightness 36–85
- `light`: brightness > 85

`categorize_videos_by_brightness.py`

Calculates video brightness and moves each MP4 and its associated files into `dark`, `mediumdark`, or `light` subdirectories. The script always presents a dry-run preview and asks for confirmation before moving files.

`run_trex_batch.py`

Recursively finds MP4 files in brightness-category directories and runs T-ReX with the corresponding settings file and YOLO model. The current tracking confidence threshold is `0.3`. T-ReX output is placed under a `thre_03` directory and renamed to identify the source video.

### Final method

`countline_bat_crossings.py`

Loads T-ReX trajectory files, applies trajectory-quality filters, and counts bats whose trajectories cross the horizontal center of the frame. The implementation follows a start-versus-end countline rule: a trajectory is counted when its first and last valid Y positions lie on opposite sides of the line.

Direction convention in screen coordinates:

- `+1`: upward crossing, interpreted as forward/away from the colony
- `-1`: downward crossing, interpreted as backward/toward the colony

Default quality filters:

- minimum trajectory lifespan: 5 frames
- minimum straight-line movement: 50 pixels
- countline position: half of the video height
- maximum of one crossing per trajectory segment

`compare_counting_methods.py`

Combines YOLO–T-ReX count summaries, manual counts, reference-method counts, and brightness measurements. It builds a consolidated comparison table and calculates bias, MAE, RMSE, relative error, Pearson correlation, and Spearman correlation overall and by brightness category.

`visualize_counting_line.py`

Uses FFmpeg to extract a selected video frame and draws the horizontal countline for visual verification.

## Requirements

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/) available on the system `PATH`
- T-ReX installed separately
- a T-ReX-compatible YOLO model
- T-ReX settings for the `dark`, `mediumdark`, and `light` categories

Python packages used by the included scripts:

```text
matplotlib
numpy
opencv-python
openpyxl
pandas
scipy
```

Create a virtual environment outside the repository or in a directory ignored by Git. Do not commit a local environment such as `yolo_env/`.

Example installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install matplotlib numpy opencv-python openpyxl pandas scipy
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Configuration

The scripts currently use configuration constants near the top of each file. Update these values before running the pipeline:

| Script | Values to configure |
|---|---|
| `extract_frames_batch_parallel.py` | `SEARCH_DIR`, `MAX_WORKERS` |
| `analyze_video_brightness.py` | `VIDEO_DIR`, `OUTPUT_CSV`, brightness thresholds |
| `categorize_videos_by_brightness.py` | `BASE_DIR`, brightness thresholds |
| `run_trex_batch.py` | `BASE_DIR`, `MODEL`, `SETTINGS_MAP`, `CONDA_PREFIX`, `TREX_PATH` |
| `compare_counting_methods.py` | input summary, reference, brightness, and output paths |

Paths in the current research scripts are local examples and are not portable defaults. Replace them with paths valid on the target machine. Raw videos, trained models, T-ReX settings, and reference datasets must be supplied separately.

## Usage

Run all commands from the repository root.

### 1. Optional frame extraction

After setting `SEARCH_DIR`:

```bash
python Preparation/extract_frames_batch_parallel.py
```

### 2. Analyze brightness

After setting `VIDEO_DIR` and `OUTPUT_CSV`:

```bash
python Preparation/analyze_video_brightness.py
```

### 3. Categorize videos by brightness

After setting `BASE_DIR`:

```bash
python Preparation/categorize_videos_by_brightness.py
```

Review the dry-run summary carefully. The script only moves files after explicit confirmation.

### 4. Run YOLO detection and T-ReX tracking

After configuring the T-ReX executable, model, input directory, and settings paths:

```bash
python Preparation/run_trex_batch.py
```

### 5. Count crossings for one video

```bash
python Final-Method/countline_bat_crossings.py \
  --data_dir /path/to/video/tracks \
  --prefix video_name \
  --output /path/to/crossings_video_name.csv
```

The input directory must contain T-ReX files named `<video_name>_id<number>.npz` or the corresponding CSV files.

### 6. Count all videos in a directory tree

```bash
python Final-Method/countline_bat_crossings.py \
  --batch_dir /path/to/tracking/results \
  --day 16Nov \
  --summary_output /path/to/counting-16Nov-summary.csv
```

Useful options:

```text
--min_frames NUMBER
--min_distance PIXELS
--no_quality_filter
--flip_direction
--format {auto,npz,csv}
```

Run the built-in countline tests with:

```bash
python Final-Method/countline_bat_crossings.py --test
```

### 7. Visualize the countline

```bash
python Final-Method/visualize_counting_line.py \
  --video /path/to/video.mp4 \
  --frame_nr 5000 \
  --output /path/to/countline.png
```

### 8. Compare counting methods

After configuring the input paths in `compare_counting_methods.py`:

```bash
python Final-Method/compare_counting_methods.py
```

## T-ReX input format used by the countline analysis

For NPZ input, the counting script expects the following T-ReX fields:

```text
X#wcentroid
Y#wcentroid
frame
video_size
tracklets    # optional segment table
```

Files must follow this naming convention:

```text
<video_name>_id<number>.npz
```

## Outputs

The pipeline can generate:

- extracted JPEG frames
- video-brightness CSV files
- brightness-category directories
- T-ReX `.npz` or `.csv` trajectory files
- per-video `crossings_<video_name>.csv` files
- daily `counting-<day>-summary.csv` files
- a master comparison CSV and printed agreement metrics
- countline visualization images

Generated outputs should not be committed to the source repository unless they are small, curated examples intended for documentation.

## Data and model availability

This repository does not distribute raw field videos, complete tracking outputs, trained model weights, manual-count spreadsheets, or model-analysis datasets. These assets may be large, contain project-specific metadata, or have separate access and licensing requirements.

## Current status

This is research code prepared for an initial public release. The core processing logic is present, but configuration is still file-based and contains machine-specific example paths. A future revision should move configuration into command-line arguments or a shared configuration file and add small anonymized fixtures for automated integration tests.

## License and citation

License and citation information will be added before the public release. The countline logic is documented in the source as following the start-versus-end approach used by Koger et al. (2023); a complete bibliographic reference should be added here before publication.
