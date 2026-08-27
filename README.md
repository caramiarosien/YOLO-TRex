# YOLO–TRex Bat Counting

This repository contains the individual Python scripts used for the video-based bat-counting analysis in the bachelor thesis.

The scripts are organised in the order in which the analysis was performed. Raw videos, trained models, TRex output files, reference tables and final results are not included.

## Scripts

### Preparation

`preparation/extract_video_frames.py` extracts one frame per second from MP4 videos with FFmpeg. It is used to create still images for video inspection and preparation.

`preparation/analyze_video_brightness.py` samples video frames, calculates mean grayscale brightness and writes the brightness measurements to a CSV file.

### Counting

`counting/visualize_counting_line.py` extracts a selected video frame and draws the horizontal counting line on it. This is used to visually check the position of the counting line.

`counting/countline_bat_crossings.py` reads TRex trajectory files, applies the trajectory-quality filters and counts trajectories that cross the horizontal counting line. It can process one video or a directory of tracking results.

### Threshold analysis

The following scripts form the threshold-sensitivity analysis. They use explicit input files and do not silently clean, normalise or exclude observations.

1. `threshold_analysis/01_audit_threshold_inputs.py` checks the completeness and metadata of the three TRex threshold runs.
2. `threshold_analysis/02_extract_net_counts.py` extracts reproducible net crossing counts and writes input hashes and run metadata.
3. `threshold_analysis/03_analyze_threshold_sensitivity.py` computes the approved threshold-sensitivity summaries for the 80-snippet population.
4. `threshold_analysis/04_define_reference_population.py` marks the documented estimated reference counts and defines the 78-snippet comparison population.
5. `threshold_analysis/05_evaluate_trex_against_manual_reference.py` calculates the approved TRex-versus-manual-reference metrics for the 78-snippet population.
6. `threshold_analysis/06_compare_trex_with_koger.py` performs the pre-specified comparison between TRex and the raw Koger reference counts.

The seventh validation and data-freeze script is intentionally not part of this public code collection. It belongs to the internal project audit workflow rather than the individual analysis scripts.

## Running the scripts

The scripts are intended to be run individually. Paths and analysis parameters are supplied through command-line arguments or configuration files, depending on the script. Replace example paths with paths on the local machine before running them.

For the counting scripts, the required environment includes Python 3.10 or newer, NumPy, pandas, OpenCV and FFmpeg. TRex and its model files are separate prerequisites and are not distributed in this repository.

Example commands:

```bash
python preparation/extract_video_frames.py
python preparation/analyze_video_brightness.py

python counting/countline_bat_crossings.py --test
python counting/visualize_counting_line.py --help
```

The threshold-analysis scripts document their required inputs in their module docstrings. They are designed to be run in sequence after the corresponding input files have been prepared.

## Scope and data protection

This repository contains code only. Do not add raw videos, personal data, trained model weights, local environment folders, credentials or unpublished analysis outputs.

The code is provided to document the analysis workflow and to support reproducibility of the reported method.
