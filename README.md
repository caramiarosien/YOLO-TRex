# Kasanka Bat Counting & Detection Pipeline

This repository contains the data preparation, mask generation, and YOLO formatting pipeline for tracking and counting flying foxes (bats) in Kasanka National Park.

The project is structured into sequentially organized modules that take raw video data, extract frames, generate and filter segmentation masks, and finally convert them into YOLO-formatted datasets for training computer vision models.

## 📁 Repository Structure

The project code is divided into four main chronological steps:

### 1. `data_extraction_and_sampling/`
Scripts to extract individual frames from raw MP4 camera recordings and strategically sample them to create a well-balanced dataset. Includes tools for proportional sampling, validation set isolation, and filtering out blurry images.

### 2. `mask_generation_and_noise_filtering/`
Computer vision scripts designed to create binary foreground masks of the bats. Crucially, this module contains physics and geometry-based filters to eliminate common artifacts like clouds, rain, and sensor noise (calculating LSNR and object solidity).

### 3. `yolo_label_generation/`
Tools to transform the cleaned binary masks into normalized YOLO segmentation polygons. It also includes scripts to build the strict YOLO directory structure (`train/val/test`) and automatically validate the generated annotations to catch overly large or broken bounding boxes.

### 4. `tracking_and_trajectory_analysis/`
Post-processing tools that analyze the outputs of the T-Rex tracking software (`.npz` files), reconstruct bat trajectories (tracklets), and visualize their flight paths across the frame.

## 🚀 Getting Started

### Prerequisites
Make sure you have python installed. The required packages include:
- `opencv-python` (cv2)
- `numpy`
- `pandas`
- `matplotlib`

*A `requirements.txt` file will be provided to easily install all dependencies.*

### Usage Pipeline
To process new data, the scripts are generally run in this order:
1. Run frame extraction from `data_extraction_and_sampling/`
2. Sample the frames you want to use for training/testing.
3. Use `generate_binary_masks.py` and filtering scripts in `mask_generation_and_noise_filtering/` to create ground truth data.
4. Convert those masks using the scripts in `yolo_label_generation/` to get your `.txt` label files.
5. Train your YOLO model (not included in this repo).

## 📝 License
[Specify License, e.g., MIT, GPL, or proprietary]
