# Refined GitHub Repository Structure for Kasanka Bat Counting

Based on your feedback, here is the updated, English-only repository structure. The folder names have been made more descriptive, script names clarify exactly what they do, and I have proposed merging some scripts that perform very similar tasks.

## 📂 1. `data_extraction_and_sampling/`
*Scripts to extract image frames from videos and sample them for training datasets.*

* **`extract_frames_single_video.py`** *(formerly Videos_in_frame.py)*: Sequentially extracts frames from MP4 videos at 1 FPS.
* **`extract_frames_batch_parallel.py`** *(formerly Videos_in_frame_2.py)*: Faster, parallel version that extracts frames from multiple videos simultaneously.
* **`rename_dataset_folders.py`** *(formerly Massenumbenennung_ordner_files.py)*: Recursively renames camera folders and files to include the date and location in the filename.
* **`sample_training_frames_proportional.py`** *(formerly Sampling_train_data.py)*: Randomly samples a proportional number of frames from various camera folders for training.
* **`sample_sharp_frames_only.py`** *(formerly Sampling_train_Bildanalyse.py)*: Samples training frames proportionally, but applies a blur filter to ensure only sharp images are selected.
* **`copy_random_frames_for_training.py`** *(formerly new_data_to_train_data.py)*: Picks random frames from camera folders and copies them into a dedicated training directory.
* **`sample_validation_set.py`** *(formerly Untitled-1.py)*: Selects images for a validation set, strictly ensuring no overlap with the training set.

## 📂 2. `mask_generation_and_noise_filtering/`
*Scripts to generate black-and-white masks from images and apply physical/geometric filters to remove noise.*

* **`generate_binary_masks.py`** *(formerly Creating_mask.py)*: Converts grayscale images into binary black-and-white masks by subtracting the background.
* **`filter_mask_noise_and_clouds.py`** *(formerly Test_dilation.py)*: Advanced script that filters out rain, noise, and clouds from masks using geometry, contrast (LSNR), and object shape.
* **`batch_clean_and_dilate_masks.py`** *(formerly clean_masks_folder.py)*: Cleans an entire folder of masks by removing tiny objects and slightly enlarging (dilating) medium ones.
* **`analyze_mask_histogram.py`** *(formerly Histogram.py)*: Plots a size distribution histogram of objects found in a mask folder to help understand the noise profile.

## 📂 3. `yolo_label_generation/`
*Scripts to convert filtered masks into the YOLO segmentation format and validate the bounding boxes.*

* **`build_yolo_directory_structure.py`** *(formerly #in Yolo Format bringen.py)*: Organizes images and labels into the strict YOLO format (`train/valid/test`) and creates the required `data.yaml` configuration file.
* **`yolo_segmentation_utils.py`** *(formerly yolo_seg_utils.py)*: Essential helper functions that convert binary masks into normalized YOLO polygons.
* **`convert_single_mask_to_yolo.py`** *(formerly mask_to_yoloseg_v3.py)*: Converts a single test mask into a YOLO label and plots a before/after comparison.
* **`analyze_single_mask_contours.py`** *(formerly analyze_mask.py)*: Prints statistics for large objects found within a single mask image.
* **`batch_convert_masks_to_yolo.py`**: A flexible script that converts an entire folder of masks into YOLO format labels. Supports multiple mask naming conventions and includes a dry-run feature.
* **`validate_yolo_labels.py`**: A robust validation script that checks labels for suspiciously large objects. Can analyze a single file or an entire directory with aggregated statistics.

## 📂 4. `tracking_and_trajectory_analysis/`
*Scripts for analyzing flight paths and tracking data from T-Rex.*

* **`analyze_trex_tracking_data.py`** *(formerly analyze_trex_npz.py)*: Parses T-Rex `.npz` tracking files, handles missing frames, and plots the flight trajectories (tracklets) of the bats.

---

### Next Steps for GitHub
The repository is now fully organized and verified.
- **Main Branch**: Contains the final restructured and fused codebase.
- **Yolo_into_polygon Branch**: Matches the clean state and can be used for feature development.
- **Documentation**: Use `README.md` as the primary guide for other users.
