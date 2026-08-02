import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION FOR PATHS ---
# Adjust drive letter or root folder as needed
SEARCH_DIR = r"E:\KasankaCameras"

# Number of concurrent video processing workers
# 4 is usually optimal for external HDDs to avoid IO bottlenecks.
MAX_WORKERS = 4

def process_single_video(file_info):
    """
    Function executed in parallel across multiple CPU workers.
    """
    video_path, output_dir, output_pattern, filename = file_info

    # Create target directory
    os.makedirs(output_dir, exist_ok=True)

    # --- FFmpeg Command (CPU Mode) ---
    command = [
        "ffmpeg",
        "-nostdin",      # Important for background execution
        "-y",            # Force overwrite
        "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "1",     # Highest JPEG quality
        "-qmin", "1",    # Prevent quality fluctuations
        "-hide_banner",
        "-loglevel", "error",
        output_pattern
    ]

    try:
        subprocess.run(command, check=True)
        return f"Completed: {filename}"
    except subprocess.CalledProcessError as e:
        return f"!!! ERROR processing {filename}: {e}"
    except FileNotFoundError:
        return "!!! ERROR: ffmpeg executable not found."

def main():
    print(f"High-Performance Batch Extractor Started.")
    print(f"Searching in: {SEARCH_DIR}")
    print(f"Concurrent processes: {MAX_WORKERS}")
    print("-" * 60)

    # Check if path exists
    if not os.path.exists(SEARCH_DIR):
        print(f"CRITICAL ERROR: Path '{SEARCH_DIR}' does not exist.")
        print("Please verify the directory path.")
        return

    # 1. Collect all videos
    tasks = []
    print("Collecting video files...")

    for root, dirs, files in os.walk(SEARCH_DIR):
        for filename in files:
            if filename.lower().endswith(".mp4") and not filename.startswith("._"):

                video_path = os.path.join(root, filename)
                video_name_no_ext = os.path.splitext(filename)[0]

                # Output paths
                output_dir = os.path.join(root, f"{video_name_no_ext}_frames")
                output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

                # Append task tuple
                tasks.append((video_path, output_dir, output_pattern, filename))

    print(f"Found {len(tasks)} videos. Starting parallel extraction...")
    print("-" * 60)

    # 2. Process in parallel
    if tasks:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(process_single_video, tasks)

            # Print output results
            for result in results:
                print(result)
    else:
        print("No MP4 files found.")

    print("-" * 60)
    print("All videos processed.")

if __name__ == "__main__":
    # Verify FFmpeg availability
    if shutil.which("ffmpeg") is None:
        print("WARNING: 'ffmpeg' executable was not found!")
        print("Please ensure FFmpeg is installed and added to system PATH.")
    else:
        main()
