import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# --- ADJUST PATH ---
MASK_DIR = "/Users/cara/Desktop/BA/YOLO/Kasanka_YOLO_Dataset_train_mask"
# -------------------

def analyze_contour_areas():
    if not os.path.exists(MASK_DIR):
        print(f"❌ Path not found: {MASK_DIR}")
        return

    all_areas = []
    file_count = 0
    
    files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith('.png')]
    total_files = len(files)

    print(f"📊 Starting analysis of {total_files} masks...")

    for filename in files:
        path = os.path.join(MASK_DIR, filename)
        
        # Load mask (0 = Grayscale)
        mask = cv2.imread(path, 0)
        
        if mask is None:
            continue
            
        # Optional: Here you could ignore "Ghost Boxes" (dark pixels) 
        # if you do thresholding. 
        # Currently we check EVERYTHING that is not black.
        # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_areas.append(area)
            
        file_count += 1
        if file_count % 100 == 0:
            print(f"  ... {file_count} images processed ({len(all_areas)} contours found)")

    if not all_areas:
        print("❌ No contours found! Are the images black?")
        return

    # Convert to Numpy array for statistics
    areas_np = np.array(all_areas)

    # --- Calculate Statistics ---
    p50 = np.percentile(areas_np, 50) # Median
    p90 = np.percentile(areas_np, 90)
    p95 = np.percentile(areas_np, 95)
    p99 = np.percentile(areas_np, 99)
    max_area = np.max(areas_np)

    print("\n--- RESULTS ---")
    print(f"Total contours: {len(areas_np)}")
    print(f"Median size:      {p50:.2f} px")
    print(f"90% are smaller than: {p90:.2f} px")
    print(f"95% are smaller than: {p95:.2f} px")
    print(f"99% are smaller than: {p99:.2f} px")
    print(f"Largest contour:     {max_area:.2f} px")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))

    # Plot 1: Overview (Logarithmic, because noise is often extremely frequent)
    plt.subplot(1, 2, 1)
    plt.hist(areas_np, bins=100, range=(0, 100), color='blue', alpha=0.7, log=True)
    plt.title("Distribution of Contour Sizes (0-100px)")
    plt.xlabel("Area (Pixels)")
    plt.ylabel("Count (Log Scale)")
    plt.axvline(p90, color='r', linestyle='dashed', linewidth=1, label=f'90%: {p90:.1f}px')
    plt.axvline(p95, color='g', linestyle='dashed', linewidth=1, label=f'95%: {p95:.1f}px')
    plt.legend()

    # Plot 2: Zoom into critical area (0-20 Pixel)
    plt.subplot(1, 2, 2)
    plt.hist(areas_np, bins=20, range=(0, 20), color='orange', alpha=0.7, edgecolor='black')
    plt.title("Zoom: Critical Area (0-20px)")
    plt.xlabel("Area (Pixels)")
    plt.ylabel("Count (Linear)")
    plt.xticks(range(0, 21, 1)) # Show every pixel value on axis
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_contour_areas()
