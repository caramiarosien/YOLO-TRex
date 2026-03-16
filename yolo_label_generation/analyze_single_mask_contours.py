import cv2
import numpy as np
from pathlib import Path

mask_path = Path("./Kasanka_YOLO_Dataset_train_mask/GH030007_frames_frame_0203_mask.png")
if not mask_path.exists():
    print(f"Mask not found: {mask_path}")
    exit(1)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(f"Found {len(contours)} contours.")
print("Listing large contours (area > 1000):")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    solidity = float(area) / rect_area if rect_area > 0 else 0
    
    if area > 1000:
        print(f"Contour {i}: Area={area:.2f}, BBox=({x},{y},{w},{h}), RectArea={rect_area}, Solidity={solidity:.2f}")
