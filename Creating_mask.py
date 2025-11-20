from pathlib import Path

import cv2
import numpy as np

from yolo_seg_utils import masks_to_yolo_seg_lines, write_yolo_seg_file


TRAIN_ROOT = Path("/Users/cara/Desktop/BA/YOLO/YOLO_Dataset/train")
IMAGE_DIR = TRAIN_ROOT / "images"
LABEL_DIR = TRAIN_ROOT / "labels"
MASK_OUTPUT_DIR = Path("/Users/cara/Desktop/Train_images/mask")

GAUSSIAN_KERNEL = (51, 51)
BINARY_THRESHOLD = 0.7
MIN_DIFFERENCE = 5
CLASS_ID = 0  # siehe data.yaml (einzige Klasse "bat")
APPROX_EPS = 0.5


def preprocess_to_binary(image: np.ndarray, binary_thresh: float, background: np.ndarray) -> np.ndarray:
    """
    Converts 2D image to binary after rescaling pixel intensity.
    """
    image_rescale = image
    threshold = binary_thresh * background
    threshold = np.where(threshold < MIN_DIFFERENCE, MIN_DIFFERENCE, threshold)
    binary_image = np.where(image_rescale < threshold, 0, 255)
    return binary_image


def build_mask(img_path: Path) -> np.ndarray:
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")

    gray = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    binary_mask = preprocess_to_binary(gray, binary_thresh=BINARY_THRESHOLD, background=background)
    return binary_mask.astype(np.uint8)


def create_labels_for_directory(
    image_dir: Path = IMAGE_DIR,
    mask_dir: Path = MASK_OUTPUT_DIR,
    label_dir: Path = LABEL_DIR,
) -> None:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"Keine Bilder in {image_dir} gefunden.")

    mask_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        try:
            mask = build_mask(img_path)
            yolo_lines = masks_to_yolo_seg_lines(mask, class_id=CLASS_ID, approx_eps=APPROX_EPS)

            mask_path = mask_dir / f"{img_path.stem}_mask.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise IOError(f"Konnte Maske nicht speichern: {mask_path}")

            label_path = label_dir / f"{img_path.stem}.txt"
            write_yolo_seg_file(yolo_lines, label_path)
            print(
                f"[{idx:04d}/{total:04d}] {img_path.name}: Maske → {mask_path.name}, "
                f"Label → {label_path.name} ({len(yolo_lines)} Objekte)"
            )
        except Exception as exc:
            print(f"Fehler bei {img_path.name}: {exc}")


if __name__ == "__main__":
    create_labels_for_directory()
