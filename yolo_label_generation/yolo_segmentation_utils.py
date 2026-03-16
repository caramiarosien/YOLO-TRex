from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def masks_to_yolo_seg_lines(
    mask: np.ndarray,
    class_id: int = 0,
    approx_eps: float = 0.0,
    include_bbox: bool = False,
) -> list[str]:
    """Convert a binary mask into YOLO-segmentation label lines.

    Parameters
    ----------
    mask:
        Grayscale or binary mask where foreground pixels are non-zero.
    class_id:
        Numeric class identifier to place at the beginning of each label line.
    approx_eps:
        ``epsilon`` passed to :func:`cv2.approxPolyDP`. Use ``0`` to keep the
        full contour; higher values simplify the polygon.
    include_bbox:
        If ``True``, prepend YOLOv8-style normalised bounding box values
        (cx cy w h). Default is ``False`` to emit only the polygon points.
    """

    if mask is None:
        raise ValueError("mask is None; make sure mask_path points to a valid image")

    # Ensure a binary mask that uses values 0 and 255.
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("mask has invalid dimensions")

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lines: list[str] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue

        if approx_eps > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon=approx_eps, closed=True)

        cnt = cnt.reshape(-1, 2).astype(np.float32)

        if include_bbox:
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox = np.array(
                [
                    (x + bw / 2.0) / float(w),
                    (y + bh / 2.0) / float(h),
                    bw / float(w),
                    bh / float(h),
                ],
                dtype=np.float32,
            )

        cnt[:, 0] /= float(w)
        cnt[:, 1] /= float(h)
        np.clip(cnt, 0.0, 1.0, out=cnt)

        coords = " ".join(f"{value:.6f}" for value in cnt.flatten())
        if include_bbox:
            bbox_str = " ".join(f"{value:.6f}" for value in bbox)
            line = f"{class_id} {bbox_str} {coords}"
        else:
            line = f"{class_id} {coords}"
        lines.append(line)

    return lines


def write_yolo_seg_file(lines: Sequence[str], output_path: Path | str) -> None:
    """Serialise YOLO segmentation lines to a text file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip() + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")
