"""
Generate segmentation masks from the dataset's pre-processed binary images.
Converts binary images → contour polygons → YOLO segmentation label format.

The binary images from "Gesture Image Pre-Processed Data" have JPEG compression
artifacts (not truly 0/255), so we apply a threshold at 127 to get clean masks.

YOLO seg label format (per line):
    class_id x1 y1 x2 y2 ... xn yn
    (all coordinates normalized to [0, 1])

Usage:
    python -m src.data.mask_generator
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    RAW_BINARY_FOLDER,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    YOLO_SEG_DIR,
)


def extract_contour_polygon(
    binary_img: np.ndarray,
    threshold: int = 127,
    min_area: int = 50,
    epsilon_ratio: float = 0.005,
) -> list[np.ndarray] | None:
    """
    Extract the largest contour polygon from a binary image.

    Args:
        binary_img: Grayscale binary image.
        threshold: Pixel threshold for binarization.
        min_area: Minimum contour area to consider valid.
        epsilon_ratio: Polygon approximation accuracy (relative to perimeter).

    Returns:
        List of contour point arrays, or None if no valid contour found.
    """
    _, thresh = cv2.threshold(binary_img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Select the largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return None

    # Approximate polygon to reduce points while preserving shape
    perimeter = cv2.arcLength(largest, True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Need at least 3 points for a valid polygon
    if len(approx) < 3:
        return None

    return [approx]


def contour_to_yolo_seg(
    contours: list[np.ndarray],
    img_width: int,
    img_height: int,
    class_id: int,
) -> str:
    """
    Convert contour polygon to YOLO segmentation label format.

    Returns:
        YOLO seg label string: "class_id x1 y1 x2 y2 ... xn yn"
    """
    points = contours[0].reshape(-1, 2)  # (N, 2)

    # Normalize coordinates to [0, 1]
    norm_x = points[:, 0] / img_width
    norm_y = points[:, 1] / img_height

    # Clip to valid range
    norm_x = np.clip(norm_x, 0.0, 1.0)
    norm_y = np.clip(norm_y, 0.0, 1.0)

    # Build label string
    coords = []
    for x, y in zip(norm_x, norm_y):
        coords.extend([f"{x:.6f}", f"{y:.6f}"])

    return f"{class_id} " + " ".join(coords)


def generate_masks_for_class(
    class_name: str,
    raw_dir: Path,
    output_labels_dir: Path,
) -> dict:
    """Generate YOLO seg labels for all images of a given class."""
    class_id = CLASS_NAMES[class_name]
    binary_dir = raw_dir / RAW_BINARY_FOLDER / class_name
    color_dir = raw_dir / RAW_COLOR_FOLDER / class_name

    stats = {"total": 0, "success": 0, "failed": 0}

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(binary_dir.glob("*")):
        if not img_path.is_file():
            continue

        stats["total"] += 1

        # Read binary image
        binary_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if binary_img is None:
            stats["failed"] += 1
            continue

        h, w = binary_img.shape

        # Extract contour polygon
        contours = extract_contour_polygon(binary_img)
        if contours is None:
            stats["failed"] += 1
            continue

        # Convert to YOLO seg format
        label_str = contour_to_yolo_seg(contours, w, h, class_id)

        # Save label file: same name as image but .txt extension
        # Use class_name prefix to avoid filename collisions across classes
        label_filename = f"{class_name}_{img_path.stem}.txt"
        label_path = output_labels_dir / label_filename
        label_path.write_text(label_str + "\n")

        stats["success"] += 1

    return stats


def generate_all_masks(raw_dir: Path | None = None) -> dict:
    """Generate masks for all classes. Returns overall statistics."""
    if raw_dir is None:
        raw_dir = RAW_DIR

    # Temporary output for all labels before splitting
    all_labels_dir = YOLO_SEG_DIR / "all_labels"
    all_labels_dir.mkdir(parents=True, exist_ok=True)

    total_stats = {"total": 0, "success": 0, "failed": 0}

    for cls_name in tqdm(CLASS_FOLDERS, desc="Generating masks"):
        stats = generate_masks_for_class(cls_name, raw_dir, all_labels_dir)
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"\n[mask_gen] Total: {total_stats['total']}")
    print(f"[mask_gen] Success: {total_stats['success']}")
    print(f"[mask_gen] Failed: {total_stats['failed']}")

    return total_stats


def main():
    generate_all_masks()


if __name__ == "__main__":
    main()
