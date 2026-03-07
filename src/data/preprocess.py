"""
Preprocessing utilities for image quality enhancement and data validation.

Usage:
    python -m src.data.preprocess
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import YOLO_SEG_DIR


def compute_dataset_stats(images_dir: Path) -> dict:
    """Compute mean and std of pixel values across the dataset."""
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0

    img_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png"))

    for img_path in tqdm(img_files, desc="Computing stats"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float64) / 255.0
        pixel_sum += img_float.mean(axis=(0, 1))
        pixel_sq_sum += (img_float ** 2).mean(axis=(0, 1))
        count += 1

    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean ** 2)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_images": count,
    }


def validate_labels(labels_dir: Path) -> dict:
    """Validate YOLO seg label files for format correctness."""
    stats = {"total": 0, "valid": 0, "empty": 0, "invalid": 0, "issues": []}

    label_files = list(labels_dir.rglob("*.txt"))
    stats["total"] = len(label_files)

    for lbl_path in label_files:
        content = lbl_path.read_text().strip()
        if not content:
            stats["empty"] += 1
            continue

        lines = content.split("\n")
        valid = True
        for line in lines:
            parts = line.split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                stats["issues"].append(f"{lbl_path.name}: too few points ({len(parts)})")
                valid = False
                break

            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    stats["issues"].append(f"{lbl_path.name}: odd number of coordinates")
                    valid = False
                    break
                if any(c < 0 or c > 1 for c in coords):
                    stats["issues"].append(f"{lbl_path.name}: coords out of [0,1]")
                    valid = False
                    break
            except ValueError:
                stats["issues"].append(f"{lbl_path.name}: invalid number format")
                valid = False
                break

        if valid:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    return stats


def main():
    """Run preprocessing validation and stats."""
    print("=== Dataset Statistics ===")
    train_imgs = YOLO_SEG_DIR / "images" / "train"
    if train_imgs.exists():
        stats = compute_dataset_stats(train_imgs)
        print(f"Training set: {stats['n_images']} images")
        print(f"Mean (RGB): {[f'{x:.4f}' for x in stats['mean']]}")
        print(f"Std  (RGB): {[f'{x:.4f}' for x in stats['std']]}")

    print("\n=== Label Validation ===")
    for split in ["train", "val", "test"]:
        labels_dir = YOLO_SEG_DIR / "labels" / split
        if labels_dir.exists():
            v = validate_labels(labels_dir)
            print(
                f"{split}: {v['valid']}/{v['total']} valid, "
                f"{v['empty']} empty, {v['invalid']} invalid"
            )
            if v["issues"][:5]:
                for issue in v["issues"][:5]:
                    print(f"  ! {issue}")


if __name__ == "__main__":
    main()
