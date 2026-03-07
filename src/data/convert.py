"""
Convert the raw dataset + generated masks into YOLO segmentation format.
Organizes images and labels into the split directories expected by YOLO.

YOLO seg directory structure:
    data/yolo_seg/
    ├ images/
    │   ├ train/
    │   ├ val/
    │   └ test/
    └ labels/
        ├ train/
        ├ val/
        └ test/

Each image gets a unique name: {class}_{original_stem}.jpg
Each label file: {class}_{original_stem}.txt

Usage:
    python -m src.data.convert
"""

import cv2
import json
import shutil
from pathlib import Path
from tqdm import tqdm

from src.config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    IMGSZ,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    SPLITS_DIR,
    YOLO_SEG_DIR,
)
from src.data.split import load_splits


def create_yolo_dirs(base_dir: Path) -> dict[str, dict[str, Path]]:
    """Create YOLO directory structure and return paths."""
    dirs = {}
    for split in ["train", "val", "test"]:
        dirs[split] = {
            "images": base_dir / "images" / split,
            "labels": base_dir / "labels" / split,
        }
        dirs[split]["images"].mkdir(parents=True, exist_ok=True)
        dirs[split]["labels"].mkdir(parents=True, exist_ok=True)
    return dirs


def convert_dataset(
    raw_dir: Path | None = None,
    splits_dir: Path | None = None,
    output_dir: Path | None = None,
    imgsz: int | None = None,
) -> dict:
    """
    Convert the raw dataset to YOLO seg format using pre-computed splits.

    Args:
        raw_dir: Path to raw data.
        splits_dir: Path to split JSON files.
        output_dir: Output directory for YOLO format data.
        imgsz: Target image size (square). None = no resize.

    Returns:
        Statistics dict.
    """
    raw_dir = raw_dir or RAW_DIR
    splits_dir = splits_dir or SPLITS_DIR
    output_dir = output_dir or YOLO_SEG_DIR
    imgsz = imgsz or IMGSZ

    color_dir = raw_dir / RAW_COLOR_FOLDER
    all_labels_dir = output_dir / "all_labels"

    # Load splits
    splits = load_splits(splits_dir)

    # Create directory structure
    dirs = create_yolo_dirs(output_dir)

    stats = {split: {"total": 0, "copied": 0, "missing_label": 0} for split in splits}

    for split_name, file_list in splits.items():
        print(f"\n[convert] Processing {split_name} split ({len(file_list)} images)...")

        for rel_path in tqdm(file_list, desc=f"  {split_name}"):
            # rel_path is like "A/1.jpg"
            cls_name, img_filename = rel_path.split("/", 1)
            img_stem = Path(img_filename).stem

            # Unique filenames across classes
            unique_name = f"{cls_name}_{img_stem}"

            stats[split_name]["total"] += 1

            # Source paths
            src_img = color_dir / cls_name / img_filename
            src_label = all_labels_dir / f"{unique_name}.txt"

            if not src_img.exists():
                print(f"  WARNING: Missing image: {src_img}")
                continue

            # Destination paths
            dst_img = dirs[split_name]["images"] / f"{unique_name}.jpg"
            dst_label = dirs[split_name]["labels"] / f"{unique_name}.txt"

            # Read, resize, and save image
            img = cv2.imread(str(src_img))
            if img is None:
                print(f"  WARNING: Could not read: {src_img}")
                continue

            if imgsz is not None:
                img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(str(dst_img), img)

            # Copy label file
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
                stats[split_name]["copied"] += 1
            else:
                stats[split_name]["missing_label"] += 1

    # Print summary
    print("\n[convert] Summary:")
    for split_name, s in stats.items():
        print(
            f"  {split_name}: {s['copied']}/{s['total']} images with labels "
            f"({s['missing_label']} missing labels)"
        )

    return stats


def main():
    convert_dataset()


if __name__ == "__main__":
    main()
