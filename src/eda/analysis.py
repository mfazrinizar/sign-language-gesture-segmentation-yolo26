"""
EDA analysis computations for the sign language gesture dataset.
Provides data for the Streamlit dashboard.
"""

import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    INDEX_TO_NAME,
    RAW_BINARY_FOLDER,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    SPLITS_DIR,
    YOLO_SEG_DIR,
)


def get_class_distribution(splits_dir: Path = SPLITS_DIR) -> dict:
    """Get class distribution for each split."""
    distribution = {}
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}.json"
        if not split_file.exists():
            continue
        with open(split_file) as f:
            files = json.load(f)
        counter = Counter(f.split("/")[0] for f in files)
        distribution[split] = dict(sorted(counter.items()))
    return distribution


def get_image_statistics(
    data_dir: Path,
    n_samples: int = 100,
) -> dict:
    """Compute image statistics from a sample of images."""
    img_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
    if not img_files:
        return {}

    # Sample if too many
    rng = np.random.RandomState(42)
    if len(img_files) > n_samples:
        indices = rng.choice(len(img_files), n_samples, replace=False)
        img_files = [img_files[i] for i in indices]

    heights, widths = [], []
    channels_list = []
    intensities = {c: [] for c in ["R", "G", "B"]}

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)
        channels_list.append(img.shape[2] if len(img.shape) == 3 else 1)

        if len(img.shape) == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i, c in enumerate(["R", "G", "B"]):
                intensities[c].append(float(rgb[:, :, i].mean()))

    return {
        "n_images": len(heights),
        "height": {"mean": np.mean(heights), "std": np.std(heights), "min": min(heights), "max": max(heights)},
        "width": {"mean": np.mean(widths), "std": np.std(widths), "min": min(widths), "max": max(widths)},
        "channels": Counter(channels_list),
        "intensity_mean": {c: np.mean(v) for c, v in intensities.items() if v},
        "intensity_std": {c: np.std(v) for c, v in intensities.items() if v},
    }


def get_sample_images(
    data_dir: Path,
    class_name: str,
    n: int = 5,
) -> list[np.ndarray]:
    """Get n sample images for a given class."""
    cls_dir = data_dir / class_name
    if not cls_dir.exists():
        return []

    imgs = []
    for img_path in sorted(cls_dir.glob("*"))[:n]:
        img = cv2.imread(str(img_path))
        if img is not None:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def get_mask_quality_stats(
    labels_dir: Path,
    n_samples: int = 500,
) -> dict:
    """Analyze mask/label quality from YOLO seg labels."""
    label_files = list(labels_dir.rglob("*.txt"))
    if not label_files:
        return {}

    rng = np.random.RandomState(42)
    if len(label_files) > n_samples:
        indices = rng.choice(len(label_files), n_samples, replace=False)
        label_files = [label_files[i] for i in indices]

    n_points = []
    areas = []
    class_counts = Counter()

    for lbl_path in label_files:
        content = lbl_path.read_text().strip()
        if not content:
            continue
        for line in content.split("\n"):
            parts = line.split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            n_pts = len(coords) // 2
            n_points.append(n_pts)
            class_counts[class_id] += 1

            # Approximate area using shoelace formula on normalized coords
            xs = coords[0::2]
            ys = coords[1::2]
            area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in zip(xs, ys, xs[1:] + [xs[0]], ys[1:] + [ys[0]])))
            areas.append(area)

    return {
        "n_labels": len(label_files),
        "points_per_polygon": {
            "mean": float(np.mean(n_points)),
            "std": float(np.std(n_points)),
            "min": int(min(n_points)),
            "max": int(max(n_points)),
        },
        "normalized_area": {
            "mean": float(np.mean(areas)),
            "std": float(np.std(areas)),
            "min": float(min(areas)),
            "max": float(max(areas)),
        },
        "class_distribution": {INDEX_TO_NAME.get(k, str(k)): v for k, v in sorted(class_counts.items())},
    }
