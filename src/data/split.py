"""
Stratified train/val/test split of the dataset.
Splits BEFORE any processing to prevent data leakage.

Usage:
    python -m src.data.split
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_FOLDERS,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def collect_file_list(raw_dir: Path) -> tuple[list[str], list[int]]:
    """
    Collect all image filenames with their class indices.
    Returns (file_paths_relative_to_class_folder, class_indices).
    """
    color_dir = raw_dir / RAW_COLOR_FOLDER
    filepaths = []
    labels = []

    for cls_idx, cls_name in enumerate(CLASS_FOLDERS):
        cls_dir = color_dir / cls_name
        if not cls_dir.is_dir():
            print(f"[split] WARNING: Missing class folder: {cls_dir}")
            continue

        for img_path in sorted(cls_dir.glob("*")):
            if img_path.is_file():
                # Store as "class_name/filename"
                filepaths.append(f"{cls_name}/{img_path.name}")
                labels.append(cls_idx)

    print(f"[split] Collected {len(filepaths)} images across {len(set(labels))} classes")
    return filepaths, labels


def split_dataset(
    filepaths: list[str],
    labels: list[int],
) -> dict[str, list[str]]:
    """
    Perform stratified train/val/test split.
    Returns dict with 'train', 'val', 'test' keys mapping to file lists.
    """
    # First split: train vs (val + test)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_files, valtest_files, train_labels, valtest_labels = train_test_split(
        filepaths,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Second split: val vs test (from the remaining)
    relative_test_ratio = TEST_RATIO / val_test_ratio
    val_files, test_files, _, _ = train_test_split(
        valtest_files,
        valtest_labels,
        test_size=relative_test_ratio,
        stratify=valtest_labels,
        random_state=RANDOM_SEED,
    )

    splits = {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }

    for name, files in splits.items():
        print(f"[split] {name}: {len(files)} images")

    return splits


def save_splits(splits: dict[str, list[str]], output_dir: Path) -> None:
    """Save split file lists to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, files in splits.items():
        out_file = output_dir / f"{name}.json"
        with open(out_file, "w") as f:
            json.dump(files, f, indent=2)
        print(f"[split] Saved {out_file}")

    # Also save a summary
    summary = {name: len(files) for name, files in splits.items()}
    summary["total"] = sum(summary.values())
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_splits(splits_dir: Path) -> dict[str, list[str]]:
    """Load previously saved splits."""
    splits = {}
    for name in ["train", "val", "test"]:
        split_file = splits_dir / f"{name}.json"
        with open(split_file) as f:
            splits[name] = json.load(f)
    return splits


def main():
    """Run the full splitting pipeline."""
    filepaths, labels = collect_file_list(RAW_DIR)
    splits = split_dataset(filepaths, labels)
    save_splits(splits, SPLITS_DIR)
    print(f"\n[split] Done! Splits saved to {SPLITS_DIR}")
    return splits


if __name__ == "__main__":
    main()
