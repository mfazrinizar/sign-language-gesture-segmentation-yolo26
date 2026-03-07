"""
Download the Sign Language Gesture Images Dataset via kagglehub.
Handles both Kaggle kernel and local environments.

Usage:
    python -m src.data.download
"""

import shutil
from pathlib import Path

import kagglehub

from src.config import (
    DATA_DIR,
    IS_KAGGLE,
    KAGGLE_DATASET,
    RAW_BINARY_FOLDER,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    CLASS_FOLDERS,
)


def download_dataset() -> Path:
    """Download dataset and return path to the raw data directory."""
    if IS_KAGGLE:
        # On Kaggle the dataset is mounted at /kaggle/input/
        kaggle_input = Path("/kaggle/input/datasets/sign-language-gesture-images-dataset")
        if kaggle_input.exists():
            print(f"[download] Kaggle environment detected. Data at: {kaggle_input}")
            return kaggle_input
        # Fallback: use kagglehub even on Kaggle
        print("[download] Kaggle input not found, downloading via kagglehub...")

    print(f"[download] Downloading {KAGGLE_DATASET} via kagglehub...")
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[download] Downloaded to: {downloaded_path}")
    return downloaded_path


def setup_raw_data(downloaded_path: Path) -> Path:
    """
    Copy/symlink downloaded data into ./data/raw/ for consistent access.
    Returns the raw data directory path.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # The downloaded dataset may have a nested structure; find the actual root
    # that contains both "Gesture Image Data" and "Gesture Image Pre-Processed Data"
    data_root = _find_data_root(downloaded_path)
    if data_root is None:
        raise FileNotFoundError(
            f"Could not find '{RAW_COLOR_FOLDER}' and '{RAW_BINARY_FOLDER}' "
            f"in {downloaded_path}"
        )

    # Symlink (preferred) or copy both folders into RAW_DIR
    for folder_name in [RAW_COLOR_FOLDER, RAW_BINARY_FOLDER]:
        src = data_root / folder_name
        dst = RAW_DIR / folder_name
        if dst.exists() or dst.is_symlink():
            print(f"[download] '{dst.name}' already exists in raw/, skipping.")
            continue
        try:
            dst.symlink_to(src)
            print(f"[download] Symlinked: {dst} -> {src}")
        except (PermissionError, OSError) as e:
            print(f"[download] Symlink failed ({e}), copying instead...")
            shutil.copytree(src, dst)
            print(f"[download] Copied: {src} -> {dst}")

    return RAW_DIR


def _find_data_root(base: Path) -> Path | None:
    """Recursively search for the directory containing both data folders."""
    if (base / RAW_COLOR_FOLDER).is_dir() and (base / RAW_BINARY_FOLDER).is_dir():
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            result = _find_data_root(child)
            if result is not None:
                return result
    return None


def validate_dataset(raw_dir: Path) -> dict:
    """Validate dataset structure and return statistics."""
    stats = {"color": {}, "binary": {}}

    for folder_type, folder_name in [
        ("color", RAW_COLOR_FOLDER),
        ("binary", RAW_BINARY_FOLDER),
    ]:
        folder_path = raw_dir / folder_name
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        for cls in CLASS_FOLDERS:
            cls_dir = folder_path / cls
            if not cls_dir.is_dir():
                print(f"[validate] WARNING: Missing class folder: {cls_dir}")
                stats[folder_type][cls] = 0
                continue
            n_images = len(list(cls_dir.glob("*")))
            stats[folder_type][cls] = n_images

    total_color = sum(stats["color"].values())
    total_binary = sum(stats["binary"].values())
    print(f"[validate] Color images: {total_color}")
    print(f"[validate] Binary images: {total_binary}")
    print(f"[validate] Classes found (color): {sum(1 for v in stats['color'].values() if v > 0)}")
    print(f"[validate] Classes found (binary): {sum(1 for v in stats['binary'].values() if v > 0)}")

    return stats


def main():
    """Download, setup, and validate the dataset."""
    downloaded_path = download_dataset()
    raw_dir = setup_raw_data(downloaded_path)
    stats = validate_dataset(raw_dir)

    print("\n[download] Dataset ready!")
    print(f"[download] Raw data at: {raw_dir}")
    return raw_dir, stats


if __name__ == "__main__":
    main()
