"""
Central configuration for the sign language gesture segmentation project.
All paths, constants, class mappings, and hyperparameter defaults.
"""

import os
from pathlib import Path

#  Environment Detection 
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

#  Project Paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
YOLO_SEG_DIR = DATA_DIR / "yolo_seg"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ULTRALYTICS_DIR = PROJECT_ROOT / "ultralytics"

#  Dataset Info 
KAGGLE_DATASET = "ahmedkhanak1995/sign-language-gesture-images-dataset"
RAW_COLOR_FOLDER = "Gesture Image Data"
RAW_BINARY_FOLDER = "Gesture Image Pre-Processed Data"

# 37 classes: A-Z (26), 0-9 (10), _ (space)
CLASS_FOLDERS = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [str(d) for d in range(10)]
    + ["_"]
)
CLASS_NAMES = {
    **{chr(c): i for i, c in enumerate(range(ord("A"), ord("Z") + 1))},
    **{str(d): 26 + d for d in range(10)},
    "_": 36,
}
INDEX_TO_NAME = {v: k for k, v in CLASS_NAMES.items()}
NUM_CLASSES = 37

#  Dataset Split 
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

#  Training Hyperparameters 
IMGSZ = 224
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 20
WORKERS = 8

# Auto-detect device
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                return "0,1"  # multi-GPU
            return "0"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

#  Model 
SEG_MODEL_YAML = "yolo26n-seg.yaml"
