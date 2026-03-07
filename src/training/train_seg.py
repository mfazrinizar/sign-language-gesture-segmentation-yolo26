"""
Training pipeline for YOLO26n-seg on sign language gesture dataset.
Handles both segmentation and inherent classification.

Supports hybrid AdamW→SGD two-phase training:
  Phase 1 — AdamW for rapid convergence (default: first 30% of epochs)
  Phase 2 — SGD for fine-tuning to flatter minima (remaining 70% of epochs)

Usage:
    python -m src.training.train_seg [--epochs N] [--batch N] [--device DEV]
    python -m src.training.train_seg --hybrid          # two-phase AdamW→SGD
    python -m src.training.train_seg --resume           # resume from last.pt
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.config import (
    BATCH_SIZE,
    CONFIGS_DIR,
    DEVICE,
    EPOCHS,
    IMGSZ,
    MODELS_DIR,
    PATIENCE,
    RESULTS_DIR,
    SEG_MODEL_YAML,
    WORKERS,
    YOLO_SEG_DIR,
)


def _make_absolute_data_yaml() -> str:
    """
    Create a temporary dataset YAML with absolute paths so training works
    regardless of the current working directory.
    """
    src_yaml = CONFIGS_DIR / "dataset_seg.yaml"
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    # Override path with absolute
    cfg["path"] = str(YOLO_SEG_DIR.resolve())

    # Write to a temp file that persists for the training session
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="dataset_seg_", delete=False
    )
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def _common_train_kwargs(
    data_yaml: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    project_dir: str,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 5,
    resume: bool = False,
) -> dict:
    """Return the shared keyword arguments for ``model.train()``."""
    return dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=WORKERS,
        patience=PATIENCE,
        project=project_dir,
        name="yolo26n-seg",
        exist_ok=True,
        resume=resume,
        # --- Augmentations tuned for small hand gesture images ---
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        shear=5.0,
        flipud=0.0,  # no vertical flip (breaks gesture meaning)
        fliplr=0.0,  # no horizontal flip (breaks gesture meaning for J, Z, etc.)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # --- Optimizer ---
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        # --- Logging / checkpointing ---
        plots=True,
        save=True,
        save_period=5,
        val=True,
        verbose=True,
    )


def train_seg(
    epochs: int = EPOCHS,
    batch: int = BATCH_SIZE,
    imgsz: int = IMGSZ,
    device: str = DEVICE,
    resume: bool = False,
    optimizer: str = "AdamW",
) -> None:
    """
    Train YOLO26n-seg model on the sign language gesture dataset (single phase).

    The segmentation model inherently performs classification (class-aware
    detection + segmentation), so a separate cls model is unnecessary.
    """
    data_yaml = _make_absolute_data_yaml()
    project_dir = str(RESULTS_DIR / "seg")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested and checkpoint exists
    checkpoint = Path(project_dir) / "yolo26n-seg" / "weights" / "last.pt"
    if resume and checkpoint.exists():
        print(f"[train_seg] Resuming from checkpoint: {checkpoint}")
        model = YOLO(str(checkpoint), task="segment")
    else:
        # Initialize model from architecture YAML (train from scratch)
        model = YOLO(SEG_MODEL_YAML, task="segment")

    print(f"[train_seg] Model: {SEG_MODEL_YAML}")
    print(f"[train_seg] Data: {data_yaml}")
    print(f"[train_seg] Image size: {imgsz}")
    print(f"[train_seg] Epochs: {epochs}, Batch: {batch}")
    print(f"[train_seg] Device: {device}, Optimizer: {optimizer}")

    kwargs = _common_train_kwargs(
        data_yaml, epochs, batch, imgsz, device, project_dir,
        optimizer=optimizer, resume=resume,
    )
    results = model.train(**kwargs)

    # Save best model
    best_model = Path(project_dir) / "yolo26n-seg" / "weights" / "best.pt"
    if best_model.exists():
        dst = MODELS_DIR / "yolo26n-seg-best.pt"
        shutil.copy2(best_model, dst)
        print(f"[train_seg] Best model saved to: {dst}")

    return results


def train_seg_hybrid(
    total_epochs: int = EPOCHS,
    phase1_fraction: float = 0.30,
    batch: int = BATCH_SIZE,
    imgsz: int = IMGSZ,
    device: str = DEVICE,
) -> None:
    """
    Hybrid two-phase training: **AdamW → SGD**.

    Phase 1 (AdamW)  — fast convergence with adaptive learning rates.
        Epochs      : ``int(total_epochs * phase1_fraction)``
        lr0         : 0.001
        weight_decay: 0.0005
        warmup      : 5 epochs

    Phase 2 (SGD)    — fine-tuning for flatter minima / better generalisation.
        Epochs      : remaining epochs
        lr0         : 0.01  (standard SGD starting LR)
        weight_decay: 0.0005
        warmup      : 3 epochs (short re-warmup)

    Phase 2 loads ``best.pt`` from Phase 1 (not ``last.pt``) so it starts from
    the best-seen weights, then continues training with SGD.

    Note: Ultralytics tracks total epochs per run.  Phase 2 is a *new* training
    run (``resume=False``) that inherits only the weights.
    """
    phase1_epochs = max(1, int(total_epochs * phase1_fraction))
    phase2_epochs = total_epochs - phase1_epochs

    data_yaml = _make_absolute_data_yaml()
    project_dir = str(RESULTS_DIR / "seg")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    #  Phase 1: AdamW 
    print("=" * 60)
    print(f"PHASE 1 — AdamW  ({phase1_epochs} epochs)")
    print("=" * 60)

    model = YOLO(SEG_MODEL_YAML, task="segment")
    kwargs_p1 = _common_train_kwargs(
        data_yaml, phase1_epochs, batch, imgsz, device, project_dir,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        weight_decay=0.0005, warmup_epochs=5,
    )
    model.train(**kwargs_p1)

    # Take the best checkpoint from phase 1
    best_p1 = Path(project_dir) / "yolo26n-seg" / "weights" / "best.pt"
    if not best_p1.exists():
        best_p1 = Path(project_dir) / "yolo26n-seg" / "weights" / "last.pt"
    print(f"[hybrid] Phase 1 checkpoint: {best_p1}")

    #  Phase 2: SGD 
    print("=" * 60)
    print(f"PHASE 2 — SGD  ({phase2_epochs} epochs)")
    print("=" * 60)

    model_p2 = YOLO(str(best_p1), task="segment")
    kwargs_p2 = _common_train_kwargs(
        data_yaml, phase2_epochs, batch, imgsz, device, project_dir,
        optimizer="SGD", lr0=0.01, lrf=0.01,
        weight_decay=0.0005, warmup_epochs=3,
        resume=False,  # new run with SGD, inheriting weights only
    )
    results = model_p2.train(**kwargs_p2)

    # Save best model
    best_model = Path(project_dir) / "yolo26n-seg" / "weights" / "best.pt"
    if best_model.exists():
        dst = MODELS_DIR / "yolo26n-seg-best.pt"
        shutil.copy2(best_model, dst)
        print(f"[train_seg] Best model saved to: {dst}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26n-seg")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid AdamW→SGD two-phase training")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW", "SGD", "Adam", "RMSProp"],
                        help="Optimizer for single-phase training")
    parser.add_argument("--phase1-fraction", type=float, default=0.30,
                        help="Fraction of total epochs for AdamW phase (hybrid mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hybrid:
        train_seg_hybrid(
            total_epochs=args.epochs,
            phase1_fraction=args.phase1_fraction,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
    else:
        train_seg(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            resume=args.resume,
            optimizer=args.optimizer,
        )


if __name__ == "__main__":
    main()
