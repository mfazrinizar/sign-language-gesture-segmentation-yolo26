"""
Comprehensive evaluation metrics for classification, detection, and segmentation.
Computes all metrics specified in the project requirements.

Metrics:
    Classification: Accuracy, Recall/Sensitivity, Precision, Specificity, F1-score
    Detection:      mAP@50, mAP@50:95, IoU (Jaccard Index), Dice Coefficient
    Segmentation:   mAP@50, mAP@50:95, IoU (Jaccard Index), Dice Coefficient

Usage:
    python -m src.evaluation.metrics --model models/yolo26n-seg-best.pt
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

from src.config import (
    CONFIGS_DIR,
    DEVICE,
    IMGSZ,
    INDEX_TO_NAME,
    MODELS_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
)


def dice_coefficient(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    total = mask_pred.sum() + mask_gt.sum()
    if total == 0:
        return 1.0  # both empty
    return 2.0 * intersection / total


def jaccard_index(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute Jaccard Index (IoU) between two binary masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0  # both empty
    return intersection / union


def compute_specificity(confusion_matrix: np.ndarray) -> np.ndarray:
    """Compute per-class specificity from confusion matrix."""
    n_classes = confusion_matrix.shape[0]
    specificities = np.zeros(n_classes)
    total = confusion_matrix.sum()

    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = total - tp - fp - fn
        specificities[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return specificities


def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path | None = None,
    device: str = DEVICE,
    imgsz: int = IMGSZ,
    split: str = "test",
) -> dict:
    """
    Run comprehensive evaluation of a trained YOLO26n-seg model.

    Returns dict with all classification, detection, and segmentation metrics.
    """
    model = YOLO(str(model_path))
    data_yaml = data_yaml or str(CONFIGS_DIR / "dataset_seg.yaml")

    print(f"[eval] Model: {model_path}")
    print(f"[eval] Data: {data_yaml}")
    print(f"[eval] Split: {split}")

    # Run YOLO validation to get built-in metrics
    results = model.val(
        data=data_yaml,
        split=split,
        device=device,
        imgsz=imgsz,
        plots=True,
        save_json=True,
        verbose=True,
    )

    # Extract YOLO metrics
    metrics = {
        "detection": {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        },
        "segmentation": {
            "mAP50": float(results.seg.map50),
            "mAP50_95": float(results.seg.map),
            "precision": float(results.seg.mp),
            "recall": float(results.seg.mr),
        },
    }

    # Per-class metrics
    if hasattr(results.box, "maps") and results.box.maps is not None:
        per_class_det = {}
        for i, ap50_95 in enumerate(results.box.maps):
            cls_name = INDEX_TO_NAME.get(i, str(i))
            per_class_det[cls_name] = float(ap50_95)
        metrics["detection"]["per_class_mAP50_95"] = per_class_det

    if hasattr(results.seg, "maps") and results.seg.maps is not None:
        per_class_seg = {}
        for i, ap50_95 in enumerate(results.seg.maps):
            cls_name = INDEX_TO_NAME.get(i, str(i))
            per_class_seg[cls_name] = float(ap50_95)
        metrics["segmentation"]["per_class_mAP50_95"] = per_class_seg

    # Classification metrics from confusion matrix
    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        if cm is not None and cm.shape[0] >= NUM_CLASSES:
            cm_cls = cm[:NUM_CLASSES, :NUM_CLASSES]

            tp = np.diag(cm_cls)
            fp = cm_cls.sum(axis=0) - tp
            fn = cm_cls.sum(axis=1) - tp
            total_per_class = cm_cls.sum(axis=1)

            precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                0,
            )
            specificity = compute_specificity(cm_cls)
            accuracy = tp.sum() / cm_cls.sum() if cm_cls.sum() > 0 else 0

            metrics["classification"] = {
                "accuracy": float(accuracy),
                "precision_macro": float(precision.mean()),
                "recall_macro": float(recall.mean()),
                "f1_macro": float(f1.mean()),
                "specificity_macro": float(specificity.mean()),
                "per_class": {},
            }

            for i in range(min(NUM_CLASSES, len(tp))):
                cls_name = INDEX_TO_NAME.get(i, str(i))
                metrics["classification"]["per_class"][cls_name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "specificity": float(specificity[i]),
                    "support": int(total_per_class[i]),
                }

    return metrics


def compute_mask_metrics(
    model_path: str | Path,
    data_dir: Path | None = None,
    device: str = DEVICE,
    imgsz: int = IMGSZ,
    split: str = "test",
    max_images: int = 500,
) -> dict:
    """
    Compute pixel-level Dice and Jaccard (IoU) by running inference on images
    and comparing predicted masks against ground-truth polygon labels.

    Args:
        model_path: Path to trained .pt model.
        data_dir: YOLO dataset root (with images/ and labels/ subdirs).
        device: Device string.
        imgsz: Image size for inference.
        split: Which split to evaluate on.
        max_images: Cap on number of images to evaluate (for speed).

    Returns:
        Dict with mean_dice, mean_jaccard, per_class_dice, per_class_jaccard.
    """
    from src.config import YOLO_SEG_DIR

    data_dir = data_dir or YOLO_SEG_DIR
    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split

    model = YOLO(str(model_path))

    img_files = sorted(images_dir.glob("*.jpg"))
    if len(img_files) > max_images:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(img_files), max_images, replace=False)
        img_files = [img_files[i] for i in indices]

    per_class_dice = {i: [] for i in range(NUM_CLASSES)}
    per_class_jaccard = {i: [] for i in range(NUM_CLASSES)}

    for img_path in img_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Parse ground-truth
        gt_text = label_path.read_text().strip()
        if not gt_text:
            continue
        gt_parts = gt_text.split()
        gt_cls = int(gt_parts[0])
        gt_coords = [float(x) for x in gt_parts[1:]]

        # Build GT mask
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_points = np.array(
            [(int(gt_coords[j] * w), int(gt_coords[j + 1] * h))
             for j in range(0, len(gt_coords), 2)],
            dtype=np.int32,
        )
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [gt_points], 1)

        # Run inference
        results = model.predict(
            str(img_path), imgsz=imgsz, device=device, verbose=False
        )
        if not results or results[0].masks is None:
            # No detection — all-zero prediction
            pred_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Use the mask of the highest-confidence detection
            masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
            confs = results[0].boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            pred_mask_raw = masks[best_idx]
            pred_mask = cv2.resize(
                pred_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST
            )
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

        d = dice_coefficient(pred_mask, gt_mask)
        j = jaccard_index(pred_mask, gt_mask)
        per_class_dice[gt_cls].append(d)
        per_class_jaccard[gt_cls].append(j)

    # Aggregate
    all_dices = [v for vals in per_class_dice.values() for v in vals]
    all_jaccards = [v for vals in per_class_jaccard.values() for v in vals]

    result = {
        "mean_dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "mean_jaccard": float(np.mean(all_jaccards)) if all_jaccards else 0.0,
        "n_images": len(all_dices),
        "per_class_dice": {},
        "per_class_jaccard": {},
    }
    for cls_id in range(NUM_CLASSES):
        cls_name = INDEX_TO_NAME.get(cls_id, str(cls_id))
        if per_class_dice[cls_id]:
            result["per_class_dice"][cls_name] = float(np.mean(per_class_dice[cls_id]))
            result["per_class_jaccard"][cls_name] = float(np.mean(per_class_jaccard[cls_id]))

    return result


def log_test_predictions(
    model_path: str | Path,
    data_dir: Path | None = None,
    device: str = DEVICE,
    imgsz: int = IMGSZ,
    split: str = "test",
    output_dir: Path | None = None,
    save_overlays: bool = True,
    max_images: int | None = None,
) -> Path:
    """
    Run inference on the test set and log per-image predictions for quality
    analysis.  Saves a CSV log and optional mask-overlay images.

    Each row in the CSV contains:
        image, gt_class, pred_class, confidence, dice, jaccard, match

    When *save_overlays* is True, overlay images (original + predicted mask in
    green + GT mask contour in red) are written under
    ``output_dir/overlays/{class}/``.

    Args:
        model_path: Path to trained .pt model.
        data_dir: YOLO dataset root (images/ and labels/ subdirs).
        device: Inference device.
        imgsz: Image size for inference.
        split: Dataset split to evaluate on.
        output_dir: Where to write the log and overlays.
        save_overlays: Whether to save mask overlay images.
        max_images: Optional cap on number of images (None = all).

    Returns:
        Path to the CSV log file.
    """
    from src.config import YOLO_SEG_DIR

    data_dir = data_dir or YOLO_SEG_DIR
    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split
    output_dir = output_dir or (RESULTS_DIR / "evaluation" / f"{split}_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    img_files = sorted(images_dir.glob("*.jpg"))
    if max_images is not None and len(img_files) > max_images:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(img_files), max_images, replace=False)
        img_files = [img_files[i] for i in sorted(indices)]

    csv_path = output_dir / "predictions_log.csv"
    summary: dict = {
        "total": 0,
        "correct": 0,
        "no_detection": 0,
        "per_class": {INDEX_TO_NAME[i]: {"tp": 0, "fp": 0, "fn": 0, "dice_sum": 0.0, "jaccard_sum": 0.0, "count": 0}
                      for i in range(NUM_CLASSES)},
    }

    overlay_dir = output_dir / "overlays"
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image", "gt_class_id", "gt_class_name",
            "pred_class_id", "pred_class_name", "confidence",
            "dice", "jaccard", "match",
        ])

        for img_path in img_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            # Parse GT
            gt_text = label_path.read_text().strip()
            if not gt_text:
                continue
            gt_parts = gt_text.split()
            gt_cls = int(gt_parts[0])
            gt_name = INDEX_TO_NAME.get(gt_cls, str(gt_cls))
            gt_coords = [float(x) for x in gt_parts[1:]]

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Build GT mask
            gt_points = np.array(
                [(int(gt_coords[j] * w), int(gt_coords[j + 1] * h))
                 for j in range(0, len(gt_coords), 2)],
                dtype=np.int32,
            )
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(gt_mask, [gt_points], 1)

            # Run inference
            results = model.predict(
                str(img_path), imgsz=imgsz, device=device, verbose=False
            )

            summary["total"] += 1
            pred_cls = -1
            pred_name = "none"
            conf = 0.0
            pred_mask = np.zeros((h, w), dtype=np.uint8)

            if results and results[0].boxes is not None and len(results[0].boxes):
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                best_idx = int(confs.argmax())
                pred_cls = int(classes[best_idx])
                pred_name = INDEX_TO_NAME.get(pred_cls, str(pred_cls))
                conf = float(confs[best_idx])

                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    pred_mask_raw = masks[best_idx]
                    pred_mask = cv2.resize(
                        pred_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    pred_mask = (pred_mask > 0.5).astype(np.uint8)
            else:
                summary["no_detection"] += 1

            d = dice_coefficient(pred_mask, gt_mask)
            j = jaccard_index(pred_mask, gt_mask)
            match = int(pred_cls == gt_cls)
            summary["correct"] += match

            # Per-class stats
            cls_key = gt_name
            summary["per_class"][cls_key]["count"] += 1
            summary["per_class"][cls_key]["dice_sum"] += d
            summary["per_class"][cls_key]["jaccard_sum"] += j
            if match:
                summary["per_class"][cls_key]["tp"] += 1
            else:
                summary["per_class"][cls_key]["fn"] += 1
                if pred_name != "none":
                    pred_key = pred_name
                    if pred_key in summary["per_class"]:
                        summary["per_class"][pred_key]["fp"] += 1

            writer.writerow([
                img_path.name, gt_cls, gt_name,
                pred_cls, pred_name, f"{conf:.4f}",
                f"{d:.4f}", f"{j:.4f}", match,
            ])

            # Save overlay image
            if save_overlays:
                cls_overlay_dir = overlay_dir / gt_name
                cls_overlay_dir.mkdir(parents=True, exist_ok=True)

                overlay = img.copy()
                # Green semi-transparent mask for prediction
                green = np.zeros_like(overlay)
                green[:, :, 1] = 255
                pred_region = pred_mask.astype(bool)
                overlay[pred_region] = cv2.addWeighted(
                    overlay[pred_region], 0.6, green[pred_region], 0.4, 0
                )
                # Red contour for GT mask
                gt_contours, _ = cv2.findContours(
                    gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, gt_contours, -1, (0, 0, 255), 1)

                # Text label
                label_text = f"GT:{gt_name} P:{pred_name} c:{conf:.2f} D:{d:.2f}"
                cv2.putText(
                    overlay, label_text, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                )
                cv2.imwrite(str(cls_overlay_dir / img_path.name), overlay)

    # Finalize per-class averages
    for cls in summary["per_class"].values():
        n = cls["count"]
        cls["mean_dice"] = cls["dice_sum"] / n if n > 0 else 0.0
        cls["mean_jaccard"] = cls["jaccard_sum"] / n if n > 0 else 0.0
        del cls["dice_sum"]
        del cls["jaccard_sum"]

    summary["accuracy"] = summary["correct"] / summary["total"] if summary["total"] > 0 else 0.0

    # Save JSON summary
    summary_path = output_dir / "predictions_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] Prediction log saved: {csv_path}")
    print(f"[eval] Prediction summary saved: {summary_path}")
    print(f"[eval] Accuracy (top-1 class match): {summary['accuracy']:.4f}")
    print(f"[eval] No-detection images: {summary['no_detection']}/{summary['total']}")
    if save_overlays:
        print(f"[eval] Overlay images saved: {overlay_dir}")

    return csv_path


def save_evaluation_report(metrics: dict, output_dir: Path) -> None:
    """Save metrics as JSON and generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Report saved: {report_path}")

    # Plot per-class metrics if available
    if "classification" in metrics and "per_class" in metrics["classification"]:
        _plot_classification_metrics(metrics["classification"], output_dir)
    if "detection" in metrics and "per_class_mAP50_95" in metrics["detection"]:
        _plot_detection_metrics(metrics["detection"], output_dir)
    if "segmentation" in metrics and "per_class_mAP50_95" in metrics["segmentation"]:
        _plot_segmentation_metrics(metrics["segmentation"], output_dir)


def _plot_classification_metrics(cls_metrics: dict, output_dir: Path) -> None:
    """Generate classification metrics bar chart."""
    per_class = cls_metrics["per_class"]
    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1"] for c in classes]

    fig, ax = plt.subplots(figsize=(20, 6))
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "classification_metrics.png", dpi=150)
    plt.close()
    print(f"[eval] Plot saved: {output_dir / 'classification_metrics.png'}")


def _plot_detection_metrics(det_metrics: dict, output_dir: Path) -> None:
    """Generate detection mAP bar chart."""
    per_class = det_metrics["per_class_mAP50_95"]
    classes = list(per_class.keys())
    values = [per_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(classes, values, alpha=0.8, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("mAP@50:95")
    ax.set_title("Per-Class Detection mAP@50:95")
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "detection_map.png", dpi=150)
    plt.close()


def _plot_segmentation_metrics(seg_metrics: dict, output_dir: Path) -> None:
    """Generate segmentation mAP bar chart."""
    per_class = seg_metrics["per_class_mAP50_95"]
    classes = list(per_class.keys())
    values = [per_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(classes, values, alpha=0.8, color="coral")
    ax.set_xlabel("Class")
    ax.set_ylabel("mAP@50:95")
    ax.set_title("Per-Class Segmentation mAP@50:95")
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_map.png", dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26n-seg model")
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "yolo26n-seg-best.pt"),
        help="Path to trained model",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument(
        "--log-predictions", action="store_true",
        help="Run inference and log per-image predictions (masks, classes, scores)",
    )
    parser.add_argument(
        "--save-overlays", action="store_true",
        help="Save mask overlay images alongside prediction log",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = evaluate_model(args.model, split=args.split, device=args.device)

    # Compute pixel-level Dice and Jaccard
    print("\n[eval] Computing pixel-level Dice & Jaccard (IoU)...")
    mask_metrics = compute_mask_metrics(
        args.model, split=args.split, device=args.device
    )
    metrics["segmentation"]["dice_coefficient"] = mask_metrics["mean_dice"]
    metrics["segmentation"]["jaccard_index_iou"] = mask_metrics["mean_jaccard"]
    metrics["segmentation"]["per_class_dice"] = mask_metrics["per_class_dice"]
    metrics["segmentation"]["per_class_jaccard"] = mask_metrics["per_class_jaccard"]
    # Also add to detection (same masks, different framing)
    metrics["detection"]["dice_coefficient"] = mask_metrics["mean_dice"]
    metrics["detection"]["jaccard_index_iou"] = mask_metrics["mean_jaccard"]

    eval_dir = RESULTS_DIR / "evaluation"
    save_evaluation_report(metrics, eval_dir)

    # Log per-image predictions for inference quality analysis (test only)
    if args.log_predictions or args.split == "test":
        print("\n[eval] Logging per-image predictions for inference quality analysis...")
        log_test_predictions(
            args.model,
            split=args.split,
            device=args.device,
            save_overlays=args.save_overlays,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if "classification" in metrics:
        c = metrics["classification"]
        print(f"\nClassification:")
        print(f"  Accuracy:    {c['accuracy']:.4f}")
        print(f"  Precision:   {c['precision_macro']:.4f}")
        print(f"  Recall:      {c['recall_macro']:.4f}")
        print(f"  F1-Score:    {c['f1_macro']:.4f}")
        print(f"  Specificity: {c['specificity_macro']:.4f}")

    d = metrics["detection"]
    print(f"\nDetection:")
    print(f"  mAP@50:      {d['mAP50']:.4f}")
    print(f"  mAP@50:95:   {d['mAP50_95']:.4f}")

    s = metrics["segmentation"]
    print(f"\nSegmentation:")
    print(f"  mAP@50:      {s['mAP50']:.4f}")
    print(f"  mAP@50:95:   {s['mAP50_95']:.4f}")
    if "dice_coefficient" in s:
        print(f"  Dice Coeff:  {s['dice_coefficient']:.4f}")
    if "jaccard_index_iou" in s:
        print(f"  Jaccard/IoU: {s['jaccard_index_iou']:.4f}")


if __name__ == "__main__":
    main()
