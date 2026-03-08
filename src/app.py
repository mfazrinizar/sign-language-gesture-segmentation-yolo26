"""
Sign Language Gesture Segmentation — Combined Streamlit Dashboard.

Tabs:
    1. Live Demo      — Real-time webcam inference + image upload + model browser
    2. Dataset Overview — Class distribution, split statistics
    3. Image Explorer  — Browse samples per class (color + binary + mask overlay)
    4. Training Results — Loss curves, metrics plots, confusion matrices
    5. Analysis        — Per-class evaluation metrics, prediction analysis

Usage:
    conda activate sign-yolo26
    streamlit run src/app.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The vendored ultralytics/ folder shadows the installed package.
# Add ultralytics/ so `from ultralytics import YOLO` resolves correctly.
_ULTRA_DIR = PROJECT_ROOT / "ultralytics"
if _ULTRA_DIR.is_dir() and str(_ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(_ULTRA_DIR))

import cv2
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
from PIL import Image

from src.config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    INDEX_TO_NAME,
    NUM_CLASSES,
    RAW_BINARY_FOLDER,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    YOLO_SEG_DIR,
    MODELS_DIR,
)


# Page config

st.set_page_config(
    page_title="Sign Language Gesture Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Color palette for classes (deterministic)

_RNG_COLORS = np.random.RandomState(42)
CLASS_COLORS = {
    name: tuple(_RNG_COLORS.randint(60, 255, 3).tolist()) for name in CLASS_FOLDERS
}



# Cached helpers

@st.cache_resource(show_spinner="Loading YOLO model …")
def load_model(model_path: str):
    """Load a YOLO model once and cache it."""
    from ultralytics import YOLO
    return YOLO(model_path)


def list_available_models() -> list[Path]:
    """Return .pt files found in models/."""
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.pt"))


def run_inference(model, img_bgr: np.ndarray, conf: float = 0.90):
    """Run segmentation inference. Returns list of result dicts."""
    results = model.predict(img_bgr, conf=conf, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            cls_name = INDEX_TO_NAME.get(cls_id, str(cls_id))
            conf_val = float(r.boxes.conf[i].item())
            box = r.boxes.xyxy[i].cpu().numpy().astype(int)
            mask = None
            if r.masks is not None and i < len(r.masks):
                mask = r.masks.data[i].cpu().numpy()
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf_val,
                "box": box,
                "mask": mask,
            })
    return detections


def draw_detections(img_bgr: np.ndarray, detections: list, alpha: float = 0.45) -> np.ndarray:
    """Draw bounding boxes + mask overlays on image."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    overlay = vis.copy()

    for det in detections:
        color = CLASS_COLORS.get(det["class_name"], (0, 255, 0))
        x1, y1, x2, y2 = det["box"]

        # Mask overlay
        if det["mask"] is not None:
            mask_resized = cv2.resize(
                det["mask"].astype(np.float32), (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = color

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
    return vis



# Sidebar — model selector

st.sidebar.title("Sign Language Segmentation")
st.sidebar.markdown("---")

models = list_available_models()
model_names = [p.name for p in models]

if model_names:
    selected_model_name = st.sidebar.selectbox(
        "🔧 Select Model", model_names, index=0,
    )
    selected_model_path = MODELS_DIR / selected_model_name
else:
    st.sidebar.warning("No models found in `models/`")
    selected_model_path = None

conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 1.0, 0.90, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Project root:** `{PROJECT_ROOT}`  \n"
    f"**Classes:** {NUM_CLASSES}  \n"
    f"**Models dir:** `models/`"
)



# TABS

tab_demo, tab_overview, tab_explorer, tab_training, tab_analysis = st.tabs([
    "Live Demo",
    "Dataset Overview",
    "Image Explorer",
    "Training Results",
    "Analysis",
])



# TAB 1 — LIVE DEMO

with tab_demo:
    st.header("Real-Time Gesture Segmentation Demo")

    if selected_model_path is None:
        st.error("No model selected. Place `.pt` files in the `models/` directory.")
        st.stop()

    model = load_model(str(selected_model_path))

    demo_mode = st.radio(
        "Input source",
        ["📷 Upload Image", "🎥 Webcam (Live)"],
        horizontal=True,
    )

    if demo_mode == "📷 Upload Image":
        uploaded = st.file_uploader(
            "Browse an image …",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Running inference …"):
                detections = run_inference(model, img_bgr, conf=conf_threshold)
                vis = draw_detections(img_bgr, detections)

            col_in, col_out = st.columns(2)
            with col_in:
                st.subheader("Input")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_out:
                st.subheader("Prediction")
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

            if detections:
                st.subheader("Detection Results")
                rows = []
                for d in detections:
                    rows.append({
                        "Class": d["class_name"],
                        "Confidence": f"{d['confidence']:.3f}",
                        "Box": f"({d['box'][0]}, {d['box'][1]}) → ({d['box'][2]}, {d['box'][3]})",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No detections at current confidence threshold.")

    else:  # Webcam
        st.markdown(
            "> Press **Start** to begin webcam capture. "
            "Press **Stop** to end. Inference runs on every captured frame."
        )

        col_ctrl, col_fps = st.columns([3, 1])
        with col_ctrl:
            start_btn = st.button("▶️ Start Webcam", type="primary")
            stop_btn = st.button("⏹️ Stop Webcam")
        with col_fps:
            fps_display = st.empty()

        frame_holder = st.empty()
        info_holder = st.empty()

        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        if start_btn:
            st.session_state.webcam_running = True
        if stop_btn:
            st.session_state.webcam_running = False

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Check camera permissions.")
                st.session_state.webcam_running = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                try:
                    while st.session_state.webcam_running:
                        t0 = time.perf_counter()
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to read frame.")
                            break

                        detections = run_inference(model, frame, conf=conf_threshold)
                        vis = draw_detections(frame, detections)

                        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
                        fps_display.metric("FPS", f"{fps:.1f}")

                        frame_holder.image(
                            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True,
                        )

                        if detections:
                            labels = ", ".join(
                                f"**{d['class_name']}** ({d['confidence']:.2f})"
                                for d in detections
                            )
                            info_holder.markdown(f"Detected: {labels}")
                        else:
                            info_holder.info("No gesture detected.")

                except Exception:
                    pass
                finally:
                    cap.release()
                    st.session_state.webcam_running = False



# TAB 2 — DATASET OVERVIEW

with tab_overview:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Classes", "37")
    col2.metric("Images per Class", "1,500")
    col3.metric("Total Images", "55,500")

    st.subheader("Class Distribution by Split")
    if SPLITS_DIR.exists():
        from src.eda.analysis import get_class_distribution
        dist = get_class_distribution(SPLITS_DIR)
        if dist:
            df_rows = []
            for split, counts in dist.items():
                for cls, count in counts.items():
                    df_rows.append({"Split": split, "Class": cls, "Count": count})
            df = pd.DataFrame(df_rows)

            summary = {s: sum(c.values()) for s, c in dist.items()}
            cols = st.columns(len(summary))
            for col, (split, total) in zip(cols, summary.items()):
                col.metric(f"{split.capitalize()} Set", f"{total:,}")

            fig, ax = plt.subplots(figsize=(18, 5))
            pivot = df.pivot(index="Class", columns="Split", values="Count").fillna(0)
            pivot = pivot.reindex(CLASS_FOLDERS)
            pivot.plot(kind="bar", ax=ax, alpha=0.8, color=["#4C78A8", "#F58518", "#54A24B"])
            ax.set_title("Samples per Class per Split", fontsize=14)
            ax.set_ylabel("Count")
            ax.set_xlabel("Class")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Split data not found. Run `python -m src.data.split` first.")



# TAB 3 — IMAGE EXPLORER

with tab_explorer:
    st.header("Image Explorer")

    from src.eda.analysis import get_sample_images

    selected_class = st.selectbox("Select Class", CLASS_FOLDERS, key="explorer_class")
    n_samples = st.slider("Number of samples", 1, 10, 5, key="explorer_n")

    if RAW_DIR.exists():
        col_color, col_binary = st.columns(2)

        with col_color:
            st.subheader("Color Images")
            color_imgs = get_sample_images(RAW_DIR / RAW_COLOR_FOLDER, selected_class, n_samples)
            if color_imgs:
                cols = st.columns(min(n_samples, 5))
                for i, img in enumerate(color_imgs):
                    with cols[i % 5]:
                        st.image(img, caption=f"Sample {i+1}", use_container_width=True)

        with col_binary:
            st.subheader("Binary (Pre-Processed)")
            binary_imgs = get_sample_images(RAW_DIR / RAW_BINARY_FOLDER, selected_class, n_samples)
            if binary_imgs:
                cols = st.columns(min(n_samples, 5))
                for i, img in enumerate(binary_imgs):
                    with cols[i % 5]:
                        st.image(img, caption=f"Sample {i+1}", use_container_width=True)

        # Mask overlay
        labels_dir = YOLO_SEG_DIR / "all_labels"
        if labels_dir.exists():
            st.subheader("Mask Overlay")
            color_dir = RAW_DIR / RAW_COLOR_FOLDER / selected_class
            color_files = sorted(color_dir.glob("*"))[:n_samples]

            cols = st.columns(min(n_samples, 5))
            for i, img_path in enumerate(color_files):
                label_path = labels_dir / f"{selected_class}_{img_path.stem}.txt"
                if label_path.exists():
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]
                    content = label_path.read_text().strip()
                    parts = content.split()
                    coords = [float(x) for x in parts[1:]]
                    points = np.array(
                        [(int(coords[j] * w), int(coords[j+1] * h)) for j in range(0, len(coords), 2)],
                        dtype=np.int32,
                    )
                    overlay = img_rgb.copy()
                    cv2.fillPoly(overlay, [points], (0, 255, 0))
                    result = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
                    cv2.polylines(result, [points], True, (0, 200, 0), 1)
                    with cols[i % 5]:
                        st.image(result, caption=f"Mask {i+1}", use_container_width=True)

        # --- Test Prediction Overlays ---
        overlay_dir = RESULTS_DIR / "evaluation" / "test_predictions" / "overlays"
        if overlay_dir.exists():
            st.subheader("Test Prediction Overlays")
            cls_overlay_dir = overlay_dir / selected_class
            if cls_overlay_dir.exists():
                overlay_files = sorted(cls_overlay_dir.glob("*.jpg"))[:n_samples]
                if overlay_files:
                    cols = st.columns(min(len(overlay_files), 5))
                    for i, f in enumerate(overlay_files):
                        with cols[i % 5]:
                            st.image(str(f), caption=f.stem, use_container_width=True)
    else:
        st.warning("Raw data not found. Run `python -m src.data.download` first.")



# TAB 4 — TRAINING RESULTS

with tab_training:
    st.header("Training Results")

    # Discover available training runs
    seg_dir = RESULTS_DIR / "seg"
    run_dirs = sorted(seg_dir.glob("yolo26n-seg-*")) if seg_dir.exists() else []

    if not run_dirs:
        st.info("No training runs found in `results/seg/`.")
    else:
        run_names = [d.name for d in run_dirs]
        selected_run = st.selectbox("Training Run", run_names, index=len(run_names) - 1)
        run_path = seg_dir / selected_run

        # -- Training CSV metrics --
        csv_path = run_path / "results.csv"
        if csv_path.exists():
            results_df = pd.read_csv(csv_path)
            results_df.columns = results_df.columns.str.strip()

            st.subheader("Loss Curves")
            loss_cols = [c for c in results_df.columns if "loss" in c.lower() and "train" in c.lower()]
            val_loss_cols = [c for c in results_df.columns if "loss" in c.lower() and "val" in c.lower()]

            if loss_cols:
                n_loss = len(loss_cols)
                fig, axes = plt.subplots(1, n_loss, figsize=(5 * n_loss, 4))
                if n_loss == 1:
                    axes = [axes]
                for ax, col in zip(axes, loss_cols):
                    ax.plot(results_df["epoch"], results_df[col], label="Train", color="#4C78A8", linewidth=1.5)
                    # Try to find matching val loss
                    val_col = col.replace("train/", "val/")
                    if val_col in results_df.columns:
                        ax.plot(results_df["epoch"], results_df[val_col], label="Val", color="#F58518", linewidth=1.5)
                    short_name = col.split("/")[-1] if "/" in col else col
                    ax.set_title(short_name, fontsize=12)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Detection & Segmentation Metrics")
            metric_cols = [c for c in results_df.columns if c.startswith("metrics/")]
            if metric_cols:
                box_cols = [c for c in metric_cols if "(B)" in c]
                mask_cols = [c for c in metric_cols if "(M)" in c]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                for col in box_cols:
                    short = col.replace("metrics/", "").replace("(B)", "")
                    ax1.plot(results_df["epoch"], results_df[col], label=short, linewidth=1.5)
                ax1.set_title("Box Metrics", fontsize=12)
                ax1.set_xlabel("Epoch")
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1.05)

                for col in mask_cols:
                    short = col.replace("metrics/", "").replace("(M)", "")
                    ax2.plot(results_df["epoch"], results_df[col], label=short, linewidth=1.5)
                ax2.set_title("Mask Metrics", fontsize=12)
                ax2.set_xlabel("Epoch")
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1.05)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Learning Rate Schedule")
            lr_cols = [c for c in results_df.columns if c.startswith("lr/")]
            if lr_cols:
                fig, ax = plt.subplots(figsize=(10, 3))
                for col in lr_cols:
                    ax.plot(results_df["epoch"], results_df[col], label=col.replace("lr/", ""), linewidth=1.5)
                ax.set_title("Learning Rate", fontsize=12)
                ax.set_xlabel("Epoch")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # -- Training images gallery --
        st.subheader("Training Visualizations")
        img_types = {
            "Confusion Matrix": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
            "Overall Results": ["results.png", "labels.jpg"],
            "PR / F1 Curves (Box)": ["BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
            "PR / F1 Curves (Mask)": ["MaskPR_curve.png", "MaskF1_curve.png", "MaskP_curve.png", "MaskR_curve.png"],
        }

        for section, filenames in img_types.items():
            existing = [run_path / f for f in filenames if (run_path / f).exists()]
            if existing:
                with st.expander(section, expanded=(section == "Confusion Matrix")):
                    cols = st.columns(min(len(existing), 2))
                    for i, p in enumerate(existing):
                        with cols[i % 2]:
                            st.image(str(p), caption=p.stem, use_container_width=True)

        # -- Batch samples --
        train_batches = sorted(run_path.glob("train_batch*.jpg"))
        val_batches = sorted(run_path.glob("val_batch*.jpg"))
        if train_batches or val_batches:
            with st.expander("Training / Validation Batch Samples"):
                if train_batches:
                    st.markdown("**Training batches**")
                    cols = st.columns(min(len(train_batches), 3))
                    for i, p in enumerate(train_batches[:6]):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.stem, use_container_width=True)
                if val_batches:
                    st.markdown("**Validation batches**")
                    cols = st.columns(min(len(val_batches), 3))
                    for i, p in enumerate(val_batches[:6]):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.stem, use_container_width=True)

        # -- args.yaml --
        args_path = run_path / "args.yaml"
        if args_path.exists():
            with st.expander("Training Configuration (args.yaml)"):
                st.code(args_path.read_text(), language="yaml")



# TAB 5 — ANALYSIS

with tab_analysis:
    st.header("Evaluation Analysis")

    # --- Load evaluation report ---
    eval_paths = [
        RESULTS_DIR / "evaluation_report.json",
        RESULTS_DIR / "evaluation" / "evaluation_report.json",
    ]
    eval_report = None
    for ep in eval_paths:
        if ep.exists():
            with open(ep) as f:
                eval_report = json.load(f)
            break

    if eval_report is None:
        st.warning("No evaluation report found.")
    else:
        # ---- High-level metrics ----
        st.subheader("Overall Metrics")
        det = eval_report.get("detection", {})
        seg = eval_report.get("segmentation", {})
        clf = eval_report.get("classification", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detection mAP@50", f"{det.get('mAP50', 0):.4f}")
        c2.metric("Detection mAP@50-95", f"{det.get('mAP50_95', 0):.4f}")
        c3.metric("Segmentation mAP@50", f"{seg.get('mAP50', 0):.4f}")
        c4.metric("Segmentation mAP@50-95", f"{seg.get('mAP50_95', 0):.4f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Precision", f"{det.get('precision', 0):.4f}")
        c6.metric("Recall", f"{det.get('recall', 0):.4f}")
        c7.metric("Dice Coefficient", f"{det.get('dice_coefficient', 0):.4f}")
        c8.metric("Jaccard (IoU)", f"{det.get('jaccard_index_iou', 0):.4f}")

        if clf:
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Classification Accuracy", f"{clf.get('accuracy', 0):.4f}")
            c10.metric("Precision (macro)", f"{clf.get('precision_macro', 0):.4f}")
            c11.metric("Recall (macro)", f"{clf.get('recall_macro', 0):.4f}")
            c12.metric("F1 (macro)", f"{clf.get('f1_macro', 0):.4f}")

        # ---- Per-class mAP comparison: Detection vs Segmentation ----
        st.subheader("Per-Class mAP@50-95 — Detection vs Segmentation")
        det_map = det.get("per_class_mAP50_95", {})
        seg_map = seg.get("per_class_mAP50_95", {})
        if det_map and seg_map:
            classes = list(det_map.keys())
            df_map = pd.DataFrame({
                "Class": classes,
                "Detection mAP": [det_map[c] for c in classes],
                "Segmentation mAP": [seg_map.get(c, 0) for c in classes],
            })
            df_map["Gap"] = df_map["Detection mAP"] - df_map["Segmentation mAP"]
            df_map = df_map.sort_values("Class").reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(18, 6))
            x = np.arange(len(df_map))
            w = 0.35
            ax.bar(x - w/2, df_map["Detection mAP"], w, label="Detection", color="#4C78A8", alpha=0.85)
            ax.bar(x + w/2, df_map["Segmentation mAP"], w, label="Segmentation", color="#F58518", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(df_map["Class"], rotation=45, ha="right")
            ax.set_ylabel("mAP@50-95")
            ax.set_title("Per-Class mAP@50-95", fontsize=14)
            ax.legend()
            ax.set_ylim(0.75, 1.01)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Show classes with largest gap
            st.markdown("**Classes with largest Det–Seg gap:**")
            top_gap = df_map.nlargest(10, "Gap")[["Class", "Detection mAP", "Segmentation mAP", "Gap"]]
            st.dataframe(top_gap.style.format({"Detection mAP": "{:.4f}", "Segmentation mAP": "{:.4f}", "Gap": "{:.4f}"}), use_container_width=True)

        # ---- Per-class Dice & Jaccard ----
        st.subheader("Per-Class Dice & Jaccard (IoU)")
        per_dice = det.get("per_class_dice") or seg.get("per_class_dice", {})
        per_jaccard = det.get("per_class_jaccard") or seg.get("per_class_jaccard", {})
        if per_dice and per_jaccard:
            classes = sorted(per_dice.keys())
            df_dj = pd.DataFrame({
                "Class": classes,
                "Dice": [per_dice[c] for c in classes],
                "Jaccard (IoU)": [per_jaccard[c] for c in classes],
            })

            fig, ax = plt.subplots(figsize=(18, 5))
            x = np.arange(len(df_dj))
            w = 0.35
            ax.bar(x - w/2, df_dj["Dice"], w, label="Dice", color="#54A24B", alpha=0.85)
            ax.bar(x + w/2, df_dj["Jaccard (IoU)"], w, label="Jaccard (IoU)", color="#E45756", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(df_dj["Class"], rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.set_title("Per-Class Dice & Jaccard", fontsize=14)
            ax.legend()
            ax.set_ylim(0.8, 1.0)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Bottom performers
            st.markdown("**Lowest Dice scores:**")
            low_dice = df_dj.nsmallest(10, "Dice")[["Class", "Dice", "Jaccard (IoU)"]]
            st.dataframe(low_dice.style.format({"Dice": "{:.4f}", "Jaccard (IoU)": "{:.4f}"}), use_container_width=True)

        # ---- Dice heatmap ----
        if per_dice:
            st.subheader("Dice Score Heatmap")
            classes_sorted = sorted(per_dice.keys())
            dice_vals = np.array([[per_dice[c] for c in classes_sorted]])
            fig, ax = plt.subplots(figsize=(18, 2))
            sns.heatmap(
                dice_vals, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=classes_sorted, yticklabels=["Dice"],
                ax=ax, vmin=0.90, vmax=1.0, linewidths=0.5,
                annot_kws={"size": 8},
            )
            ax.set_title("Dice by Class", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # --- Predictions summary ---
    st.markdown("---")
    st.subheader("Predictions Summary")
    pred_summary_paths = [
        RESULTS_DIR / "predictions_summary.json",
        RESULTS_DIR / "evaluation" / "test_predictions" / "predictions_summary.json",
    ]
    pred_summary = None
    for pp in pred_summary_paths:
        if pp.exists():
            with open(pp) as f:
                pred_summary = json.load(f)
            break

    if pred_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", f"{pred_summary.get('total', 0):,}")
        c2.metric("Correct", f"{pred_summary.get('correct', 0):,}")
        c3.metric("No Detection", f"{pred_summary.get('no_detection', 0):,}")
        accuracy = pred_summary.get("accuracy", pred_summary.get("correct", 0) / max(pred_summary.get("total", 1), 1))
        c4.metric("Accuracy", f"{accuracy:.4f}")

        per_class = pred_summary.get("per_class", {})
        if per_class:
            classes = sorted(per_class.keys())
            df_pred = pd.DataFrame([
                {
                    "Class": c,
                    "Count": per_class[c].get("count", 0),
                    "TP": per_class[c].get("tp", 0),
                    "FP": per_class[c].get("fp", 0),
                    "FN": per_class[c].get("fn", 0),
                    "Mean Dice": per_class[c].get("mean_dice", 0),
                    "Mean Jaccard": per_class[c].get("mean_jaccard", 0),
                }
                for c in classes
            ])

            # Distribution of mean dice per class
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            colors_dice = plt.cm.RdYlGn((df_pred["Mean Dice"] - df_pred["Mean Dice"].min()) /
                                         max(df_pred["Mean Dice"].max() - df_pred["Mean Dice"].min(), 1e-6))
            ax1.barh(df_pred["Class"], df_pred["Mean Dice"], color=colors_dice)
            ax1.set_xlabel("Mean Dice")
            ax1.set_title("Mean Dice per Class", fontsize=12)
            ax1.set_xlim(0.90, 1.0)
            ax1.grid(axis="x", alpha=0.3)
            ax1.invert_yaxis()

            colors_jac = plt.cm.RdYlGn((df_pred["Mean Jaccard"] - df_pred["Mean Jaccard"].min()) /
                                        max(df_pred["Mean Jaccard"].max() - df_pred["Mean Jaccard"].min(), 1e-6))
            ax2.barh(df_pred["Class"], df_pred["Mean Jaccard"], color=colors_jac)
            ax2.set_xlabel("Mean Jaccard (IoU)")
            ax2.set_title("Mean Jaccard per Class", fontsize=12)
            ax2.set_xlim(0.85, 1.0)
            ax2.grid(axis="x", alpha=0.3)
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Full Per-Class Prediction Table"):
                st.dataframe(
                    df_pred.style.format({
                        "Mean Dice": "{:.4f}",
                        "Mean Jaccard": "{:.4f}",
                    }),
                    use_container_width=True,
                    height=600,
                )

    # --- Predictions log (CSV) analysis ---
    st.markdown("---")
    st.subheader("Predictions Log Analysis")
    pred_log_path = RESULTS_DIR / "predictions_log.csv"
    if pred_log_path.exists():
        df_log = pd.read_csv(pred_log_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", f"{len(df_log):,}")
        c2.metric("Mean Confidence", f"{df_log['confidence'].mean():.4f}")
        c3.metric("Mean Dice", f"{df_log['dice'].mean():.4f}")

        # Confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.hist(df_log["confidence"], bins=50, color="#4C78A8", alpha=0.8, edgecolor="white")
        ax1.set_title("Confidence Distribution", fontsize=12)
        ax1.set_xlabel("Confidence")
        ax1.set_ylabel("Count")
        ax1.axvline(df_log["confidence"].mean(), color="red", linestyle="--", label=f"Mean={df_log['confidence'].mean():.3f}")
        ax1.legend()

        ax2.hist(df_log["dice"], bins=50, color="#54A24B", alpha=0.8, edgecolor="white")
        ax2.set_title("Dice Score Distribution", fontsize=12)
        ax2.set_xlabel("Dice")
        ax2.set_ylabel("Count")
        ax2.axvline(df_log["dice"].mean(), color="red", linestyle="--", label=f"Mean={df_log['dice'].mean():.3f}")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Confidence vs Dice scatter (sampled for performance)
        st.subheader("Confidence vs Dice")
        sample_n = min(2000, len(df_log))
        df_sample = df_log.sample(sample_n, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df_sample["confidence"], df_sample["dice"],
            c=df_sample["gt_class_id"], cmap="tab20", alpha=0.5, s=10,
        )
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Dice Score", fontsize=11)
        ax.set_title("Confidence vs Dice (sampled)", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Per-class box plot of dice
        st.subheader("Dice Distribution by Class")
        fig, ax = plt.subplots(figsize=(18, 6))
        class_order = sorted(df_log["gt_class_name"].unique())
        df_log["gt_class_name"] = pd.Categorical(df_log["gt_class_name"], categories=class_order)
        bp = df_log.boxplot(column="dice", by="gt_class_name", ax=ax, grid=False, rot=45)
        ax.set_title("Dice Distribution by Class", fontsize=13)
        ax.set_xlabel("Class")
        ax.set_ylabel("Dice")
        ax.set_ylim(0.85, 1.0)
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Show worst predictions
        with st.expander("Worst Predictions (lowest dice scores)"):
            worst = df_log.nsmallest(20, "dice")[["image", "gt_class_name", "pred_class_name", "confidence", "dice", "jaccard"]]
            st.dataframe(worst.style.format({"confidence": "{:.4f}", "dice": "{:.4f}", "jaccard": "{:.4f}"}), use_container_width=True)

    else:
        st.info("No predictions log CSV found.")

    # --- Evaluation images ---
    eval_img_dir = RESULTS_DIR / "evaluation"
    eval_images = list(eval_img_dir.glob("*.png")) if eval_img_dir.exists() else []
    if eval_images:
        st.markdown("---")
        st.subheader("Evaluation Plots")
        cols = st.columns(min(len(eval_images), 3))
        for i, p in enumerate(sorted(eval_images)):
            with cols[i % 3]:
                st.image(str(p), caption=p.stem, use_container_width=True)
