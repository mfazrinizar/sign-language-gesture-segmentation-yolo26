"""
Streamlit EDA Dashboard for Sign Language Gesture Segmentation.

Tabs:
    1. Dataset Overview — class distribution, split statistics
    2. Image Explorer — browse samples per class (color + binary + mask overlay)
    3. Statistics — image intensity distributions, mask quality analysis
    4. Training Results — loss curves, metrics (if available)

Usage:
    streamlit run src/eda/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    INDEX_TO_NAME,
    RAW_BINARY_FOLDER,
    RAW_COLOR_FOLDER,
    RAW_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    YOLO_SEG_DIR,
)
from src.eda.analysis import (
    get_class_distribution,
    get_image_statistics,
    get_mask_quality_stats,
    get_sample_images,
)


st.set_page_config(
    page_title="Sign Language Gesture — EDA",
    layout="wide",
)

st.title("Sign Language Gesture Segmentation — EDA Dashboard")


#  Tab Layout 
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Image Explorer",
    "Statistics",
    "Training Results",
])


#  Tab 1: Dataset Overview 
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Classes", "37")
    col2.metric("Images per Class", "1,500")
    col3.metric("Total Images", "55,500")

    st.subheader("Class Distribution by Split")
    if SPLITS_DIR.exists():
        dist = get_class_distribution(SPLITS_DIR)
        if dist:
            # Create DataFrame
            df_rows = []
            for split, counts in dist.items():
                for cls, count in counts.items():
                    df_rows.append({"Split": split, "Class": cls, "Count": count})
            df = pd.DataFrame(df_rows)

            # Summary
            summary = {s: sum(c.values()) for s, c in dist.items()}
            cols = st.columns(len(summary))
            for col, (split, total) in zip(cols, summary.items()):
                col.metric(f"{split.capitalize()} Set", f"{total:,}")

            # Bar chart
            fig, ax = plt.subplots(figsize=(18, 5))
            pivot = df.pivot(index="Class", columns="Split", values="Count").fillna(0)
            pivot = pivot.reindex(CLASS_FOLDERS)
            pivot.plot(kind="bar", ax=ax, alpha=0.8)
            ax.set_title("Samples per Class per Split")
            ax.set_ylabel("Count")
            ax.set_xlabel("Class")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Split data not found. Run `python -m src.data.split` first.")


#  Tab 2: Image Explorer 
with tab2:
    st.header("Image Explorer")

    selected_class = st.selectbox("Select Class", CLASS_FOLDERS)
    n_samples = st.slider("Number of samples", 1, 10, 5)

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
            st.subheader("Binary (Pre-Processed) Images")
            binary_imgs = get_sample_images(RAW_DIR / RAW_BINARY_FOLDER, selected_class, n_samples)
            if binary_imgs:
                cols = st.columns(min(n_samples, 5))
                for i, img in enumerate(binary_imgs):
                    with cols[i % 5]:
                        st.image(img, caption=f"Sample {i+1}", use_container_width=True)

        # Show mask overlay if masks exist
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

                    # Parse label and draw polygon
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
    else:
        st.warning("Raw data not found. Run `python -m src.data.download` first.")


#  Tab 3: Statistics 
with tab3:
    st.header("Image Statistics")

    if RAW_DIR.exists():
        color_dir = RAW_DIR / RAW_COLOR_FOLDER
        if st.button("Compute Statistics (may take a moment)"):
            with st.spinner("Computing..."):
                stats = get_image_statistics(color_dir, n_samples=500)

            st.subheader("Image Dimensions")
            dim_df = pd.DataFrame({
                "Metric": ["Mean", "Std", "Min", "Max"],
                "Height": [stats["height"]["mean"], stats["height"]["std"], stats["height"]["min"], stats["height"]["max"]],
                "Width": [stats["width"]["mean"], stats["width"]["std"], stats["width"]["min"], stats["width"]["max"]],
            })
            st.table(dim_df)

            st.subheader("Channel Intensity (RGB)")
            int_df = pd.DataFrame({
                "Channel": list(stats["intensity_mean"].keys()),
                "Mean": list(stats["intensity_mean"].values()),
                "Std": list(stats["intensity_std"].values()),
            })
            st.table(int_df)

    st.subheader("Mask Quality Analysis")
    labels_dir = YOLO_SEG_DIR / "all_labels"
    if labels_dir.exists():
        if st.button("Analyze Mask Quality"):
            with st.spinner("Analyzing masks..."):
                mask_stats = get_mask_quality_stats(labels_dir)

            st.json(mask_stats)

            if "normalized_area" in mask_stats:
                st.metric("Avg Normalized Area", f"{mask_stats['normalized_area']['mean']:.4f}")
            if "points_per_polygon" in mask_stats:
                st.metric("Avg Points per Polygon", f"{mask_stats['points_per_polygon']['mean']:.1f}")
    else:
        st.info("Generate masks first with `python -m src.data.mask_generator`")


#  Tab 4: Training Results 
with tab4:
    st.header("Training Results")

    results_seg = RESULTS_DIR / "seg" / "yolo26n-seg"
    if results_seg.exists():
        # Show training plots if they exist
        plot_files = list(results_seg.glob("*.png"))
        if plot_files:
            st.subheader("Training Plots")
            for plot_path in sorted(plot_files):
                st.image(str(plot_path), caption=plot_path.stem)

        # Show CSV results if available
        csv_path = results_seg / "results.csv"
        if csv_path.exists():
            st.subheader("Training Metrics")
            results_df = pd.read_csv(csv_path)
            results_df.columns = results_df.columns.str.strip()

            # Plot loss curves
            loss_cols = [c for c in results_df.columns if "loss" in c.lower()]
            if loss_cols:
                fig, axes = plt.subplots(1, len(loss_cols), figsize=(5 * len(loss_cols), 4))
                if len(loss_cols) == 1:
                    axes = [axes]
                for ax, col in zip(axes, loss_cols):
                    ax.plot(results_df[col])
                    ax.set_title(col)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                plt.tight_layout()
                st.pyplot(fig)

        # Show evaluation report
        eval_report = RESULTS_DIR / "evaluation" / "evaluation_report.json"
        if eval_report.exists():
            st.subheader("Evaluation Report")
            with open(eval_report) as f:
                report = json.load(f)
            st.json(report)
    else:
        st.info("No training results found yet. Train the model first with `python -m src.training.train_seg`.")
