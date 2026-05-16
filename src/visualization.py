"""
Visualization helpers.

Two flavors:
* Matplotlib/Seaborn figures, saved to disk for the HTML report.
* Plotly figures, returned in-memory for the Streamlit dashboard.

Each Matplotlib helper accepts an output Path and writes a PNG. They never
call plt.show() so they're safe to call in headless environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluator import EvaluationResult
from . import metrics as M
from .data_loader import all_image_ids

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Matplotlib (file-on-disk) helpers
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm: pd.DataFrame, out_path: Path) -> Path:
    """Heatmap of the confusion matrix; rows=GT, cols=Prediction."""
    fig, ax = plt.subplots(figsize=(max(6, len(cm.columns) * 0.8),
                                    max(5, len(cm.index) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_class_metrics(by_class: pd.DataFrame, out_path: Path) -> Path:
    """Grouped bar chart of precision/recall/F1 per class."""
    fig, ax = plt.subplots(figsize=(max(7, len(by_class) * 1.2), 5))
    if by_class.empty:
        ax.text(0.5, 0.5, "No classes evaluated", ha="center", va="center")
    else:
        df_long = by_class.melt(
            id_vars=["class"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(data=df_long, x="class", y="value", hue="metric", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-class metrics")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_condition_metrics(
    by_condition: Dict[str, pd.DataFrame], out_path: Path
) -> Path:
    """One row of subplots, one per condition, F1 per slice."""
    conds = [c for c, df in by_condition.items() if not df.empty]
    if not conds:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No condition metadata provided",
                ha="center", va="center")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4),
                             squeeze=False)
    for ax, cond in zip(axes[0], conds):
        df = by_condition[cond]
        sns.barplot(data=df, x=cond, y="f1", ax=ax,
                    color="#4c78a8")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"F1 by {cond}")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_pr_curves(
    result: EvaluationResult, out_path: Path, max_classes: int = 8
) -> Path:
    """Precision-recall curves for the top-supported classes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    by_class = result.by_class
    if by_class.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        top = by_class.sort_values("support", ascending=False).head(max_classes)
        for _, row in top.iterrows():
            cls = row["class"]
            prec, rec = _build_pr_for_class(result, cls)
            if prec.size == 0:
                continue
            ax.plot(rec, prec, label=f"{cls} (AP50={result.ap_by_class.get(cls, {}).get('ap50', 0):.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_title("Precision-Recall curves (at IoU=0.5)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _build_pr_for_class(result: EvaluationResult, cls: str):
    """Helper: build a PR curve from already-matched predictions DataFrame."""
    df = result.predictions_df
    if df.empty:
        return np.array([]), np.array([])
    sub = df[df["class"] == cls].copy()
    if sub.empty:
        return np.array([]), np.array([])
    sub = sub.sort_values("confidence", ascending=False)
    tps = (sub["status"] == "tp").astype(int).values
    fps = (sub["status"] == "fp").astype(int).values
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    gt_df = result.ground_truth_df
    n_gt = int((gt_df["class"] == cls).sum()) if not gt_df.empty else 0
    recall = cum_tp / max(n_gt, 1)
    return precision, recall


def plot_class_distribution(class_counts: Dict[str, int], out_path: Path) -> Path:
    """Bar chart of class counts in the dataset."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_counts) * 0.9), 4))
    if not class_counts:
        ax.text(0.5, 0.5, "No annotations", ha="center", va="center")
    else:
        items = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
        classes, counts = zip(*items)
        sns.barplot(x=list(classes), y=list(counts), ax=ax, color="#54a24b")
        ax.set_title("GT class distribution")
        ax.set_ylabel("instances")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confidence_histogram(result: EvaluationResult, out_path: Path) -> Path:
    """Confidence histograms, separated by TP vs FP."""
    fig, ax = plt.subplots(figsize=(7, 4))
    df = result.predictions_df
    if df.empty:
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
    else:
        for status, color in [("tp", "#4c78a8"), ("fp", "#e45756")]:
            vals = df[df["status"] == status]["confidence"]
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.6, label=status.upper(),
                        color=color)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence distribution: TP vs FP")
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Drawing predictions / GT on an image (used by failure case dumping)
# --------------------------------------------------------------------------- #
def draw_boxes_on_image(
    image_path: Path,
    gt_boxes: Sequence[Dict[str, Any]],
    pred_boxes: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Path:
    """Render GT (green) and predictions (red for FP, blue for TP) on the image.

    Uses OpenCV if available; silently no-op if image can't be opened.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed; skipping image render.")
        return out_path

    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not read image %s", image_path)
        return out_path

    for gt in gt_boxes:
        x, y, w, h = [int(v) for v in gt["bbox"]]
        color = (0, 180, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, gt["class"], (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for pred in pred_boxes:
        x, y, w, h = [int(v) for v in pred["bbox"]]
        color = (255, 0, 0) if pred.get("status") == "tp" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{pred['class']} {pred.get('confidence', 0):.2f}"
        cv2.putText(img, label, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


# --------------------------------------------------------------------------- #
# Plotly versions (for the Streamlit dashboard)
# --------------------------------------------------------------------------- #
def plotly_class_metrics(by_class: pd.DataFrame):
    import plotly.express as px
    if by_class.empty:
        return None
    df_long = by_class.melt(
        id_vars=["class"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(df_long, x="class", y="value", color="metric", barmode="group",
                 title="Per-class metrics", range_y=[0, 1.05])
    return fig


def plotly_condition_metrics(by_condition: Dict[str, pd.DataFrame], cond: str):
    import plotly.express as px
    df = by_condition.get(cond, pd.DataFrame())
    if df.empty:
        return None
    fig = px.bar(df, x=cond, y="f1", color="f1",
                 title=f"F1 by {cond}", range_y=[0, 1.05],
                 color_continuous_scale="RdYlGn")
    return fig


def plotly_confusion_matrix(cm: pd.DataFrame):
    import plotly.express as px
    if cm.empty:
        return None
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    aspect="auto", title="Confusion matrix")
    fig.update_xaxes(title="Prediction")
    fig.update_yaxes(title="Ground truth")
    return fig


def plotly_confidence_hist(predictions_df: pd.DataFrame):
    import plotly.express as px
    if predictions_df.empty:
        return None
    fig = px.histogram(predictions_df, x="confidence", color="status",
                       nbins=20, barmode="overlay", opacity=0.6,
                       title="Confidence distribution: TP vs FP")
    return fig
