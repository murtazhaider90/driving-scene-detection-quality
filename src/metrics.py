"""
Metric primitives: IoU, greedy matching, precision/recall, AP/mAP.

Keeps a clean separation from evaluator.py:
* this module is pure functions over numpy arrays / simple Python types,
* evaluator.py glues these together across many images.

Box convention everywhere: [x, y, width, height], top-left origin.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# COCO-style object size buckets, in pixels^2.
# Default thresholds match the COCO definition: small <32^2, medium <96^2.
SMALL_MAX_DEFAULT = 32 * 32
MEDIUM_MAX_DEFAULT = 96 * 96


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #
def box_area(box: Sequence[float]) -> float:
    """Area of a single [x, y, w, h] box."""
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def iou_xywh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def iou_matrix(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return an (N_pred, N_gt) matrix of IoU values."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    mat = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            mat[i, j] = iou_xywh(pb, gb)
    return mat


# --------------------------------------------------------------------------- #
# Matching (per image, per class)
# --------------------------------------------------------------------------- #
def match_predictions_to_gt(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Match predictions to GT for one image using greedy IoU matching.

    Greedy by descending confidence, restricted to same class.

    Returns:
      pred_records: list aligned with input predictions (after the confidence
        filter), each entry = {class, bbox, confidence, status, iou,
        matched_gt_index}. status is "tp" or "fp".
      unmatched_gt_indices: indices into `ground_truth` that no prediction
        claimed (i.e. false negatives).
    """
    filtered_preds = [
        (i, p) for i, p in enumerate(predictions) if p["confidence"] >= confidence_threshold
    ]
    filtered_preds.sort(key=lambda kv: kv[1]["confidence"], reverse=True)

    gt_used = [False] * len(ground_truth)
    pred_records: List[Dict[str, Any]] = []

    for _, pred in filtered_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(ground_truth):
            if gt_used[j]:
                continue
            if gt["class"] != pred["class"]:
                continue
            iou = iou_xywh(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_used[best_gt_idx] = True
            pred_records.append(
                {
                    **pred,
                    "status": "tp",
                    "iou": best_iou,
                    "matched_gt_index": best_gt_idx,
                }
            )
        else:
            pred_records.append(
                {
                    **pred,
                    "status": "fp",
                    "iou": best_iou,
                    "matched_gt_index": -1,
                }
            )

    unmatched_gt = [j for j, used in enumerate(gt_used) if not used]
    return pred_records, unmatched_gt


# --------------------------------------------------------------------------- #
# Aggregated metrics
# --------------------------------------------------------------------------- #
def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, F1 from counts. Robust to zeros."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def precision_recall_curve(
    pred_records_by_image: Dict[str, List[Dict[str, Any]]],
    n_gt_total: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a PR curve by sweeping confidence over all predictions of one class.

    pred_records_by_image: {image_id: [{confidence, status: "tp"|"fp"}, ...]}
                           (already filtered to one class)
    n_gt_total: total ground-truth instances of that class.

    Returns (precision, recall, thresholds).
    """
    flat: List[Tuple[float, int]] = []
    for records in pred_records_by_image.values():
        for r in records:
            flat.append((r["confidence"], 1 if r["status"] == "tp" else 0))

    if not flat or n_gt_total == 0:
        return np.array([1.0]), np.array([0.0]), np.array([1.0])

    flat.sort(key=lambda x: x[0], reverse=True)
    confs = np.array([c for c, _ in flat], dtype=np.float64)
    tps = np.array([t for _, t in flat], dtype=np.float64)
    fps = 1.0 - tps

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / max(n_gt_total, 1)
    return precision, recall, confs


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """11-point interpolated AP (the original PASCAL VOC definition).

    Simple, stable, and good enough for relative model comparisons. For
    headline numbers on big datasets, switch to pycocotools.
    """
    if precision.size == 0 or recall.size == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


# --------------------------------------------------------------------------- #
# Size buckets (COCO convention)
# --------------------------------------------------------------------------- #
def size_bucket(
    box: Sequence[float],
    small_max: float = SMALL_MAX_DEFAULT,
    medium_max: float = MEDIUM_MAX_DEFAULT,
) -> str:
    """Return 'small', 'medium', or 'large' for a [x, y, w, h] box."""
    area = box_area(box)
    if area < small_max:
        return "small"
    if area < medium_max:
        return "medium"
    return "large"
