"""
Main evaluation pipeline.

`Evaluator.run()` returns an EvaluationResult that contains:
  * per-image, per-prediction records (status = tp/fp + IoU + confidence)
  * per-image lists of false negatives
  * aggregated metrics overall, per-class, and per-condition
  * mAP-like AP@[0.5:0.95] for each class
  * a confusion matrix at the configured IoU threshold

Designed to feed both the report generator and the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import metrics as M
from .data_loader import all_image_ids, get_metadata_for

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    """All evaluation artefacts in a single in-memory object."""

    # Per-prediction long-format DataFrame.
    # Columns: image_id, class, bbox, confidence, status, iou,
    #          lighting, weather, occlusion, size_bucket
    predictions_df: pd.DataFrame

    # Per-GT-box long-format DataFrame for false negatives + matched GT.
    # Columns: image_id, class, bbox, occluded, matched (bool),
    #          lighting, weather, occlusion, size_bucket
    ground_truth_df: pd.DataFrame

    # Aggregated metrics.
    overall: Dict[str, float] = field(default_factory=dict)
    by_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_condition: Dict[str, pd.DataFrame] = field(default_factory=dict)
    by_size: pd.DataFrame = field(default_factory=pd.DataFrame)

    # AP per class (averaged across config IoU thresholds) and at IoU=0.5.
    ap_by_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mAP: float = 0.0
    mAP50: float = 0.0

    # Confusion matrix at the main IoU threshold. Rows = GT, cols = predictions.
    # Last row/col are "background" (unmatched).
    confusion_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Echo back useful pieces of config for downstream consumers.
    classes: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class Evaluator:
    """Runs evaluation over a dataset given GT, predictions, and metadata."""

    def __init__(
        self,
        ground_truth: Dict[str, List[Dict[str, Any]]],
        predictions: Dict[str, List[Dict[str, Any]]],
        metadata: Dict[str, Dict[str, str]],
        config: Dict[str, Any],
    ):
        self.gt = ground_truth
        self.preds = predictions
        self.meta = metadata
        self.config = config

        eval_cfg = config.get("evaluation", {})
        self.iou_threshold: float = eval_cfg.get("iou_threshold", 0.5)
        self.conf_threshold: float = eval_cfg.get("confidence_threshold", 0.25)
        self.map_thresholds: List[float] = eval_cfg.get(
            "map_iou_thresholds", [0.5, 0.75, 0.95]
        )

        size_cfg = config.get("object_sizes", {})
        self.small_max: float = size_cfg.get("small_max", M.SMALL_MAX_DEFAULT)
        self.medium_max: float = size_cfg.get("medium_max", M.MEDIUM_MAX_DEFAULT)

        # Sorted, deterministic class universe across GT + predictions.
        cls_set = {b["class"] for boxes in self.gt.values() for b in boxes}
        cls_set |= {b["class"] for boxes in self.preds.values() for b in boxes}
        self.classes: List[str] = sorted(cls_set)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self) -> EvaluationResult:
        """Run the full pipeline. Returns an EvaluationResult."""
        logger.info(
            "Running evaluation: %d images, %d classes, IoU=%.2f, conf>=%.2f",
            len(all_image_ids(self.gt, self.preds)),
            len(self.classes),
            self.iou_threshold,
            self.conf_threshold,
        )

        pred_df, gt_df = self._match_all_images()

        overall = self._aggregate_overall(pred_df, gt_df)
        by_class = self._aggregate_by_class(pred_df, gt_df)
        by_condition = self._aggregate_by_condition(pred_df, gt_df)
        by_size = self._aggregate_by_size(pred_df, gt_df)
        ap_by_class, mAP, mAP50 = self._compute_map(pred_df, gt_df)
        cm = self._confusion_matrix()

        return EvaluationResult(
            predictions_df=pred_df,
            ground_truth_df=gt_df,
            overall=overall,
            by_class=by_class,
            by_condition=by_condition,
            by_size=by_size,
            ap_by_class=ap_by_class,
            mAP=mAP,
            mAP50=mAP50,
            confusion_matrix=cm,
            classes=self.classes,
            config=self.config,
        )

    # --------------------------------------------------------------------- #
    # Step 1: per-image matching at the primary IoU threshold
    # --------------------------------------------------------------------- #
    def _match_all_images(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pred_rows: List[Dict[str, Any]] = []
        gt_rows: List[Dict[str, Any]] = []

        for image_id in all_image_ids(self.gt, self.preds):
            preds = self.preds.get(image_id, [])
            gts = self.gt.get(image_id, [])
            tags = get_metadata_for(self.meta, image_id)

            pred_records, unmatched_gt = M.match_predictions_to_gt(
                preds,
                gts,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.conf_threshold,
            )

            # Track which GT got matched for the gt_df.
            matched_gt_set = {
                r["matched_gt_index"] for r in pred_records if r["status"] == "tp"
            }

            for r in pred_records:
                pred_rows.append(
                    {
                        "image_id": image_id,
                        "class": r["class"],
                        "bbox": r["bbox"],
                        "confidence": r["confidence"],
                        "status": r["status"],
                        "iou": r["iou"],
                        "matched_gt_index": r["matched_gt_index"],
                        "lighting": tags["lighting"],
                        "weather": tags["weather"],
                        "occlusion": tags["occlusion"],
                        "size_bucket": M.size_bucket(
                            r["bbox"], self.small_max, self.medium_max
                        ),
                    }
                )

            for idx, gt in enumerate(gts):
                gt_rows.append(
                    {
                        "image_id": image_id,
                        "class": gt["class"],
                        "bbox": gt["bbox"],
                        "occluded": gt.get("occluded", False),
                        "matched": idx in matched_gt_set,
                        "lighting": tags["lighting"],
                        "weather": tags["weather"],
                        "occlusion": tags["occlusion"],
                        "size_bucket": M.size_bucket(
                            gt["bbox"], self.small_max, self.medium_max
                        ),
                    }
                )

        pred_df = pd.DataFrame(pred_rows)
        gt_df = pd.DataFrame(gt_rows)

        # Ensure schema even if everything is empty.
        if pred_df.empty:
            pred_df = pd.DataFrame(
                columns=[
                    "image_id", "class", "bbox", "confidence", "status", "iou",
                    "matched_gt_index", "lighting", "weather", "occlusion",
                    "size_bucket",
                ]
            )
        if gt_df.empty:
            gt_df = pd.DataFrame(
                columns=[
                    "image_id", "class", "bbox", "occluded", "matched",
                    "lighting", "weather", "occlusion", "size_bucket",
                ]
            )
        return pred_df, gt_df

    # --------------------------------------------------------------------- #
    # Step 2: aggregations
    # --------------------------------------------------------------------- #
    @staticmethod
    def _counts(pred_subset: pd.DataFrame, gt_subset: pd.DataFrame) -> Dict[str, int]:
        tp = int((pred_subset["status"] == "tp").sum()) if not pred_subset.empty else 0
        fp = int((pred_subset["status"] == "fp").sum()) if not pred_subset.empty else 0
        fn = int((~gt_subset["matched"]).sum()) if not gt_subset.empty else 0
        return {"tp": tp, "fp": fp, "fn": fn}

    def _aggregate_overall(
        self, pred_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> Dict[str, float]:
        c = self._counts(pred_df, gt_df)
        m = M.precision_recall_f1(c["tp"], c["fp"], c["fn"])
        m["num_images"] = len(all_image_ids(self.gt, self.preds))
        m["num_gt_boxes"] = int(len(gt_df))
        m["num_predictions"] = int(len(pred_df))
        return m

    def _aggregate_by_class(
        self, pred_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []
        for cls in self.classes:
            p_sub = pred_df[pred_df["class"] == cls]
            g_sub = gt_df[gt_df["class"] == cls]
            c = self._counts(p_sub, g_sub)
            m = M.precision_recall_f1(c["tp"], c["fp"], c["fn"])
            m["class"] = cls
            m["support"] = int(len(g_sub))
            rows.append(m)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[["class", "support", "tp", "fp", "fn",
                     "precision", "recall", "f1"]]
        return df

    def _aggregate_by_condition(
        self, pred_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for cond in ["lighting", "weather", "occlusion"]:
            values = sorted(
                set(pred_df[cond].dropna().tolist())
                | set(gt_df[cond].dropna().tolist())
            )
            rows = []
            for v in values:
                p_sub = pred_df[pred_df[cond] == v]
                g_sub = gt_df[gt_df[cond] == v]
                c = self._counts(p_sub, g_sub)
                m = M.precision_recall_f1(c["tp"], c["fp"], c["fn"])
                m[cond] = v
                m["support"] = int(len(g_sub))
                rows.append(m)
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df[[cond, "support", "tp", "fp", "fn",
                         "precision", "recall", "f1"]]
            out[cond] = df
        return out

    def _aggregate_by_size(
        self, pred_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []
        for bucket in ["small", "medium", "large"]:
            p_sub = pred_df[pred_df["size_bucket"] == bucket]
            g_sub = gt_df[gt_df["size_bucket"] == bucket]
            c = self._counts(p_sub, g_sub)
            m = M.precision_recall_f1(c["tp"], c["fp"], c["fn"])
            m["size_bucket"] = bucket
            m["support"] = int(len(g_sub))
            rows.append(m)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[["size_bucket", "support", "tp", "fp", "fn",
                     "precision", "recall", "f1"]]
        return df

    # --------------------------------------------------------------------- #
    # Step 3: AP / mAP across IoU thresholds, per class
    # --------------------------------------------------------------------- #
    def _compute_map(
        self, pred_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, float]], float, float]:
        ap_by_class: Dict[str, Dict[str, float]] = {}
        per_class_means: List[float] = []
        per_class_50: List[float] = []

        for cls in self.classes:
            ap_at_iou: Dict[float, float] = {}
            n_gt_total = int(((gt_df["class"] == cls)).sum()) if not gt_df.empty else 0

            for iou_t in self.map_thresholds:
                # Rematch only for this class at this IoU threshold.
                pred_by_img: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for image_id in all_image_ids(self.gt, self.preds):
                    preds_img = [
                        p for p in self.preds.get(image_id, [])
                        if p["class"] == cls
                        and p["confidence"] >= self.conf_threshold
                    ]
                    gts_img = [g for g in self.gt.get(image_id, []) if g["class"] == cls]
                    if not preds_img:
                        continue
                    records, _ = M.match_predictions_to_gt(
                        preds_img,
                        gts_img,
                        iou_threshold=iou_t,
                        confidence_threshold=self.conf_threshold,
                    )
                    pred_by_img[image_id] = records

                prec, rec, _ = M.precision_recall_curve(pred_by_img, n_gt_total)
                ap = M.average_precision(prec, rec)
                ap_at_iou[float(iou_t)] = ap

            mean_ap = float(np.mean(list(ap_at_iou.values()))) if ap_at_iou else 0.0
            ap50 = float(ap_at_iou.get(0.5, mean_ap))
            ap_by_class[cls] = {
                "ap_mean": mean_ap,
                "ap50": ap50,
                "by_iou": ap_at_iou,
            }
            per_class_means.append(mean_ap)
            per_class_50.append(ap50)

        mAP = float(np.mean(per_class_means)) if per_class_means else 0.0
        mAP50 = float(np.mean(per_class_50)) if per_class_50 else 0.0
        return ap_by_class, mAP, mAP50

    # --------------------------------------------------------------------- #
    # Step 4: confusion matrix
    # --------------------------------------------------------------------- #
    def _confusion_matrix(self) -> pd.DataFrame:
        """Class-confusion at the primary IoU threshold.

        For each prediction, we look for the GT (of any class) with highest
        IoU >= iou_threshold and record (gt_class, pred_class). Unmatched
        predictions go to (background -> pred_class); unmatched GTs go to
        (gt_class -> background).
        """
        labels = self.classes + ["background"]
        idx = {c: i for i, c in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

        for image_id in all_image_ids(self.gt, self.preds):
            preds = [
                p for p in self.preds.get(image_id, [])
                if p["confidence"] >= self.conf_threshold
            ]
            gts = self.gt.get(image_id, [])
            gt_used = [False] * len(gts)

            # Sort by descending confidence to make matching stable.
            preds_sorted = sorted(preds, key=lambda p: p["confidence"], reverse=True)

            for pred in preds_sorted:
                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(gts):
                    if gt_used[j]:
                        continue
                    iou = M.iou_xywh(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_j >= 0 and best_iou >= self.iou_threshold:
                    gt_used[best_j] = True
                    cm[idx[gts[best_j]["class"]], idx[pred["class"]]] += 1
                else:
                    # False positive against background.
                    cm[idx["background"], idx[pred["class"]]] += 1

            for j, used in enumerate(gt_used):
                if not used:
                    cm[idx[gts[j]["class"]], idx["background"]] += 1

        return pd.DataFrame(cm, index=labels, columns=labels)
