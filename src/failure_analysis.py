"""
Failure-case mining.

Given an EvaluationResult, this module surfaces images and predictions that
are the most useful for debugging or retraining:

  * worst_images_by_fn       - images where the model missed lots of objects
  * worst_images_by_fp       - images where the model hallucinated lots of objects
  * low_confidence_correct   - TPs that barely cleared the confidence bar
  * high_confidence_wrong    - confident FPs (the most embarrassing failures)
  * missed_critical          - missed safety-critical objects (pedestrians etc.)
  * worst_conditions         - condition slices with the lowest F1
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from .evaluator import EvaluationResult

logger = logging.getLogger(__name__)


def _safety(value: Any) -> Any:
    """Make values JSON-safe (numpy -> python)."""
    if hasattr(value, "item"):
        return value.item()
    return value


# --------------------------------------------------------------------------- #
# Per-image failure counts
# --------------------------------------------------------------------------- #
def per_image_failure_counts(result: EvaluationResult) -> pd.DataFrame:
    """Return per-image TP/FP/FN counts, sorted by descending (FN + FP)."""
    pred_df = result.predictions_df
    gt_df = result.ground_truth_df

    image_ids = sorted(
        set(pred_df["image_id"].tolist()) | set(gt_df["image_id"].tolist())
    )
    rows = []
    for img_id in image_ids:
        p = pred_df[pred_df["image_id"] == img_id]
        g = gt_df[gt_df["image_id"] == img_id]
        tp = int((p["status"] == "tp").sum())
        fp = int((p["status"] == "fp").sum())
        fn = int((~g["matched"]).sum())

        if not g.empty:
            tags = g.iloc[0]
        elif not p.empty:
            tags = p.iloc[0]
        else:
            tags = pd.Series({"lighting": "unknown", "weather": "unknown",
                              "occlusion": "unknown"})

        rows.append({
            "image_id": img_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_errors": fp + fn,
            "lighting": tags.get("lighting", "unknown"),
            "weather": tags.get("weather", "unknown"),
            "occlusion": tags.get("occlusion", "unknown"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_errors", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #
class FailureMiner:
    """Mines interesting failure cases from an EvaluationResult."""

    def __init__(self, result: EvaluationResult, config: Dict[str, Any]):
        self.result = result
        fa = config.get("failure_analysis", {})
        self.fn_threshold: int = fa.get("fn_threshold", 2)
        self.fp_threshold: int = fa.get("fp_threshold", 2)
        self.low_conf_correct: float = fa.get("low_confidence_correct", 0.4)
        self.high_conf_wrong: float = fa.get("high_confidence_wrong", 0.7)
        self.top_k: int = fa.get("top_k_failures", 20)
        self.critical_classes: List[str] = config.get("evaluation", {}).get(
            "critical_classes",
            ["pedestrian", "person", "bicycle", "cyclist", "motorcycle"],
        )

    # ------------------------------------------------------------------- #
    def mine(self) -> Dict[str, Any]:
        """Return a dict of failure case categories."""
        counts = per_image_failure_counts(self.result)

        report: Dict[str, Any] = {
            "worst_images_by_fn": self._top_images(counts, "fn", self.fn_threshold),
            "worst_images_by_fp": self._top_images(counts, "fp", self.fp_threshold),
            "low_confidence_correct": self._low_confidence_correct(),
            "high_confidence_wrong": self._high_confidence_wrong(),
            "missed_critical": self._missed_critical(),
            "worst_conditions": self._worst_conditions(),
            "per_image_counts": counts.to_dict(orient="records"),
        }
        return report

    # ------------------------------------------------------------------- #
    def _top_images(
        self, counts: pd.DataFrame, col: str, threshold: int
    ) -> List[Dict[str, Any]]:
        if counts.empty:
            return []
        sub = counts[counts[col] >= threshold].sort_values(col, ascending=False)
        return sub.head(self.top_k).to_dict(orient="records")

    def _low_confidence_correct(self) -> List[Dict[str, Any]]:
        df = self.result.predictions_df
        if df.empty:
            return []
        sub = df[
            (df["status"] == "tp") & (df["confidence"] < self.low_conf_correct)
        ].sort_values("confidence", ascending=True)
        return sub.head(self.top_k).to_dict(orient="records")

    def _high_confidence_wrong(self) -> List[Dict[str, Any]]:
        df = self.result.predictions_df
        if df.empty:
            return []
        sub = df[
            (df["status"] == "fp") & (df["confidence"] >= self.high_conf_wrong)
        ].sort_values("confidence", ascending=False)
        return sub.head(self.top_k).to_dict(orient="records")

    def _missed_critical(self) -> List[Dict[str, Any]]:
        df = self.result.ground_truth_df
        if df.empty:
            return []
        sub = df[
            (~df["matched"])
            & (df["class"].isin(self.critical_classes))
        ]
        return sub.head(self.top_k).to_dict(orient="records")

    def _worst_conditions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Pick the worst-performing slice per condition (lowest F1, support>0)."""
        worst: Dict[str, List[Dict[str, Any]]] = {}
        for cond, df in self.result.by_condition.items():
            if df.empty:
                worst[cond] = []
                continue
            sub = df[df["support"] > 0].sort_values("f1", ascending=True)
            worst[cond] = sub.head(3).to_dict(orient="records")
        return worst


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
def generate_recommendations(
    result: EvaluationResult, mined: Dict[str, Any], config: Dict[str, Any]
) -> List[str]:
    """Produce a list of plain-English recommendations from the evaluation."""
    recs: List[str] = []

    # 1. Worst classes - only flag genuinely weak ones, not "worst of the top".
    by_class = result.by_class
    if not by_class.empty:
        weak = by_class[(by_class["support"] > 0) & (by_class["f1"] < 0.7)]
        worst_classes = weak.sort_values("f1").head(3)
        for _, row in worst_classes.iterrows():
            recs.append(
                f"Class '{row['class']}' has F1={row['f1']:.2f} "
                f"(support={int(row['support'])}). "
                "Collect more training data and review annotation quality."
            )

    # 2. Worst condition slices
    for cond, slices in mined.get("worst_conditions", {}).items():
        for s in slices:
            if s.get("support", 0) >= 3 and s.get("f1", 1.0) < 0.6:
                recs.append(
                    f"Performance drops under {cond}='{s[cond]}' (F1={s['f1']:.2f}). "
                    f"Add more {cond}='{s[cond]}' samples to the training set."
                )

    # 3. Missed safety-critical objects
    n_missed = len(mined.get("missed_critical", []))
    if n_missed > 0:
        recs.append(
            f"{n_missed} safety-critical objects were missed "
            "(pedestrians/cyclists). Prioritise these for relabeling and retraining."
        )

    # 4. High-confidence wrong predictions
    n_high_fp = len(mined.get("high_confidence_wrong", []))
    if n_high_fp > 0:
        recs.append(
            f"{n_high_fp} confident false positives detected. "
            "Mine these as hard negatives for the next training round."
        )

    # 5. Size weaknesses
    by_size = result.by_size
    if not by_size.empty:
        small = by_size[by_size["size_bucket"] == "small"]
        if not small.empty and small.iloc[0].get("support", 0) > 0:
            f1_small = float(small.iloc[0]["f1"])
            if f1_small < 0.5:
                recs.append(
                    f"Small-object F1={f1_small:.2f}. "
                    "Consider higher input resolution or multi-scale training."
                )

    if not recs:
        recs.append("No major weaknesses detected on this dataset slice. "
                    "Expand the evaluation set to stress-test edge cases.")
    return recs
