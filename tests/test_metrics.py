"""
Tests for the metric primitives and the end-to-end evaluator.
Run with: pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable when pytest is run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import metrics as M
from src.data_loader import load_config
from src.evaluator import Evaluator


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #
class TestIoU:
    def test_identical_boxes_give_one(self):
        a = [10, 10, 20, 20]
        assert M.iou_xywh(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes_give_zero(self):
        a = [0, 0, 10, 10]
        b = [100, 100, 10, 10]
        assert M.iou_xywh(a, b) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes, second is shifted 5 right - intersection 5x10=50,
        # union 10*10 + 10*10 - 50 = 150 -> 1/3.
        a = [0, 0, 10, 10]
        b = [5, 0, 10, 10]
        assert M.iou_xywh(a, b) == pytest.approx(1.0 / 3.0)

    def test_zero_area_box(self):
        a = [0, 0, 0, 0]
        b = [0, 0, 10, 10]
        assert M.iou_xywh(a, b) == 0.0

    def test_negative_size_handled(self):
        # Defensive: negative widths shouldn't blow up.
        a = [0, 0, -5, -5]
        b = [0, 0, 10, 10]
        assert M.iou_xywh(a, b) == 0.0


# --------------------------------------------------------------------------- #
# precision_recall_f1
# --------------------------------------------------------------------------- #
class TestPRF:
    def test_perfect_score(self):
        m = M.precision_recall_f1(10, 0, 0)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_no_predictions(self):
        m = M.precision_recall_f1(0, 0, 5)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_typical_case(self):
        m = M.precision_recall_f1(tp=8, fp=2, fn=4)
        assert m["precision"] == pytest.approx(0.8)
        assert m["recall"] == pytest.approx(8 / 12)
        # F1 = 2 * (0.8 * 8/12) / (0.8 + 8/12)
        expected = 2 * 0.8 * (8 / 12) / (0.8 + 8 / 12)
        assert m["f1"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# match_predictions_to_gt
# --------------------------------------------------------------------------- #
class TestMatching:
    def test_one_perfect_match(self):
        gt = [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}]
        preds = [{"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9}]
        records, unmatched = M.match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        assert len(records) == 1 and records[0]["status"] == "tp"
        assert unmatched == []

    def test_class_mismatch_is_fp(self):
        gt = [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}]
        preds = [{"class": "truck", "bbox": [0, 0, 10, 10], "confidence": 0.9}]
        records, unmatched = M.match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        assert records[0]["status"] == "fp"
        assert unmatched == [0]

    def test_below_iou_threshold_is_fp(self):
        gt = [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}]
        preds = [{"class": "car", "bbox": [8, 0, 10, 10], "confidence": 0.9}]
        records, _ = M.match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        # IoU = 2*10/(100+100-20) = 20/180 ~= 0.11 -> below threshold.
        assert records[0]["status"] == "fp"

    def test_confidence_filter(self):
        gt = [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}]
        preds = [{"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.1}]
        records, unmatched = M.match_predictions_to_gt(
            preds, gt, iou_threshold=0.5, confidence_threshold=0.25
        )
        assert records == []
        assert unmatched == [0]  # GT remains unmatched -> FN

    def test_greedy_prefers_higher_confidence(self):
        gt = [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}]
        preds = [
            {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.5},
            {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        ]
        records, _ = M.match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        # Should be one TP (the conf=0.9 one) and one FP.
        statuses = [(r["confidence"], r["status"]) for r in records]
        tp_conf = [c for c, s in statuses if s == "tp"]
        assert tp_conf == [0.9]


# --------------------------------------------------------------------------- #
# AP
# --------------------------------------------------------------------------- #
class TestAveragePrecision:
    def test_perfect_curve_is_one(self):
        precision = np.ones(10)
        recall = np.linspace(0.1, 1.0, 10)
        assert M.average_precision(precision, recall) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        assert M.average_precision(np.array([]), np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# Size buckets
# --------------------------------------------------------------------------- #
class TestSizeBucket:
    def test_small(self):
        assert M.size_bucket([0, 0, 10, 10]) == "small"

    def test_medium(self):
        assert M.size_bucket([0, 0, 50, 50]) == "medium"

    def test_large(self):
        assert M.size_bucket([0, 0, 200, 200]) == "large"


# --------------------------------------------------------------------------- #
# Full evaluator on a tiny in-memory dataset
# --------------------------------------------------------------------------- #
class TestEvaluator:
    @pytest.fixture
    def config(self):
        return load_config(ROOT / "config" / "default.yaml")

    def test_perfect_dataset(self, config):
        gt = {
            "img1.jpg": [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}],
            "img2.jpg": [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}],
        }
        preds = {
            "img1.jpg": [{"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9}],
            "img2.jpg": [{"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9}],
        }
        result = Evaluator(gt, preds, {}, config).run()
        assert result.overall["precision"] == 1.0
        assert result.overall["recall"] == 1.0
        assert result.overall["f1"] == 1.0
        assert result.overall["tp"] == 2

    def test_all_misses(self, config):
        gt = {
            "img1.jpg": [{"class": "car", "bbox": [0, 0, 10, 10], "occluded": False}],
        }
        result = Evaluator(gt, {"img1.jpg": []}, {}, config).run()
        assert result.overall["fn"] == 1
        assert result.overall["tp"] == 0
        assert result.overall["recall"] == 0.0

    def test_class_wise_metrics(self, config):
        gt = {
            "img1.jpg": [
                {"class": "car", "bbox": [0, 0, 10, 10], "occluded": False},
                {"class": "pedestrian", "bbox": [20, 0, 10, 10], "occluded": False},
            ],
        }
        preds = {
            "img1.jpg": [
                # Car correctly predicted.
                {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9},
                # Pedestrian missed (no prediction).
                # Extra false positive of class truck.
                {"class": "truck", "bbox": [50, 50, 10, 10], "confidence": 0.8},
            ]
        }
        result = Evaluator(gt, preds, {}, config).run()
        by_class = result.by_class.set_index("class")
        assert by_class.loc["car", "tp"] == 1
        assert by_class.loc["car", "fp"] == 0
        assert by_class.loc["pedestrian", "fn"] == 1
        assert by_class.loc["truck", "fp"] == 1
