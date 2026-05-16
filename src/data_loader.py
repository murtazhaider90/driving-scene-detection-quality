"""
Data loading and normalization.

The pipeline consumes three artefacts:

* ground_truth.json  - per-image list of GT boxes (with optional occlusion flag)
* predictions.json   - per-image list of predicted boxes (with confidence)
* metadata.json      - per-image scene tags (lighting / weather / occlusion)

All bboxes use [x, y, width, height] (top-left origin, pixels).

This module also provides a converter from COCO-format detection results,
so the same downstream pipeline works on COCO subsets like KITTI/BDD/nuImages
exports that have been turned into COCO JSON.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> Dict[str, Any]:
    """Load the YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Native (project) JSON format
# --------------------------------------------------------------------------- #
def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return {image_id: [box, ...]} with normalized box dicts.

    Each box dict will have keys: class, bbox, occluded.
    """
    raw = _read_json(path)
    gt: Dict[str, List[Dict[str, Any]]] = {}
    for entry in raw:
        image_id = entry["image_id"]
        boxes = []
        for b in entry.get("boxes", []):
            boxes.append(
                {
                    "class": b["class"],
                    "bbox": [float(v) for v in b["bbox"]],
                    "occluded": bool(b.get("occluded", False)),
                }
            )
        gt[image_id] = boxes
    logger.info("Loaded ground truth for %d images", len(gt))
    return gt


def load_predictions(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return {image_id: [pred_box, ...]} with confidence."""
    raw = _read_json(path)
    preds: Dict[str, List[Dict[str, Any]]] = {}
    for entry in raw:
        image_id = entry["image_id"]
        boxes = []
        for b in entry.get("boxes", []):
            boxes.append(
                {
                    "class": b["class"],
                    "bbox": [float(v) for v in b["bbox"]],
                    "confidence": float(b.get("confidence", 1.0)),
                }
            )
        preds[image_id] = boxes
    logger.info("Loaded predictions for %d images", len(preds))
    return preds


def load_metadata(path: Optional[str | Path]) -> Dict[str, Dict[str, str]]:
    """Return {image_id: {lighting, weather, occlusion}}.

    If path is None or missing, returns {} (every image will be tagged "unknown").
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("Metadata file not found: %s (using 'unknown' tags)", p)
        return {}
    raw = _read_json(p)
    meta: Dict[str, Dict[str, str]] = {}
    for entry in raw:
        image_id = entry["image_id"]
        meta[image_id] = {
            "lighting": entry.get("lighting", "unknown"),
            "weather": entry.get("weather", "unknown"),
            "occlusion": entry.get("occlusion", "unknown"),
        }
    return meta


def get_metadata_for(
    metadata: Dict[str, Dict[str, str]], image_id: str
) -> Dict[str, str]:
    """Safe lookup that always returns the three tag keys."""
    return metadata.get(
        image_id,
        {"lighting": "unknown", "weather": "unknown", "occlusion": "unknown"},
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def all_image_ids(
    ground_truth: Dict[str, List[Dict[str, Any]]],
    predictions: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Union of image IDs across GT and predictions, sorted for determinism."""
    return sorted(set(ground_truth.keys()) | set(predictions.keys()))


def class_distribution(
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, int]:
    """Count GT instances per class."""
    counts: Dict[str, int] = defaultdict(int)
    for boxes in ground_truth.values():
        for b in boxes:
            counts[b["class"]] += 1
    return dict(counts)


# --------------------------------------------------------------------------- #
# COCO -> native conversion
# --------------------------------------------------------------------------- #
def coco_to_native(
    coco_annotations_path: str | Path,
    coco_results_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Convert COCO-format annotations (+ optional results) to native dicts.

    COCO format:
      - "images":       [{id, file_name, ...}]
      - "annotations":  [{image_id, category_id, bbox, iscrowd, ...}]
      - "categories":   [{id, name}]
      - results file:   [{image_id, category_id, bbox, score}]

    Returns (ground_truth_dict, predictions_dict). predictions_dict is empty
    if no results path is given.
    """
    coco = _read_json(coco_annotations_path)
    images = {img["id"]: img["file_name"] for img in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    gt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        file_name = images.get(ann["image_id"])
        if file_name is None:
            continue
        gt[file_name].append(
            {
                "class": categories.get(ann["category_id"], "unknown"),
                "bbox": [float(v) for v in ann["bbox"]],
                "occluded": bool(ann.get("iscrowd", 0)),
            }
        )

    preds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if coco_results_path is not None:
        results = _read_json(coco_results_path)
        for r in results:
            file_name = images.get(r["image_id"])
            if file_name is None:
                continue
            preds[file_name].append(
                {
                    "class": categories.get(r["category_id"], "unknown"),
                    "bbox": [float(v) for v in r["bbox"]],
                    "confidence": float(r.get("score", 1.0)),
                }
            )

    return dict(gt), dict(preds)
