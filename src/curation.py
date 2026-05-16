"""
Dataset curation tools.

These functions inspect the ground-truth + metadata side of the dataset
(independent of predictions) to surface dataset-level issues:

  * class imbalance / underrepresented classes
  * condition imbalance / underrepresented conditions
  * images with no annotations
  * suspicious annotations (tiny boxes, extreme aspect ratios)
  * near-duplicate images (dHash)

The duplicate detector is optional - it only runs if you pass an image
directory and Pillow is installed.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Class / condition balance
# --------------------------------------------------------------------------- #
def class_balance_report(
    ground_truth: Dict[str, List[Dict[str, Any]]],
    min_instances: int = 50,
) -> Dict[str, Any]:
    """Per-class counts + which classes are underrepresented."""
    counts: Dict[str, int] = {}
    for boxes in ground_truth.values():
        for b in boxes:
            counts[b["class"]] = counts.get(b["class"], 0) + 1

    total = sum(counts.values())
    rows = [
        {
            "class": cls,
            "instances": n,
            "share": (n / total) if total > 0 else 0.0,
            "underrepresented": n < min_instances,
        }
        for cls, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "total_instances": total,
        "by_class": rows,
        "underrepresented_classes": [r["class"] for r in rows if r["underrepresented"]],
    }


def condition_balance_report(
    metadata: Dict[str, Dict[str, str]],
    min_share: float = 0.1,
) -> Dict[str, Any]:
    """For each condition (lighting/weather/occlusion), report share + flags."""
    out: Dict[str, Any] = {}
    if not metadata:
        return {"lighting": {}, "weather": {}, "occlusion": {}}

    for cond in ["lighting", "weather", "occlusion"]:
        counter: Counter = Counter()
        for tags in metadata.values():
            counter[tags.get(cond, "unknown")] += 1
        total = sum(counter.values())
        rows = []
        for value, n in counter.most_common():
            share = (n / total) if total > 0 else 0.0
            rows.append({
                "value": value,
                "count": n,
                "share": share,
                "underrepresented": share < min_share,
            })
        out[cond] = {
            "total_images": total,
            "values": rows,
            "underrepresented": [r["value"] for r in rows if r["underrepresented"]],
        }
    return out


# --------------------------------------------------------------------------- #
# Annotation quality
# --------------------------------------------------------------------------- #
def empty_annotation_images(
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Images present in the GT file with no boxes."""
    return sorted([img_id for img_id, boxes in ground_truth.items() if not boxes])


def suspicious_annotations(
    ground_truth: Dict[str, List[Dict[str, Any]]],
    min_area: float = 4.0,
    max_aspect_ratio: float = 25.0,
) -> List[Dict[str, Any]]:
    """Flag degenerate boxes: zero-area, extreme aspect ratio, negative dims."""
    suspicious: List[Dict[str, Any]] = []
    for img_id, boxes in ground_truth.items():
        for idx, box in enumerate(boxes):
            x, y, w, h = box["bbox"]
            issues = []
            if w <= 0 or h <= 0:
                issues.append("non_positive_size")
            else:
                if w * h < min_area:
                    issues.append("tiny_box")
                ratio = max(w / h, h / w)
                if ratio > max_aspect_ratio:
                    issues.append("extreme_aspect_ratio")
            if issues:
                suspicious.append({
                    "image_id": img_id,
                    "box_index": idx,
                    "class": box["class"],
                    "bbox": box["bbox"],
                    "issues": issues,
                })
    return suspicious


# --------------------------------------------------------------------------- #
# Near-duplicate detection (optional, uses Pillow if available)
# --------------------------------------------------------------------------- #
def _dhash(image_path: Path, hash_size: int = 8) -> Optional[int]:
    """64-bit difference hash. Returns None if image can't be opened."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed - skipping duplicate detection.")
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.LANCZOS
            )
            pixels = list(img.getdata())
        diff = 0
        bit = 0
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                if left > right:
                    diff |= 1 << bit
                bit += 1
        return diff
    except Exception as exc:
        logger.warning("Could not hash %s: %s", image_path, exc)
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(
    image_dir: str | Path,
    image_ids: List[str],
    threshold: int = 5,
) -> List[Dict[str, Any]]:
    """Group images whose dHash hamming distance is <= threshold.

    Returns a list of dicts: {hash, members: [image_id, ...]}.
    Only returns groups with 2+ members.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        logger.warning("Image dir not found: %s - skipping duplicates.", image_dir)
        return []

    hashes: Dict[str, int] = {}
    for img_id in image_ids:
        path = image_dir / img_id
        if not path.exists():
            continue
        h = _dhash(path)
        if h is not None:
            hashes[img_id] = h

    if not hashes:
        return []

    # Group by approximate equality via union-find lite.
    ids = list(hashes.keys())
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if _hamming(hashes[a], hashes[b]) <= threshold:
                union(a, b)

    groups: Dict[str, List[str]] = {}
    for img_id in ids:
        root = find(img_id)
        groups.setdefault(root, []).append(img_id)

    return [
        {"representative": root, "members": sorted(members)}
        for root, members in groups.items()
        if len(members) > 1
    ]


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
def run_curation(
    ground_truth: Dict[str, List[Dict[str, Any]]],
    metadata: Dict[str, Dict[str, str]],
    config: Dict[str, Any],
    image_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run all curation checks and return one bundled report."""
    cur_cfg = config.get("curation", {})
    report: Dict[str, Any] = {
        "class_balance": class_balance_report(
            ground_truth, min_instances=cur_cfg.get("min_instances_per_class", 50)
        ),
        "condition_balance": condition_balance_report(
            metadata, min_share=cur_cfg.get("min_share_per_condition", 0.1)
        ),
        "empty_annotation_images": empty_annotation_images(ground_truth),
        "suspicious_annotations": suspicious_annotations(
            ground_truth,
            min_area=cur_cfg.get("min_box_area", 4.0),
            max_aspect_ratio=cur_cfg.get("max_aspect_ratio", 25.0),
        ),
        "near_duplicates": [],
    }
    if image_dir is not None:
        report["near_duplicates"] = find_near_duplicates(
            image_dir,
            list(ground_truth.keys()),
            threshold=cur_cfg.get("duplicate_hash_threshold", 5),
        )
    return report
