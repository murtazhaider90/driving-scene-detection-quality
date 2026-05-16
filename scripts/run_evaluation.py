#!/usr/bin/env python3
"""
CLI entrypoint for the evaluation pipeline.

Example:

    python scripts/run_evaluation.py \\
        --ground-truth data/sample/ground_truth.json \\
        --predictions  data/sample/predictions.json \\
        --metadata     data/sample/metadata.json \\
        --config       config/default.yaml \\
        --output       reports/demo_report \\
        --image-dir    data/sample/images   # optional, enables failure thumbs

The script writes:
  reports/demo_report/report.html
  reports/demo_report/report.md
  reports/demo_report/metrics.json
  reports/demo_report/failures.json
  reports/demo_report/curation.json
  reports/demo_report/predictions_long.csv
  reports/demo_report/ground_truth_long.csv
  reports/demo_report/plots/*.png
  reports/demo_report/failures/*.jpg    (only if --image-dir is given)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when running this script directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    coco_to_native,
    load_config,
    load_ground_truth,
    load_metadata,
    load_predictions,
)
from src.evaluator import Evaluator  # noqa: E402
from src.failure_analysis import FailureMiner, generate_recommendations  # noqa: E402
from src.curation import run_curation  # noqa: E402
from src.report_generator import generate_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate object detection on driving-scene data."
    )
    p.add_argument("--ground-truth", required=False, default=None,
                   help="Path to native ground-truth JSON.")
    p.add_argument("--predictions", required=False, default=None,
                   help="Path to native predictions JSON.")
    p.add_argument("--metadata", required=False, default=None,
                   help="Path to per-image metadata JSON (optional).")
    p.add_argument("--coco-annotations", required=False, default=None,
                   help="Path to COCO-format annotations JSON (alternative input).")
    p.add_argument("--coco-results", required=False, default=None,
                   help="Path to COCO-format results JSON (alternative input).")
    p.add_argument("--config", required=False, default="config/default.yaml",
                   help="Path to YAML config.")
    p.add_argument("--output", required=True,
                   help="Output directory for the report.")
    p.add_argument("--image-dir", required=False, default=None,
                   help="Image directory; enables failure thumbnails + duplicates.")
    p.add_argument("--title", required=False,
                   default=None, help="Override the report title.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    log = logging.getLogger("run_evaluation")

    # 1. Config
    config = load_config(args.config)
    log.info("Loaded config from %s", args.config)

    # 2. Inputs - either native JSON or COCO. Native wins if both provided.
    if args.ground_truth and args.predictions:
        gt = load_ground_truth(args.ground_truth)
        preds = load_predictions(args.predictions)
    elif args.coco_annotations:
        gt, preds = coco_to_native(args.coco_annotations, args.coco_results)
        log.info("Loaded COCO format: %d images of GT, %d images of preds",
                 len(gt), len(preds))
    else:
        log.error("Must provide --ground-truth + --predictions, or "
                  "--coco-annotations.")
        return 1

    metadata = load_metadata(args.metadata)

    # 3. Evaluate
    evaluator = Evaluator(gt, preds, metadata, config)
    result = evaluator.run()
    log.info("Overall: P=%.3f R=%.3f F1=%.3f mAP50=%.3f",
             result.overall.get("precision", 0),
             result.overall.get("recall", 0),
             result.overall.get("f1", 0),
             result.mAP50)

    # 4. Failure mining + recommendations
    miner = FailureMiner(result, config)
    mined = miner.mine()
    recs = generate_recommendations(result, mined, config)

    # 5. Curation
    curation = run_curation(gt, metadata, config, image_dir=args.image_dir)

    # 6. Report
    title = args.title or config.get("report", {}).get(
        "title", "Autonomous Driving Model Evaluation Report"
    )
    paths = generate_report(
        result=result,
        mined=mined,
        curation=curation,
        recommendations=recs,
        ground_truth=gt,
        output_dir=args.output,
        title=title,
        image_dir=args.image_dir,
    )
    log.info("Report written to %s", paths["html"])
    print(f"\nReport: {paths['html']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
