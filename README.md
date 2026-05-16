# Autonomous Driving ML Data Quality & Failure Analysis Platform

A Python pipeline that evaluates object detection models on driving-scene
imagery, mines failure cases across lighting / weather / occlusion / object
size, and produces dataset curation and model debugging insights. Comes with
a CLI, an HTML/Markdown reporter, and a Streamlit dashboard.

> **One-line pitch (CV-ready):**
> Built a Python pipeline to evaluate object detection models on driving-scene
> data and identify failure cases across conditions including lighting,
> weather, and occlusion. Developed tools for performance tracking and
> dataset curation, improving model debugging and retraining insights.

---

## Why data quality matters in autonomous driving

In autonomous driving, model failures are rarely uniform — they cluster around
the long tail: rain, fog, dusk, night, partial occlusion, small distant
pedestrians, unusual viewing angles. Aggregate metrics like mAP hide those
weaknesses and let unsafe regressions ship.

This tool slices evaluation by condition rather than reporting a single
number, surfaces the specific images and predictions that fail, and emits
curation recommendations that feed directly into the next round of labelling
and retraining. The same workflow is what production AV ML teams run
internally; this project rebuilds the core of it in a single-developer-friendly
shape.

---

## Features

**Evaluation pipeline (`src/evaluator.py`)**
- Greedy IoU matching between predictions and ground truth, per image / class
- Precision, recall, F1, TP/FP/FN counts overall and per class
- AP per class across configurable IoU thresholds; mAP and mAP@0.5
- Confusion matrix (incl. background class for FP / FN)
- COCO-format input adapter

**Condition-based failure analysis (`src/failure_analysis.py`)**
- F1 sliced by lighting, weather, occlusion, and COCO-style object size
- Per-image FN / FP counts; worst-N images for each
- High-confidence false positives (hard-negative mining candidates)
- Low-confidence true positives (calibration suspects)
- Missed safety-critical objects (pedestrians, cyclists, motorcycles, …)
- Worst-performing condition slices with auto-generated recommendations

**Dataset curation (`src/curation.py`)**
- Underrepresented class detection (configurable threshold)
- Underrepresented condition detection
- Suspicious-annotation flags (zero-area, extreme aspect ratio)
- Empty-annotation image detection
- Near-duplicate detection via dHash

**Reporting (`src/report_generator.py`)**
- Self-contained HTML report with embedded PNG charts
- Markdown report for PRs and docs
- Machine-readable artefacts: `metrics.json`, `failures.json`, `curation.json`,
  long-format `predictions_long.csv` / `ground_truth_long.csv`
- Per-failure-image thumbnails with GT + prediction boxes drawn on

**Streamlit dashboard (`dashboard/app.py`)**
- Live mode: run evaluation in-process and explore interactively
- Report mode: load a previous run from disk
- Per-class, per-condition, confusion-matrix, confidence-histogram, failure-mining,
  curation, and recommendations tabs

---

## Project structure

```
autonomous-driving-data-quality/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # IoU/confidence thresholds, size buckets, etc.
├── data/
│   └── sample/
│       ├── ground_truth.json
│       ├── predictions.json
│       ├── metadata.json
│       └── images/           # optional - drop your .jpg files here
├── src/
│   ├── data_loader.py        # JSON + COCO loaders, config
│   ├── metrics.py            # IoU, matching, precision/recall, AP
│   ├── evaluator.py          # main pipeline, aggregations, mAP, confusion matrix
│   ├── failure_analysis.py   # failure mining + auto recommendations
│   ├── curation.py           # class/condition balance, duplicates, suspicious GTs
│   ├── visualization.py      # matplotlib + plotly + cv2 box drawing
│   └── report_generator.py   # HTML + Markdown report rendering
├── scripts/
│   └── run_evaluation.py     # CLI entrypoint
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_metrics.py       # 21 unit tests + evaluator integration tests
├── notebooks/
│   └── exploratory_analysis.ipynb
└── reports/                  # generated outputs land here
```

---

## Installation

```bash
git clone <your-fork>
cd autonomous-driving-data-quality
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core pipeline has no GPU/PyTorch dependency. You only need `ultralytics`
or `torch` if you also want to run a real detector on your images to
produce a `predictions.json` (see "Generating predictions" below).

---

## Quickstart

Run the included demo dataset (10 frames, 3 conditions, 5 classes):

```bash
python scripts/run_evaluation.py \
  --ground-truth data/sample/ground_truth.json \
  --predictions  data/sample/predictions.json \
  --metadata     data/sample/metadata.json \
  --config       config/default.yaml \
  --output       reports/demo_report
```

You should see (numbers will match exactly on the bundled sample):

```
Overall: P=0.897 R=0.788 F1=0.839 mAP50=0.858
```

Outputs land in `reports/demo_report/`:

```
report.html              # self-contained HTML report (open in browser)
report.md                # markdown summary
metrics.json             # machine-readable metrics
failures.json            # all mined failure cases
curation.json            # curation findings
predictions_long.csv     # one row per prediction, with IoU + status + tags
ground_truth_long.csv    # one row per GT box, with matched flag
plots/*.png              # all charts
failures/*.jpg           # annotated failure thumbnails (if --image-dir given)
```

To launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard defaults to the sample dataset so you can poke around
immediately.

---

## Dataset format

This project uses a simple native JSON format. All bounding boxes are
`[x, y, width, height]` in pixels, top-left origin.

**`ground_truth.json`** — one entry per image:
```json
[
  {
    "image_id": "frame_001.jpg",
    "boxes": [
      {"class": "car", "bbox": [100, 120, 80, 60], "occluded": false}
    ]
  }
]
```

**`predictions.json`** — one entry per image, with confidence:
```json
[
  {
    "image_id": "frame_001.jpg",
    "boxes": [
      {"class": "car", "bbox": [102, 118, 78, 62], "confidence": 0.91}
    ]
  }
]
```

**`metadata.json`** — optional per-image scene tags:
```json
[
  {"image_id": "frame_001.jpg",
   "lighting": "day",
   "weather":  "clear",
   "occlusion": "low"}
]
```

If your dataset already ships COCO JSON, use the converter:

```bash
python scripts/run_evaluation.py \
  --coco-annotations  annotations/instances_val.json \
  --coco-results      detections/results.json \
  --metadata          metadata.json \
  --output            reports/coco_run
```

### Using real driving datasets

The pipeline is dataset-agnostic. Tested layouts and conversion sketches:

- **BDD100K** — convert the JSON to either native or COCO format and reuse
  the `weather` / `timeofday` / `scene` fields directly as metadata.
- **KITTI** — convert KITTI label files to native JSON
  (class name + `[x, y, w, h]`); use the truncation/occlusion column as
  `occluded` and the daytime/highway split as metadata.
- **nuImages** — export to COCO via the nuScenes devkit and use the
  scene-description tags as condition metadata.
- **COCO subset** — filter to the road-relevant categories (`person`,
  `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`,
  `stop sign`).

### Generating predictions

Any model that emits a `predictions.json` with the format above will work.
For Ultralytics YOLOv8 the conversion is a few lines:

```python
from ultralytics import YOLO
import json

model = YOLO("yolov8n.pt")
out = []
for img_path in image_paths:
    r = model(img_path, verbose=False)[0]
    boxes = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append({
            "class": model.names[int(b.cls)],
            "bbox":  [x1, y1, x2 - x1, y2 - y1],
            "confidence": float(b.conf),
        })
    out.append({"image_id": img_path.name, "boxes": boxes})

json.dump(out, open("predictions.json", "w"))
```

---

## Configuration

All thresholds live in `config/default.yaml` and can be overridden per run.
Highlights:

```yaml
evaluation:
  iou_threshold: 0.5            # primary matching threshold
  confidence_threshold: 0.25
  map_iou_thresholds: [0.5, ..., 0.95]   # COCO-style sweep for mAP
  critical_classes: ["pedestrian", "person", "bicycle", "cyclist", "motorcycle"]

failure_analysis:
  fn_threshold: 2               # images with >= 2 FNs are flagged
  high_confidence_wrong: 0.7    # FPs above this conf are mined

curation:
  min_instances_per_class: 50   # below = underrepresented
  min_share_per_condition: 0.10
```

---

## Example outputs

Running the bundled sample dataset produces:

- **Overall**: P=0.897, R=0.788, F1=0.839, mAP@0.5=0.858
- **Per-class**: `pedestrian` F1=0.50 (the weakest class — 6 missed)
- **By weather**: `fog` F1=0.50, `rain` F1=0.67, `clear` F1=0.90
- **By lighting**: `night` F1=0.67 vs `day` F1=0.89
- **By occlusion**: `high` F1=0.67 vs `low` F1=0.91
- **Mined failures**: 6 missed pedestrians, 3 high-confidence FPs

These translate directly into auto-generated recommendations:

> - Class 'pedestrian' has F1=0.50. Collect more training data and review
>   annotation quality.
> - Performance drops under weather='fog' (F1=0.50). Add more fog samples.
> - 6 safety-critical objects were missed. Prioritise for relabeling.
> - 3 confident false positives detected. Mine as hard negatives.

---

## Tests

```bash
pytest tests/ -v
```

21 unit tests covering IoU geometry, greedy matching, PR/F1 corner cases,
AP calculation, size bucketing, and three full-pipeline integration tests
against in-memory datasets.

---

## Engineering Highlights

- **Modular evaluation pipeline** — pure metric primitives in `metrics.py`
  are independently testable; the `Evaluator` glues them together and emits
  a single typed `EvaluationResult`. The reporter and dashboard depend only
  on that artefact.
- **Model-agnostic prediction format** — any detector that produces JSON
  with `image_id`, `class`, `bbox`, `confidence` works. A COCO adapter is
  included for the common case.
- **Condition-based failure analysis** — performance is sliced across
  lighting, weather, occlusion, and COCO object-size buckets, surfacing the
  long-tail slices that aggregate mAP hides.
- **Dataset curation insights** — class/condition imbalance, suspicious
  annotations, empty-annotation images, and dHash-based near-duplicates,
  all in one curation pass.
- **Automated reporting** — HTML + Markdown reports plus machine-readable
  `metrics.json` / `failures.json` / `curation.json` ready for CI pipelines.
- **Dashboard for debugging** — Streamlit app with a live mode (run the
  pipeline) and a report mode (inspect a previous run), Plotly charts,
  and per-tab drilldowns.
- **Tested core** — 21 pytest cases including integration tests against
  small synthetic datasets, catching matching/aggregation regressions before
  they reach the report.

---

## How this maps to real ML engineering work

| What this project does                          | What it mirrors in production AV / CV teams                       |
|-------------------------------------------------|-------------------------------------------------------------------|
| Sliced evaluation by lighting/weather/occlusion | Slice-based eval dashboards (Tesla, Waymo, Cruise blog posts)     |
| Mining high-conf FPs and missed criticals       | Active learning / hard-negative mining loops                      |
| Recommending dataset additions                  | Data engine feedback into labelling pipelines                     |
| Tracking metrics + artefacts per run            | Foundation for an MLflow / Weights & Biases experiment tracker    |
| Confusion matrix + per-class AP                 | The standard scorecard CI gates models on                         |
| Dataset curation pass                           | Label QA / annotation-vendor feedback workflow                    |

---

## Project limitations

- **AP implementation is the 11-point VOC variant**, not the full COCO area
  under PR curve. Fine for relative model comparisons; switch to
  `pycocotools` if you want headline numbers identical to public benchmarks.
- **The matcher is greedy by confidence**, not Hungarian. Greedy is
  standard for detection metrics and matches PASCAL VOC / pycocotools, but
  it's worth knowing for edge cases.
- **No tracking / temporal evaluation** — single-frame detection only.
  Multi-object tracking metrics (MOTA, IDF1) are out of scope.
- **Sample dataset is synthetic** — the 10 demo images use placeholder
  filenames. Plug in a real dataset to get real conclusions.
- **No GPU inference loop** — this project evaluates predictions you
  already have. Generation is a separate concern (see the YOLOv8 snippet).

---

## Future improvements

- Pluggable matchers (Hungarian, soft-NMS) behind the same interface
- pycocotools-backed mAP for headline-grade numbers
- Multi-run comparison view (regression detection across model versions)
- Embedding-based duplicate / near-duplicate detection (CLIP rather than dHash)
- Optional active-learning export: write the top-K hard examples to a
  CSV that the labelling pipeline can ingest
- CI integration template (GitHub Actions) that fails a PR if the F1 on
  a critical condition slice drops by more than a threshold

---

## License

MIT.
