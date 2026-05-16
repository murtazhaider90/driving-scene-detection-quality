"""
HTML + Markdown report generation.

Given an EvaluationResult, a mined failure dict, a curation dict, and a
recommendations list, this module:

  1. Renders all plots into PNGs under <output_dir>/plots/.
  2. Optionally draws bounding boxes on the worst failure images.
  3. Renders an HTML report (Jinja2) and a Markdown report.

The HTML report is intentionally self-contained: just open the file in a
browser. No JS, no external assets beyond the locally-saved PNGs.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Template

from .data_loader import class_distribution
from .evaluator import EvaluationResult
from . import visualization as V

logger = logging.getLogger(__name__)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
         margin: 2em auto; padding: 0 1.5em; color: #222; }
  h1, h2, h3 { color: #1f2937; }
  h1 { border-bottom: 2px solid #e5e7eb; padding-bottom: 0.3em; }
  h2 { margin-top: 2em; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr);
              gap: 1em; margin: 1em 0; }
  .kpi { background: #f3f4f6; border-radius: 8px; padding: 1em;
         text-align: center; }
  .kpi .value { font-size: 1.8em; font-weight: 600; color: #1d4ed8; }
  .kpi .label { font-size: 0.85em; color: #6b7280; margin-top: 0.3em; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0;
          font-size: 0.9em; }
  th, td { border: 1px solid #e5e7eb; padding: 0.45em 0.7em; text-align: left; }
  th { background: #f9fafb; }
  tr:nth-child(even) td { background: #fafafa; }
  img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px;
        margin: 0.5em 0; }
  .failure-grid { display: grid;
                  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                  gap: 1em; }
  .failure-card { background: #fafafa; padding: 0.7em; border-radius: 6px;
                  border: 1px solid #e5e7eb; }
  .failure-card .meta { font-size: 0.8em; color: #6b7280; }
  .rec { background: #fef3c7; border-left: 4px solid #f59e0b;
         padding: 0.6em 1em; margin: 0.5em 0; border-radius: 4px; }
  .small { font-size: 0.85em; color: #6b7280; }
  code { background: #f3f4f6; padding: 0.1em 0.4em; border-radius: 4px; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p class="small">Generated {{ generated_at }} &middot; IoU threshold {{ iou_threshold }}
&middot; confidence threshold {{ conf_threshold }}</p>

<h2>Summary</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="value">{{ overall.precision | round(3) }}</div>
                   <div class="label">Precision</div></div>
  <div class="kpi"><div class="value">{{ overall.recall | round(3) }}</div>
                   <div class="label">Recall</div></div>
  <div class="kpi"><div class="value">{{ overall.f1 | round(3) }}</div>
                   <div class="label">F1</div></div>
  <div class="kpi"><div class="value">{{ map50 | round(3) }}</div>
                   <div class="label">mAP@0.5</div></div>
  <div class="kpi"><div class="value">{{ overall.tp }}</div>
                   <div class="label">True positives</div></div>
  <div class="kpi"><div class="value">{{ overall.fp }}</div>
                   <div class="label">False positives</div></div>
  <div class="kpi"><div class="value">{{ overall.fn }}</div>
                   <div class="label">False negatives</div></div>
  <div class="kpi"><div class="value">{{ overall.num_images }}</div>
                   <div class="label">Images evaluated</div></div>
</div>

<h2>Per-class performance</h2>
<img src="plots/class_metrics.png" alt="Per-class metrics">
{{ class_table | safe }}

<h2>Confusion matrix</h2>
<img src="plots/confusion_matrix.png" alt="Confusion matrix">

<h2>Performance by condition</h2>
<img src="plots/condition_metrics.png" alt="Condition metrics">
{% for cond, table in condition_tables.items() %}
  <h3>{{ cond | capitalize }}</h3>
  {{ table | safe }}
{% endfor %}

<h2>Performance by object size</h2>
{{ size_table | safe }}

<h2>Precision-Recall curves</h2>
<img src="plots/pr_curves.png" alt="PR curves">

<h2>Confidence distribution</h2>
<img src="plots/confidence_histogram.png" alt="Confidence histogram">

<h2>Worst failure images</h2>
<p class="small">Top images by combined false positives + false negatives.</p>
<div class="failure-grid">
{% for f in worst_failures %}
  <div class="failure-card">
    {% if f.image_rendered %}<img src="{{ f.image_rendered }}" alt="failure">{% endif %}
    <div><b>{{ f.image_id }}</b></div>
    <div class="meta">FN={{ f.fn }} &middot; FP={{ f.fp }} &middot; TP={{ f.tp }}</div>
    <div class="meta">{{ f.lighting }} &middot; {{ f.weather }} &middot;
                       occlusion={{ f.occlusion }}</div>
  </div>
{% else %}
  <p>No failures over threshold.</p>
{% endfor %}
</div>

<h2>Dataset curation findings</h2>
<h3>Class balance</h3>
{{ class_balance_table | safe }}
{% if underrep_classes %}
  <p><b>Underrepresented classes:</b> {{ underrep_classes | join(", ") }}</p>
{% endif %}

<h3>Condition balance</h3>
{% for cond, table in condition_balance_tables.items() %}
  <h4>{{ cond | capitalize }}</h4>
  {{ table | safe }}
{% endfor %}

{% if suspicious_annotations %}
  <h3>Suspicious annotations</h3>
  <p>{{ suspicious_annotations | length }} flagged. First few:</p>
  <ul>
  {% for s in suspicious_annotations[:10] %}
    <li><code>{{ s.image_id }}</code> box #{{ s.box_index }} ({{ s.class }}):
        {{ s.issues | join(", ") }}</li>
  {% endfor %}
  </ul>
{% endif %}

{% if near_duplicates %}
  <h3>Near-duplicate image groups</h3>
  <ul>
  {% for g in near_duplicates[:10] %}
    <li>{{ g.members | join(", ") }}</li>
  {% endfor %}
  </ul>
{% endif %}

<h2>Recommendations</h2>
{% for r in recommendations %}
  <div class="rec">{{ r }}</div>
{% endfor %}

<h2>High-confidence false positives</h2>
<p class="small">Confident mistakes - good mining candidates for hard-negative training.</p>
{% if high_conf_wrong %}
  <table>
    <tr><th>Image</th><th>Class</th><th>Confidence</th><th>IoU</th>
        <th>Lighting</th><th>Weather</th></tr>
    {% for r in high_conf_wrong %}
      <tr><td>{{ r.image_id }}</td><td>{{ r["class"] }}</td>
          <td>{{ r.confidence | round(2) }}</td>
          <td>{{ r.iou | round(2) }}</td>
          <td>{{ r.lighting }}</td><td>{{ r.weather }}</td></tr>
    {% endfor %}
  </table>
{% else %}
  <p>None.</p>
{% endif %}

<h2>Missed safety-critical objects</h2>
{% if missed_critical %}
  <table>
    <tr><th>Image</th><th>Class</th><th>Occluded</th><th>Lighting</th>
        <th>Weather</th></tr>
    {% for r in missed_critical %}
      <tr><td>{{ r.image_id }}</td><td>{{ r["class"] }}</td>
          <td>{{ r.occluded }}</td>
          <td>{{ r.lighting }}</td><td>{{ r.weather }}</td></tr>
    {% endfor %}
  </table>
{% else %}
  <p>None.</p>
{% endif %}

<hr>
<p class="small">Report produced by the Autonomous Driving Data Quality &amp;
   Failure Analysis Platform.</p>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _df_to_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><i>No data.</i></p>"
    return df.round(3).to_html(index=False, border=0)


# --------------------------------------------------------------------------- #
# Plot generation
# --------------------------------------------------------------------------- #
def generate_plots(
    result: EvaluationResult,
    ground_truth: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Render all PNG plots into <output_dir>/plots/."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    out["class_metrics"] = V.plot_class_metrics(
        result.by_class, plots_dir / "class_metrics.png"
    )
    out["confusion_matrix"] = V.plot_confusion_matrix(
        result.confusion_matrix, plots_dir / "confusion_matrix.png"
    )
    out["condition_metrics"] = V.plot_condition_metrics(
        result.by_condition, plots_dir / "condition_metrics.png"
    )
    out["pr_curves"] = V.plot_pr_curves(
        result, plots_dir / "pr_curves.png"
    )
    out["confidence_histogram"] = V.plot_confidence_histogram(
        result, plots_dir / "confidence_histogram.png"
    )
    out["class_distribution"] = V.plot_class_distribution(
        class_distribution(ground_truth), plots_dir / "class_distribution.png"
    )
    return out


# --------------------------------------------------------------------------- #
# Failure image rendering
# --------------------------------------------------------------------------- #
def _render_failure_thumbnails(
    failures: List[Dict[str, Any]],
    result: EvaluationResult,
    ground_truth: Dict[str, List[Dict[str, Any]]],
    image_dir: Optional[Path],
    output_dir: Path,
    max_to_render: int,
) -> List[Dict[str, Any]]:
    """Add an image_rendered (relative path) to each failure case, if possible."""
    if image_dir is None or not Path(image_dir).exists():
        return failures
    fail_dir = output_dir / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)

    pred_df = result.predictions_df
    rendered: List[Dict[str, Any]] = []
    for f in failures[:max_to_render]:
        img_id = f["image_id"]
        src = Path(image_dir) / img_id
        if not src.exists():
            rendered.append(f)
            continue
        gt_boxes = ground_truth.get(img_id, [])
        pred_boxes = pred_df[pred_df["image_id"] == img_id].to_dict("records")
        dst = fail_dir / f"{Path(img_id).stem}_annotated.jpg"
        V.draw_boxes_on_image(src, gt_boxes, pred_boxes, dst)
        if dst.exists():
            f = {**f, "image_rendered": f"failures/{dst.name}"}
        rendered.append(f)
    # any failures past the rendering limit still appear in the table
    rendered.extend(failures[max_to_render:])
    return rendered


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def generate_report(
    result: EvaluationResult,
    mined: Dict[str, Any],
    curation: Dict[str, Any],
    recommendations: List[str],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    title: str = "Autonomous Driving Model Evaluation Report",
    image_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Generate HTML + Markdown reports plus all artefacts. Returns paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plots
    plot_paths = generate_plots(result, ground_truth, output_dir)
    logger.info("Wrote %d plots to %s", len(plot_paths), output_dir / "plots")

    # 2. Failure thumbnails
    report_cfg = result.config.get("report", {})
    n_failures_in_report = report_cfg.get("failures_in_report", 8)
    worst = list(mined.get("worst_images_by_fn", []))
    # de-duplicate while preserving order, prepending FN-heavy ones
    seen = {f["image_id"] for f in worst}
    for f in mined.get("worst_images_by_fp", []):
        if f["image_id"] not in seen:
            worst.append(f)
            seen.add(f["image_id"])

    worst = _render_failure_thumbnails(
        worst, result, ground_truth, image_dir, output_dir, n_failures_in_report
    )

    # 3. HTML report
    eval_cfg = result.config.get("evaluation", {})
    cb = curation.get("class_balance", {})
    class_balance_df = pd.DataFrame(cb.get("by_class", []))

    condition_balance_tables: Dict[str, str] = {}
    for cond, info in curation.get("condition_balance", {}).items():
        condition_balance_tables[cond] = _df_to_html(
            pd.DataFrame(info.get("values", []))
        )

    condition_tables = {
        cond: _df_to_html(df) for cond, df in result.by_condition.items()
    }

    html_ctx: Dict[str, Any] = {
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iou_threshold": eval_cfg.get("iou_threshold", 0.5),
        "conf_threshold": eval_cfg.get("confidence_threshold", 0.25),
        "overall": result.overall,
        "map50": result.mAP50,
        "class_table": _df_to_html(result.by_class),
        "size_table": _df_to_html(result.by_size),
        "condition_tables": condition_tables,
        "worst_failures": worst[:n_failures_in_report],
        "class_balance_table": _df_to_html(class_balance_df),
        "underrep_classes": cb.get("underrepresented_classes", []),
        "condition_balance_tables": condition_balance_tables,
        "suspicious_annotations": curation.get("suspicious_annotations", []),
        "near_duplicates": curation.get("near_duplicates", []),
        "recommendations": recommendations,
        "high_conf_wrong": mined.get("high_confidence_wrong", []),
        "missed_critical": mined.get("missed_critical", []),
    }

    html = Template(HTML_TEMPLATE).render(**html_ctx)
    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")

    # 4. Markdown report (text-only, no images)
    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(html_ctx, result), encoding="utf-8")

    # 5. Raw JSON dumps for downstream tools / the dashboard.
    (output_dir / "metrics.json").write_text(
        json.dumps(_metrics_to_json(result), indent=2), encoding="utf-8"
    )
    (output_dir / "failures.json").write_text(
        json.dumps(_sanitize(mined), indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "curation.json").write_text(
        json.dumps(_sanitize(curation), indent=2, default=str), encoding="utf-8"
    )
    result.predictions_df.to_csv(output_dir / "predictions_long.csv", index=False)
    result.ground_truth_df.to_csv(output_dir / "ground_truth_long.csv", index=False)

    logger.info("Report written to %s", html_path)
    return {"html": html_path, "markdown": md_path, "output_dir": output_dir}


# --------------------------------------------------------------------------- #
# Auxiliaries
# --------------------------------------------------------------------------- #
def _render_markdown(ctx: Dict[str, Any], result: EvaluationResult) -> str:
    overall = ctx["overall"]
    lines = [
        f"# {ctx['title']}",
        "",
        f"_Generated {ctx['generated_at']}._",
        "",
        "## Summary",
        f"- Precision: **{overall.get('precision', 0):.3f}**",
        f"- Recall:    **{overall.get('recall', 0):.3f}**",
        f"- F1:        **{overall.get('f1', 0):.3f}**",
        f"- mAP@0.5:   **{ctx['map50']:.3f}**",
        f"- TP / FP / FN: {overall.get('tp')} / {overall.get('fp')} / "
        f"{overall.get('fn')}",
        f"- Images: {overall.get('num_images')}",
        "",
        "## Per-class performance",
        result.by_class.round(3).to_markdown(index=False)
        if not result.by_class.empty else "_No data._",
        "",
        "## Performance by condition",
    ]
    for cond, df in result.by_condition.items():
        lines.append(f"### {cond}")
        lines.append(
            df.round(3).to_markdown(index=False) if not df.empty else "_No data._"
        )
        lines.append("")

    lines.append("## Recommendations")
    for r in ctx["recommendations"]:
        lines.append(f"- {r}")
    return "\n".join(lines)


def _metrics_to_json(result: EvaluationResult) -> Dict[str, Any]:
    return {
        "overall": result.overall,
        "mAP": result.mAP,
        "mAP50": result.mAP50,
        "by_class": result.by_class.to_dict(orient="records")
        if not result.by_class.empty else [],
        "by_size": result.by_size.to_dict(orient="records")
        if not result.by_size.empty else [],
        "by_condition": {
            cond: df.to_dict(orient="records")
            for cond, df in result.by_condition.items()
        },
        "ap_by_class": result.ap_by_class,
        "classes": result.classes,
    }


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to JSON-safe ones."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj
