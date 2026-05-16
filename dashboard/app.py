"""
Streamlit dashboard.

Two modes:

  A) "Live" mode  - point it at a GT/predictions/metadata triple via the
                    sidebar and it runs the full pipeline in-process.
  B) "Report" mode - point it at an existing reports/<run> directory and
                    it reads the JSON/CSV artefacts already on disk.

Run with:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    load_config,
    load_ground_truth,
    load_metadata,
    load_predictions,
)
from src.evaluator import Evaluator, EvaluationResult  # noqa: E402
from src.failure_analysis import FailureMiner, generate_recommendations  # noqa: E402
from src.curation import run_curation  # noqa: E402
from src import visualization as V  # noqa: E402


st.set_page_config(
    page_title="AD Data Quality Dashboard",
    page_icon="🚗",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Sidebar: configure data sources
# --------------------------------------------------------------------------- #
st.sidebar.title("🚗 AD Data Quality")
mode = st.sidebar.radio("Mode", ["Live evaluation", "Load existing report"])


@st.cache_data(show_spinner=False)
def _run_pipeline_cached(
    gt_path: str, pred_path: str, meta_path: str, config_path: str
):
    """Cache so re-runs are instant."""
    config = load_config(config_path)
    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)
    meta = load_metadata(meta_path) if meta_path else {}
    result = Evaluator(gt, preds, meta, config).run()
    miner = FailureMiner(result, config)
    mined = miner.mine()
    recs = generate_recommendations(result, mined, config)
    curation = run_curation(gt, meta, config)
    return result, mined, recs, curation, config


def _live_inputs():
    st.sidebar.subheader("Files")
    gt = st.sidebar.text_input(
        "Ground truth JSON",
        value=str(ROOT / "data" / "sample" / "ground_truth.json"),
    )
    pr = st.sidebar.text_input(
        "Predictions JSON",
        value=str(ROOT / "data" / "sample" / "predictions.json"),
    )
    md = st.sidebar.text_input(
        "Metadata JSON (optional)",
        value=str(ROOT / "data" / "sample" / "metadata.json"),
    )
    cfg = st.sidebar.text_input(
        "Config YAML", value=str(ROOT / "config" / "default.yaml")
    )
    if st.sidebar.button("Run evaluation", type="primary"):
        st.cache_data.clear()
    return gt, pr, md, cfg


def _report_inputs():
    st.sidebar.subheader("Report directory")
    return st.sidebar.text_input(
        "Path to a previous run",
        value=str(ROOT / "reports" / "demo_report"),
    )


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
result: EvaluationResult | None = None
mined: dict = {}
recs: list = []
curation: dict = {}

if mode == "Live evaluation":
    gt_path, pr_path, md_path, cfg_path = _live_inputs()
    if not Path(gt_path).exists() or not Path(pr_path).exists():
        st.warning("Provide valid ground truth and predictions paths in the sidebar.")
        st.stop()
    try:
        result, mined, recs, curation, _ = _run_pipeline_cached(
            gt_path, pr_path, md_path, cfg_path
        )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

else:
    report_dir = Path(_report_inputs())
    if not report_dir.exists():
        st.warning("Report directory does not exist. Run an evaluation first.")
        st.stop()
    metrics = json.loads((report_dir / "metrics.json").read_text())
    mined = json.loads((report_dir / "failures.json").read_text())
    curation = json.loads((report_dir / "curation.json").read_text())
    pred_df = pd.read_csv(report_dir / "predictions_long.csv")
    gt_df = pd.read_csv(report_dir / "ground_truth_long.csv")

    # Reconstitute a minimal EvaluationResult so the same render code works.
    from dataclasses import asdict  # noqa: F401
    result = EvaluationResult(
        predictions_df=pred_df,
        ground_truth_df=gt_df,
        overall=metrics["overall"],
        by_class=pd.DataFrame(metrics["by_class"]),
        by_condition={
            k: pd.DataFrame(v) for k, v in metrics["by_condition"].items()
        },
        by_size=pd.DataFrame(metrics["by_size"]),
        ap_by_class=metrics.get("ap_by_class", {}),
        mAP=metrics.get("mAP", 0.0),
        mAP50=metrics.get("mAP50", 0.0),
        confusion_matrix=pd.DataFrame(),  # not persisted; tab will say so
        classes=metrics.get("classes", []),
    )
    recs = []  # not persisted; OK


# --------------------------------------------------------------------------- #
# Header / KPIs
# --------------------------------------------------------------------------- #
st.title("Autonomous Driving Data Quality Dashboard")

overall = result.overall
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{overall.get('precision', 0):.3f}")
c2.metric("Recall", f"{overall.get('recall', 0):.3f}")
c3.metric("F1", f"{overall.get('f1', 0):.3f}")
c4.metric("mAP@0.5", f"{result.mAP50:.3f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("True positives", overall.get("tp", 0))
c6.metric("False positives", overall.get("fp", 0))
c7.metric("False negatives", overall.get("fn", 0))
c8.metric("Images evaluated", overall.get("num_images", 0))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs([
    "Per-class",
    "Conditions",
    "Confusion",
    "Confidence",
    "Failures",
    "Curation",
    "Recommendations",
])

# ----- Per-class -----
with tabs[0]:
    st.subheader("Per-class performance")
    fig = V.plotly_class_metrics(result.by_class)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result.by_class, use_container_width=True)

    st.subheader("AP per class")
    ap_rows = [
        {"class": c, "AP50": v.get("ap50", 0), "AP_mean_over_thresholds": v.get("ap_mean", 0)}
        for c, v in result.ap_by_class.items()
    ]
    if ap_rows:
        st.dataframe(pd.DataFrame(ap_rows).sort_values("AP50", ascending=False),
                     use_container_width=True)

# ----- Conditions -----
with tabs[1]:
    st.subheader("Performance by condition")
    for cond in ["lighting", "weather", "occlusion"]:
        fig = V.plotly_condition_metrics(result.by_condition, cond)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result.by_condition[cond], use_container_width=True)
        else:
            st.info(f"No data for {cond}.")

    st.subheader("Performance by object size")
    st.dataframe(result.by_size, use_container_width=True)

# ----- Confusion matrix -----
with tabs[2]:
    st.subheader("Confusion matrix")
    if result.confusion_matrix.empty:
        st.info("Confusion matrix is not stored in saved reports. Use live mode.")
    else:
        fig = V.plotly_confusion_matrix(result.confusion_matrix)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result.confusion_matrix, use_container_width=True)

# ----- Confidence -----
with tabs[3]:
    st.subheader("Confidence histogram (TP vs FP)")
    fig = V.plotly_confidence_hist(result.predictions_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions to plot.")

# ----- Failures -----
with tabs[4]:
    st.subheader("Worst images")
    counts_df = pd.DataFrame(mined.get("per_image_counts", []))
    if not counts_df.empty:
        st.dataframe(
            counts_df.sort_values("total_errors", ascending=False).head(50),
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("High-confidence false positives")
        hcw = pd.DataFrame(mined.get("high_confidence_wrong", []))
        st.dataframe(hcw, use_container_width=True) if not hcw.empty else st.info("None.")
    with col2:
        st.subheader("Low-confidence correct")
        lcc = pd.DataFrame(mined.get("low_confidence_correct", []))
        st.dataframe(lcc, use_container_width=True) if not lcc.empty else st.info("None.")

    st.subheader("Missed safety-critical objects")
    mc = pd.DataFrame(mined.get("missed_critical", []))
    st.dataframe(mc, use_container_width=True) if not mc.empty else st.info("None.")

# ----- Curation -----
with tabs[5]:
    st.subheader("Class balance")
    cb = curation.get("class_balance", {})
    cb_rows = cb.get("by_class", [])
    if cb_rows:
        st.dataframe(pd.DataFrame(cb_rows), use_container_width=True)
        underrep = cb.get("underrepresented_classes", [])
        if underrep:
            st.warning(f"Underrepresented classes: {', '.join(underrep)}")

    st.subheader("Condition balance")
    for cond, info in curation.get("condition_balance", {}).items():
        st.markdown(f"**{cond}**")
        vals = info.get("values", [])
        if vals:
            st.dataframe(pd.DataFrame(vals), use_container_width=True)
            under = info.get("underrepresented", [])
            if under:
                st.warning(f"Underrepresented {cond}: {', '.join(under)}")

    susp = curation.get("suspicious_annotations", [])
    if susp:
        st.subheader(f"Suspicious annotations ({len(susp)})")
        st.dataframe(pd.DataFrame(susp), use_container_width=True)

    dups = curation.get("near_duplicates", [])
    if dups:
        st.subheader("Near-duplicate image groups")
        st.json(dups)

# ----- Recommendations -----
with tabs[6]:
    st.subheader("Recommendations for retraining / data collection")
    if recs:
        for r in recs:
            st.info(r)
    else:
        st.write("_No recommendations available (run in live mode to regenerate)._")
