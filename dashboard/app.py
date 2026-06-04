"""Streamlit dashboard for exploring benchmark results.

Run with:  streamlit run dashboard/app.py

Reads the artifacts produced by `python -m evalharness all`:
  - results/summary.parquet   (one row per model × prompt)
  - results/raw_runs.parquet  (one row per individual call, for drill-down)
"""
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary.parquet"
RAW_PATH = RESULTS_DIR / "raw_runs.parquet"

st.set_page_config(page_title="LLM Eval Harness", layout="wide")
st.title("LLM Evaluation & Benchmarking Harness")

if not SUMMARY_PATH.exists():
    st.warning(
        "No results found. Run `python -m evalharness all` first to generate "
        "results/summary.parquet and results/raw_runs.parquet."
    )
    st.stop()

summary = pd.read_parquet(SUMMARY_PATH)

# ── Summary table ────────────────────────────────────────────────────────────
st.header("Summary")
display_cols = [
    "model_label", "prompt_file", "n", "macro_f1", "json_validity",
    "exact_match_rate", "latency_p50_ms", "latency_p95_ms", "cost_per_1k_usd",
]
cols = [c for c in display_cols if c in summary.columns]
st.dataframe(
    summary[cols].sort_values("macro_f1", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# ── Charts ───────────────────────────────────────────────────────────────────
st.header("Charts")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Macro F1 by model × prompt")
    st.bar_chart(summary.set_index("model_label")["macro_f1"])
with c2:
    st.subheader("Median latency (ms)")
    st.bar_chart(summary.set_index("model_label")["latency_p50_ms"])

# ── Per-example drill-down ───────────────────────────────────────────────────
if RAW_PATH.exists():
    st.header("Per-example drill-down")
    raw = pd.read_parquet(RAW_PATH)

    model = st.selectbox("Model", sorted(raw["model_label"].unique()))
    prompt = st.selectbox("Prompt", sorted(raw["prompt_file"].unique()))

    subset = raw[(raw["model_label"] == model) & (raw["prompt_file"] == prompt)]
    only_errors = st.checkbox("Show only errored / invalid rows", value=False)
    if only_errors:
        subset = subset[subset["is_valid"] != True]  # noqa: E712 — parquet bool column

    drill_cols = [
        c for c in ["example_id", "input", "output_text", "is_valid", "latency_ms", "error"]
        if c in subset.columns
    ]
    st.dataframe(subset[drill_cols], use_container_width=True, hide_index=True)
    st.caption(f"{len(subset)} rows for {model} / {prompt}")
