from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


logger = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
BATCH_TIMEOUT = int(os.environ.get("BATCH_READ_TIMEOUT_SECONDS", "300"))
CHUNK_SIZE = int(os.environ.get("BATCH_CHUNK_SIZE", "10"))


def api_get(path: str, timeout: int = 10) -> Optional[dict]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.RequestException:
        logger.error("GET request failed for path %s", path, exc_info=True)
        return None


def time_ago(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - ts
        minutes = int(diff.total_seconds() / 60)
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return ts_str


def severity(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.65:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


st.title("History")
st.caption("Historical anomaly browser with filtering and export")

# Section 1 - Page guard
base_resp = api_get("/anomalies?limit=500")
if base_resp is None:
    st.error("API offline")
    st.stop()

limit_state = int(st.session_state.get("hist_limit", 500))
anomalies = base_resp.get("anomalies", [])
if limit_state != 500:
    limited_resp = api_get(f"/anomalies?limit={limit_state}")
    if limited_resp is not None:
        anomalies = limited_resp.get("anomalies", anomalies)

if not anomalies:
    st.info("No anomalies stored yet. Run a batch on the Analyze page to generate results.")
    if st.button("Go to Analyze ->"):
        st.switch_page("pages/2_Analyze.py")
    st.stop()

# Precompute filter outputs to support export button in the same row.
adf = pd.DataFrame(anomalies)
all_types = sorted(adf["anomaly_type"].dropna().astype(str).unique().tolist())
selected_types = st.session_state.get("hist_types", all_types)
if not selected_types:
    selected_types = all_types
min_score = float(st.session_state.get("hist_score", 0.0))

adf["severity"] = adf["score"].apply(severity)
adf["age"] = adf["created_at"].apply(time_ago)
filtered_df = adf[
    (adf["anomaly_type"].isin(selected_types)) &
    (adf["score"] >= min_score)
] if not adf.empty else adf

# Section 2 - Filter bar + export button
st.markdown("### Filters")
filter_col1, filter_col2, filter_col3, export_col = st.columns([2, 2, 1, 1])
with filter_col1:
    st.multiselect("Anomaly type", options=all_types, default=all_types, key="hist_types")
with filter_col2:
    st.slider("Min score", 0.0, 1.0, 0.0, step=0.05, key="hist_score")
with filter_col3:
    st.selectbox("Limit", [100, 250, 500], index=[100, 250, 500].index(limit_state) if limit_state in [100, 250, 500] else 2, key="hist_limit")
with export_col:
    st.download_button(
        "Export CSV",
        data=filtered_df.to_csv(index=False),
        file_name="contextwatch_anomalies.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Section 3 - Summary metrics row
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Filtered anomalies", len(filtered_df))
metric_col2.metric("Unique types", int(filtered_df["anomaly_type"].nunique()) if not filtered_df.empty else 0)
if filtered_df.empty:
    metric_col3.metric("Avg score", "N/A")
else:
    metric_col3.metric("Avg score", f"{filtered_df['score'].mean():.3f}")

# Section 4 - Anomaly table
table_cols = [
    c for c in ["anomaly_id", "log_id", "anomaly_type", "severity", "score", "confidence", "age"]
    if c in filtered_df.columns
]
height = min(400, 50 + len(filtered_df) * 35)
st.dataframe(
    filtered_df[table_cols].style.format({"score": "{:.4f}", "confidence": "{:.2%}"}),
    use_container_width=True,
    height=height,
)

# Section 5 - Anomaly type distribution chart
if len(filtered_df) > 0:
    type_counts = filtered_df["anomaly_type"].value_counts().reset_index()
    type_counts.columns = ["anomaly_type", "count"]
    fig = px.pie(
        type_counts,
        values="count",
        names="anomaly_type",
        title=f"Anomaly type distribution ({len(filtered_df)} records)",
    )
    st.plotly_chart(fig, use_container_width=True)
