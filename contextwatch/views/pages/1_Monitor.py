from __future__ import annotations

import logging
import os
import time
from collections import Counter
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


def format_metric_value(value: Optional[float]) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def time_ago(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - ts
        minutes = int(diff.total_seconds() / 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return ts_str


st.title("Monitor")
st.caption("Live operational status and anomaly trends")

health = api_get("/health")
if health is None:
    st.error("API offline")
    st.stop()

model_info = api_get("/model/info") or {}
anomaly_resp = api_get("/anomalies?limit=500") or {}
anomalies = anomaly_resp.get("anomalies", [])
st.session_state["monitor_anomalies"] = anomalies

st.markdown("### System health")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    status_text = str(health.get("status", "unknown")).lower()
    if status_text in {"ok", "healthy", "online"}:
        st.success("API status: Online")
    else:
        st.error("API status: Offline")

with col2:
    vhm_fitted = bool(model_info.get("vhm_fitted", False))
    if vhm_fitted:
        st.success("VHM status: Fitted")
    else:
        st.error("VHM status: Not Fitted")

with col3:
    radius = model_info.get("vhm_radius")
    st.metric("VHM radius", format_metric_value(radius))

with col4:
    d_model = model_info.get("d_model", "?")
    n_layers = model_info.get("n_layers", "?")
    st.metric("Model version", f"LogBERT d={d_model} L={n_layers}")

with col5:
    st.caption("Adminer")
    st.markdown("[Open localhost:8080](http://localhost:8080)")

with st.expander("LogBERT architecture details", expanded=False):
    st.caption("Pure NumPy transformer trained from scratch - no deep learning frameworks.")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("d_model", model_info.get("d_model", "N/A"))
    a2.metric("n_heads", model_info.get("n_heads", "N/A"))
    a3.metric("n_layers", model_info.get("n_layers", "N/A"))
    a4.metric("max_seq_len", model_info.get("max_seq_len", "N/A"))

st.markdown("### Anomaly rate summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)

if anomalies:
    type_counts = Counter(str(a.get("anomaly_type", "unknown")) for a in anomalies)
    top_type, _ = type_counts.most_common(1)[0]

    latest_created_at = max(
        (str(a.get("created_at", "")) for a in anomalies if a.get("created_at")),
        default="",
    )
    last_seen = time_ago(latest_created_at) if latest_created_at else "N/A"

    summary_col1.metric("Total anomalies stored", len(anomalies))
    summary_col2.metric("Most common anomaly type", top_type)
    summary_col3.metric("Last seen", last_seen)

    adf = pd.DataFrame(
        {"anomaly_type": list(type_counts.keys()), "count": list(type_counts.values())}
    )
    fig = px.pie(adf, values="count", names="anomaly_type", title="Anomaly type distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    summary_col1.metric("Total anomalies stored", 0)
    summary_col2.metric("Most common anomaly type", "N/A")
    summary_col3.metric("Last seen", "N/A")
    st.info("No anomalies stored yet. Run a batch on the Analyze page to generate results.")
    if st.button("Go to Analyze ->"):
        st.switch_page("pages/2_Analyze.py")

auto_refresh = st.checkbox("Auto-refresh every 30s")
if auto_refresh:
    time.sleep(30)
    st.rerun()
