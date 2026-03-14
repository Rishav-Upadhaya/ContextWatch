from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for dashboard imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dashboard.components.log_table import render_log_table


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Overview")

stats = requests.get(f"{API_URL}/stats", timeout=10).json()["data"]
anomalies = requests.get(f"{API_URL}/anomalies", timeout=10).json()["data"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Logs Processed", stats.get("total_logs", 0))
col2.metric("Anomaly Rate %", round(100 * stats.get("anomaly_rate", 0), 2))
col3.metric("Avg Detection Latency", "N/A")

if anomalies:
    df = pd.DataFrame(anomalies)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        st.warning("Anomaly records have invalid timestamps.")
        st.stop()
    timeline = df.set_index("timestamp").resample("1H").size().reset_index(name="anomalies")
    st.plotly_chart(px.line(timeline, x="timestamp", y="anomalies", title="Anomalies per hour"), use_container_width=True)
    donut = df.groupby("anomaly_type").size().reset_index(name="count")
    st.plotly_chart(px.pie(donut, names="anomaly_type", values="count", hole=0.5, title="Type distribution"), use_container_width=True)

    st.subheader("Detection Lane Diagnostics")
    lane_df = df.copy()
    lane_df["rule_lane_triggered"] = lane_df["rule_lane_triggered"].fillna(False)
    lane_df["embedding_lane_score"] = pd.to_numeric(lane_df.get("embedding_lane_score"), errors="coerce").fillna(0.0)
    lane_df["embedding_lane_threshold"] = pd.to_numeric(lane_df.get("embedding_lane_threshold"), errors="coerce").fillna(0.0)

    l1, l2, l3 = st.columns(3)
    l1.metric("Rule Lane Triggered", int(lane_df["rule_lane_triggered"].sum()))
    l2.metric("Avg Embedding Score", f"{lane_df['embedding_lane_score'].mean():.4f}")
    l3.metric("Avg Embedding Threshold", f"{lane_df['embedding_lane_threshold'].mean():.4f}")

    if "arbitration_mode" in lane_df.columns and not lane_df["arbitration_mode"].dropna().empty:
        mode_df = lane_df.groupby("arbitration_mode").size().reset_index(name="count")
        st.plotly_chart(
            px.bar(mode_df, x="arbitration_mode", y="count", title="Arbitration Mode Distribution"),
            use_container_width=True,
        )

    render_log_table(df.head(20).to_dict(orient="records"), title="Recent anomalies")
else:
    st.info("No anomalies available yet.")
