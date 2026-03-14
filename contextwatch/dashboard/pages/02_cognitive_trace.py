from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Cognitive Trace")
anomalies = requests.get(f"{API_URL}/anomalies", timeout=10).json()["data"]
sessions = sorted({item["log_id"] for item in anomalies})
selected = st.selectbox("Select anomaly log id", options=sessions) if sessions else None

try:
    cognitive = requests.get(f"{API_URL}/stats/cognitive", timeout=10).json().get("data", {})
except Exception:
    cognitive = {}

st.subheader("Thoughts → Actions Coherence")
k1, k2, k3 = st.columns(3)
k1.metric("Avg Coherence", f"{(cognitive.get('avg_thought_action_coherence', 0.0) * 100):.1f}%")
k2.metric("Avg Intent-Outcome Gap", f"{(cognitive.get('avg_intent_outcome_gap', 0.0) * 100):.1f}%")
k3.metric("High-Gap Logs", int(cognitive.get("high_gap_count", 0)))

if selected:
    detail = requests.get(f"{API_URL}/anomalies/{selected}", timeout=10).json()["data"]
    log = detail["log"]
    st.subheader("Intent → Context → Tool → Response")
    c1, c2, c3, c4 = st.columns(4)
    c1.info(log["metadata"].get("intent") or log["metadata"].get("task_intent"))
    c2.warning(log["metadata"].get("context_summary") or str(log["metadata"].get("context_carried")))
    c3.write(log["metadata"].get("tool_name") or log["metadata"].get("target_agent"))
    c4.success(log["metadata"].get("response_status"))
    st.markdown(f"**Anomaly Type:** {detail['classification']['anomaly_type']}")
    if detail.get("explanation"):
        st.write(detail["explanation"])
