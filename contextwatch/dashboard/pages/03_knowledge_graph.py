from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for dashboard imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
import streamlit as st
import streamlit.components.v1 as components

from dashboard.components.graph_viz import build_graph_html


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Knowledge Graph")
session_id = st.text_input("Session ID")

if session_id:
    payload = requests.get(f"{API_URL}/graph/session/{session_id}", timeout=10).json()["data"]
    logs = payload["logs"]
    nodes = []
    edges = []
    for row in logs:
        log_id = row["log_id"]
        is_anomaly = bool(row["metadata"].get("is_anomaly"))
        nodes.append({"id": log_id, "label": log_id[:8], "color": "#ef4444" if is_anomaly else "#22c55e"})
    for i in range(1, len(logs)):
        edges.append({"source": logs[i - 1]["log_id"], "target": logs[i]["log_id"], "color": "#fb923c"})
    html = build_graph_html(nodes, edges)
    components.html(html, height=600, scrolling=True)
    st.json(payload["density"])
