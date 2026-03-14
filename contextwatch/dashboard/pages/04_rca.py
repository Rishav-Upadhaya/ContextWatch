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

st.title("Root Cause Analysis")
log_id = st.text_input("Anomaly log_id")

if st.button("Run RCA") and log_id:
    data = requests.get(f"{API_URL}/graph/rca/{log_id}", timeout=10).json()["data"]
    st.write(" → ".join(data["causal_chain"]))
    st.caption(f"Hops: {data['hop_count']}")
    st.write(data["explanation"])

    trigger = requests.get(f"{API_URL}/graph/trigger/{log_id}", timeout=10).json()["data"]
    st.subheader("Trigger Graph")
    nodes = [
        {
            "id": n["id"],
            "label": n.get("label", n["id"][:8]),
            "color": "#ef4444" if n["id"] == log_id else "#f59e0b",
        }
        for n in trigger.get("nodes", [])
    ]
    edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "color": "#fb923c" if e.get("type") == "SEQUENTIAL_CAUSAL_HINT" else "#a855f7",
            "label": e.get("type", ""),
        }
        for e in trigger.get("edges", [])
    ]
    if nodes:
        html = build_graph_html(nodes, edges)
        components.html(html, height=460, scrolling=True)
    st.caption(f"Root log: {trigger.get('root_log_id')}")
