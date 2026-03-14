from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


st.set_page_config(page_title="ContextWatch", layout="wide")
st.title("ContextWatch Dashboard")
st.caption("AI-powered observability for MCP and A2A logs")

if st.button("Check API Health"):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        st.json(resp.json())
    except Exception as exc:
        st.error(str(exc))

st.info(
    "Use the Pages sidebar to navigate: Overview, Cognitive Trace, Knowledge Graph, RCA, "
    "Group Analysis, and ML-based Group Analysis."
)
