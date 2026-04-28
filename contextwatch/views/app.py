from __future__ import annotations

import logging
import os
from typing import Optional

import requests
import streamlit as st


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ContextWatch",
    page_icon="CW",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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


st.title("ContextWatch")
st.caption("AI-driven log anomaly detection for developer and operator workflows")

st.info("Pipeline: Logs -> Normalize -> LogBERT -> VHM -> RCA -> PostgreSQL")
st.caption("Normal logs stored in PostgreSQL - browse via Adminer at localhost:8080")

with st.container(border=True):
    st.subheader("Project Overview")
    st.write(
        "ContextWatch combines a custom LogBERT encoder, VHM anomaly scoring, "
        "MA-RCA reasoning, and graph-assisted investigation in a production-ready pipeline."
    )
    st.write(
        "Use the pages in the sidebar to monitor runtime status, analyze new log batches, "
        "review historical anomalies, and run controlled finetuning."
    )

col1, col2, col3, col4 = st.columns(4)
col1.page_link("pages/1_Monitor.py", label="Open Monitor")
col2.page_link("pages/2_Analyze.py", label="Open Analyze")
col3.page_link("pages/3_History.py", label="Open History")
col4.page_link("pages/4_Train.py", label="Open Train")

health = api_get("/health")
if health:
    st.success("API is reachable.")
else:
    st.warning(f"Could not reach API at {API_URL}. Start the backend to enable dashboard pages.")
