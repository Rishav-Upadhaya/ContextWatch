from __future__ import annotations

import json
import logging
import os
import time as _time
from typing import Optional

import pandas as pd
import requests
import streamlit as st

from pages._analyze_helpers import (
    ingest_batch_api,
    parse_logs,
    render_anomaly_details,
    severity,
)
from pages._graph_helpers import render_searchable_graph


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


st.title("Analyze")
st.caption("Batch log ingestion, anomaly triage, RCA, and graph investigation")

if "batch_input" not in st.session_state:
    st.session_state["batch_input"] = ""

normal_example = json.dumps(
    [
        {
            "log_id": "n-001",
            "protocol": "MCP",
            "method": "tool_call",
            "params": {"message": "tool search completed successfully"},
            "timestamp": "2026-04-13T10:00:00Z",
        },
        {
            "log_id": "n-002",
            "protocol": "A2A",
            "method": "tool_call",
            "params": {"message": "memory context loaded"},
            "timestamp": "2026-04-13T10:00:01Z",
        },
        {
            "log_id": "n-003",
            "protocol": "MCP",
            "method": "tool_call",
            "params": {"message": "response generated for user prompt"},
            "timestamp": "2026-04-13T10:00:02Z",
        },
    ],
    indent=2,
)

anomaly_example = json.dumps(
    [
        {
            "log_id": "a-001",
            "protocol": "MCP",
            "method": "tool_call",
            "params": {"event": "TOOL_HALLUCINATION", "message": "agent returned fabricated tool output"},
            "timestamp": "2026-04-13T10:01:00Z",
        },
        {
            "log_id": "a-002",
            "protocol": "A2A",
            "method": "tool_call",
            "params": {"event": "CONTEXT_POISONING", "message": "tainted context propagated across agents"},
            "timestamp": "2026-04-13T10:01:01Z",
        },
    ],
    indent=2,
)

st.markdown("### Batch input")
st.text_area("Paste logs (JSON array / JSONL / {\"logs\": [...]})", height=200, key="batch_input")
b1, b2, b3 = st.columns(3)
if b1.button("Load example (normal logs)", use_container_width=True):
    st.session_state["batch_input"] = normal_example
    st.rerun()
if b2.button("Load example (anomaly logs)", use_container_width=True):
    st.session_state["batch_input"] = anomaly_example
    st.rerun()
if b3.button("Clear", use_container_width=True):
    st.session_state["batch_input"] = ""
    st.rerun()

uploaded = st.file_uploader("Or upload a .json / .jsonl file", type=["json", "jsonl"])
if uploaded is not None:
    uploaded_text = uploaded.getvalue().decode("utf-8")
    if st.session_state.get("_uploaded_batch_text") != uploaded_text:
        st.session_state["batch_input"] = uploaded_text
        st.session_state["_uploaded_batch_text"] = uploaded_text
        st.rerun()

st.markdown("### Run controls")
include_rca = st.checkbox("Run deep RCA (slower, richer insights)", value=False)
run_col1, run_col2 = st.columns([2, 1])
run_batch = run_col1.button("Analyze Batch", type="primary", use_container_width=True)
if run_col2.button("Clear Results", use_container_width=True):
    st.session_state.pop("batch_result", None)
    st.session_state.pop("graph_data", None)
    st.rerun()

with st.expander("View API request (curl)", expanded=False):
    try:
        preview_logs = parse_logs(st.session_state.get("batch_input", ""))[:2]
        curl_cmd = (
            f'curl -X POST {API_URL}/ingest/batch \\\n'
            f'  -H "Content-Type: application/json" \\\n'
            f"  -d '{json.dumps({'logs': preview_logs, 'include_rca': include_rca}, indent=2)}'"
        )
        st.code(curl_cmd, language="bash")
    except (ValueError, Exception):
        st.code(f"curl -X POST {API_URL}/ingest/batch -H \"Content-Type: application/json\" -d '{{...}}'", language="bash")

if run_batch:
    try:
        logs = parse_logs(st.session_state.get("batch_input", ""))
        with st.spinner(f"Processing {len(logs)} logs..."):
            result, error = ingest_batch_api(logs, include_rca=include_rca)
        if error:
            st.error(f"{error}")
        else:
            st.session_state["batch_result"] = result
            st.session_state["graph_data"] = result.get("graph", {})
            st.session_state["last_run_summary"] = (
                f"Last run: {len(logs)} logs · "
                f"{(len(logs) + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks · "
                f"{result['anomaly_count']} anomalies · "
                f"{result['normal_count']} normal · "
                f"completed at {_time.strftime('%H:%M:%S')}"
            )
            st.rerun()
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")

if st.session_state.get("last_run_summary"):
    st.info(st.session_state["last_run_summary"])

batch_result = st.session_state.get("batch_result")
if batch_result:
    st.markdown("### Results table")
    df = pd.DataFrame(batch_result.get("results", []))
    if not df.empty and "anomaly_score" in df.columns:
        df["severity"] = df["anomaly_score"].apply(severity)
    show_cols = [
        c
        for c in [
            "severity",
            "log_id",
            "anomaly_type",
            "explanation",
            "confidence",
            "anomaly_score",
            "cosine_distance",
        ]
        if c in df.columns
    ]

    def color_row(row: pd.Series) -> list[str]:
        bg = "background-color: rgba(127,29,29,0.3)" if row.get("is_anomaly") else "background-color: rgba(20,83,45,0.3)"
        return [bg] * len(row)

    if show_cols:
        st.dataframe(
            df[show_cols].style.format(
                {"confidence": "{:.2%}", "anomaly_score": "{:.4f}", "cosine_distance": "{:.4f}"}
            ).apply(color_row, axis=1),
            use_container_width=True,
            height=320,
        )

    anom_df = df[df["is_anomaly"]] if "is_anomaly" in df.columns else df.iloc[0:0]
    if not anom_df.empty:
        st.markdown("### Anomaly detail and RCA")
        render_anomaly_details(anom_df)

    if batch_result.get("anomaly_count", 0) > 0:
        st.markdown("### Knowledge graph")
        render_searchable_graph(st.session_state.get("graph_data", {}))
