from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

from dashboard.components.graph_viz import build_graph_html


API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("DASHBOARD_API_KEY", "")
ANOMALY_TYPES = [
    "TOOL_HALLUCINATION",
    "CONTEXT_POISONING",
    "REGISTRY_OVERFLOW",
    "DELEGATION_CHAIN_FAILURE",
]


st.title("ML-based Group Analysis")
st.caption("Analyze a file or selected subset using ML detector lane scoring and arbitration diagnostics.")


def _parse_upload(content: str) -> list[dict]:
    rows: list[dict] = []
    stripped = content.strip()
    if not stripped:
        return rows

    if stripped.startswith("["):
        loaded = json.loads(stripped)
        if isinstance(loaded, list):
            rows.extend([x for x in loaded if isinstance(x, dict)])
        return rows

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _meta_from_raw(raw: dict) -> tuple[str, str, str]:
    protocol = str(raw.get("protocol", ""))
    if protocol == "MCP":
        params = raw.get("params", {}) if isinstance(raw.get("params"), dict) else {}
        data = params.get("data", {}) if isinstance(params.get("data"), dict) else {}
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        session = raw.get("session", {}) if isinstance(raw.get("session"), dict) else {}
        return str(session.get("id", "")), str(meta.get("tool", "")), str(data.get("event", ""))
    session_id = str(raw.get("session_id", ""))
    target = str(raw.get("target_agent", ""))
    message_type = str(raw.get("message_type", ""))
    return session_id, target, message_type


def _build_cause_key(detail: dict) -> str:
    anomaly_type = str(detail.get("classification", {}).get("anomaly_type") or "UNKNOWN")
    log = detail.get("log", {}) if isinstance(detail.get("log"), dict) else {}
    metadata = log.get("metadata", {}) if isinstance(log.get("metadata"), dict) else {}
    reason = str(detail.get("classification", {}).get("reasoning") or "")
    tool = str(metadata.get("tool_name") or metadata.get("target_agent") or "n/a")
    event = str(metadata.get("event") or metadata.get("message_type") or "n/a")
    reason_tag = (reason[:80] + "...") if len(reason) > 80 else reason
    return f"{anomaly_type} | {tool} | {event} | {reason_tag or 'ml_lane'}"


def _run_group_analysis_ml(rows: list[dict], api_key: str) -> tuple[list[dict], list[dict], dict | None]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()

    processed: list[dict] = []
    failures: list[dict] = []

    for idx, row in enumerate(rows, 1):
        try:
            response = requests.post(f"{API_URL}/analyze", json=row, headers=headers, timeout=20)
            if response.status_code >= 400:
                failures.append({"index": idx, "status": response.status_code, "body": response.text[:300]})
                continue

            payload = response.json().get("data", {})
            log_id = payload.get("log_id")
            detail = None
            if payload.get("is_anomaly") and log_id:
                detail_response = requests.get(f"{API_URL}/anomalies/{log_id}", headers=headers, timeout=15)
                if detail_response.status_code == 200:
                    detail = detail_response.json().get("data")

            session_id, tool_or_target, event_or_type = _meta_from_raw(row)
            processed.append(
                {
                    "index": idx,
                    "log_id": log_id,
                    "protocol": row.get("protocol"),
                    "session_id": session_id,
                    "tool_or_target": tool_or_target,
                    "event_or_type": event_or_type,
                    "is_anomaly": bool(payload.get("is_anomaly")),
                    "anomaly_type": payload.get("anomaly_type"),
                    "anomaly_score": payload.get("anomaly_score"),
                    "confidence": payload.get("confidence"),
                    "rule_lane_triggered": bool(payload.get("rule_lane_triggered", False)),
                    "embedding_lane_score": payload.get("embedding_lane_score"),
                    "embedding_lane_threshold": payload.get("embedding_lane_threshold"),
                    "arbitration_mode": payload.get("arbitration_mode"),
                    "detail": detail,
                }
            )
        except Exception as exc:
            failures.append({"index": idx, "status": "exception", "body": str(exc)[:300]})

    if not processed:
        return processed, failures, None

    result_df = pd.DataFrame(processed)
    total = len(result_df)
    anomalies = int(result_df["is_anomaly"].sum())
    mode_counts = Counter([str(x) for x in result_df["arbitration_mode"].fillna("unknown").tolist()])

    ml_summary = {
        "total_logs_analysed": total,
        "anomalies_found": anomalies,
        "anomaly_density_pct": round((anomalies / total * 100.0), 2) if total else 0.0,
        "rule_lane_triggered_count": int(result_df["rule_lane_triggered"].sum()),
        "avg_embedding_lane_score": round(float(pd.to_numeric(result_df["embedding_lane_score"], errors="coerce").fillna(0.0).mean()), 4),
        "avg_embedding_lane_threshold": round(float(pd.to_numeric(result_df["embedding_lane_threshold"], errors="coerce").fillna(0.0).mean()), 4),
        "by_arbitration_mode": dict(mode_counts),
        "assessment": "ML group analysis completed with lane-level detector diagnostics.",
    }

    return processed, failures, ml_summary


api_key = st.text_input("API Key (required if auth enabled)", value=DEFAULT_API_KEY, type="password")
source = st.radio("Input source", options=["Upload file", "Paste logs"], horizontal=True)

raw_rows: list[dict] = []
if source == "Upload file":
    uploaded = st.file_uploader("Upload JSONL or JSON array", type=["jsonl", "json", "txt"])
    if uploaded is not None:
        try:
            raw_rows = _parse_upload(uploaded.getvalue().decode("utf-8", errors="ignore"))
        except Exception as exc:
            st.error(f"Failed to parse file: {exc}")
else:
    pasted = st.text_area("Paste JSONL or JSON array", height=220)
    if pasted.strip():
        try:
            raw_rows = _parse_upload(pasted)
        except Exception as exc:
            st.error(f"Failed to parse text: {exc}")

if raw_rows:
    meta_rows = []
    for i, row in enumerate(raw_rows, 1):
        session_id, tool_or_target, event_or_type = _meta_from_raw(row)
        meta_rows.append(
            {
                "row": i,
                "protocol": row.get("protocol", ""),
                "session_id": session_id,
                "tool_or_target": tool_or_target,
                "event_or_type": event_or_type,
                "log_id": row.get("log_id", ""),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    st.subheader("Input Group Preview")
    st.dataframe(meta_df.head(200), use_container_width=True, height=260)

    with st.expander("Filter specific group from uploaded logs", expanded=False):
        protocols = sorted([x for x in meta_df["protocol"].dropna().unique().tolist() if str(x).strip()])
        sessions = sorted([x for x in meta_df["session_id"].dropna().unique().tolist() if str(x).strip()])
        selected_protocols = st.multiselect("Protocol", options=protocols, default=protocols)
        selected_sessions = st.multiselect("Session IDs", options=sessions, default=[])
        row_range = st.slider("Row range", 1, len(meta_df), (1, len(meta_df)))

    filtered_df = meta_df.copy()
    if selected_protocols:
        filtered_df = filtered_df[filtered_df["protocol"].isin(selected_protocols)]
    if selected_sessions:
        filtered_df = filtered_df[filtered_df["session_id"].isin(selected_sessions)]
    filtered_df = filtered_df[(filtered_df["row"] >= row_range[0]) & (filtered_df["row"] <= row_range[1])]

    selected_rows_idx = set(filtered_df["row"].tolist())
    selected_rows = [row for i, row in enumerate(raw_rows, 1) if i in selected_rows_idx]

    st.caption(f"Selected logs for ML group analysis: {len(selected_rows)} / {len(raw_rows)}")

    if st.button("Analyze Selected Group (ML)", type="primary"):
        if not selected_rows:
            st.warning("No rows selected.")
            st.stop()

        with st.spinner("Running ML-based group analysis..."):
            processed, failures, ml_summary = _run_group_analysis_ml(selected_rows, api_key)

        if failures:
            st.warning(f"Some rows failed: {len(failures)}")
            st.dataframe(pd.DataFrame(failures).head(100), use_container_width=True)

        if not processed:
            st.error("No rows were processed successfully.")
            st.stop()

        results_df = pd.DataFrame(processed)

        st.subheader("Group-Specific Stats")
        total = len(results_df)
        anomalies = int(results_df["is_anomaly"].sum())
        anomaly_rate = (anomalies / total) if total else 0.0
        avg_score = float(results_df[results_df["is_anomaly"]]["anomaly_score"].fillna(0).mean()) if anomalies else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Logs in Group", total)
        c2.metric("Anomalies", anomalies)
        c3.metric("Anomaly Rate", f"{(anomaly_rate * 100):.2f}%")
        c4.metric("Avg Anomaly Score", f"{avg_score:.4f}")

        if isinstance(ml_summary, dict) and ml_summary:
            st.subheader("ML Session Summary")
            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric("Logs Analysed", int(ml_summary.get("total_logs_analysed", 0) or 0))
            ms2.metric("Findings", int(ml_summary.get("anomalies_found", 0) or 0))
            ms3.metric("Density", f"{float(ml_summary.get('anomaly_density_pct', 0.0) or 0.0):.2f}%")
            ms4.metric("Rule Lane Triggered", int(ml_summary.get("rule_lane_triggered_count", 0) or 0))

            assessment = str(ml_summary.get("assessment") or "")
            if assessment:
                st.caption(assessment)

            st.subheader("ML Lane Diagnostics")
            ld1, ld2 = st.columns(2)
            ld1.metric("Avg Embedding Score", f"{float(ml_summary.get('avg_embedding_lane_score', 0.0) or 0.0):.4f}")
            ld2.metric("Avg Embedding Threshold", f"{float(ml_summary.get('avg_embedding_lane_threshold', 0.0) or 0.0):.4f}")

            by_mode = ml_summary.get("by_arbitration_mode", {}) if isinstance(ml_summary.get("by_arbitration_mode"), dict) else {}
            if by_mode:
                mode_df = pd.DataFrame(
                    [{"arbitration_mode": k, "count": int(v)} for k, v in sorted(by_mode.items(), key=lambda x: x[0])]
                )
                st.plotly_chart(
                    px.bar(mode_df, x="arbitration_mode", y="count", title="Arbitration Mode Distribution"),
                    use_container_width=True,
                )

            findings_rows = []
            for _, row in results_df.iterrows():
                if not row.get("is_anomaly"):
                    continue
                findings_rows.append(
                    {
                        "log_id": row.get("log_id"),
                        "protocol": row.get("protocol"),
                        "event_or_type": row.get("event_or_type"),
                        "anomaly_type": row.get("anomaly_type"),
                        "anomaly_score": row.get("anomaly_score"),
                        "confidence": row.get("confidence"),
                        "rule_lane_triggered": row.get("rule_lane_triggered"),
                        "embedding_lane_score": row.get("embedding_lane_score"),
                        "embedding_lane_threshold": row.get("embedding_lane_threshold"),
                        "arbitration_mode": row.get("arbitration_mode"),
                    }
                )

            if findings_rows:
                st.subheader("ML Findings Detail")
                st.dataframe(pd.DataFrame(findings_rows), use_container_width=True, height=280)

        st.subheader("Anomaly Analytics")
        anomaly_df = results_df[results_df["is_anomaly"]].copy()
        if anomaly_df.empty:
            st.info("No anomalies in this selected group.")
            st.stop()

        type_dist = (
            anomaly_df.groupby("anomaly_type")
            .size()
            .reindex(ANOMALY_TYPES, fill_value=0)
            .reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(type_dist, x="anomaly_type", y="count", title="Anomaly Type Distribution (Selected Group)"),
            use_container_width=True,
        )

        st.plotly_chart(
            px.histogram(anomaly_df, x="anomaly_score", nbins=20, title="Anomaly Score Distribution"),
            use_container_width=True,
        )

        st.subheader("Correlation: Shared Causes / Reasons")
        cause_counter: Counter[str] = Counter()
        type_to_causes: dict[str, set[str]] = defaultdict(set)
        rows_for_table = []

        for _, r in anomaly_df.iterrows():
            detail = r.get("detail")
            cause_key = "unknown"
            reason = "ml_lane"
            root_log = "n/a"
            if isinstance(detail, dict):
                cause_key = _build_cause_key(detail)
                reason = str(detail.get("classification", {}).get("reasoning") or "ml_lane")
                root_log = str((detail.get("rca") or {}).get("root_cause_log_id") or "n/a")

            cause_counter[cause_key] += 1
            type_to_causes[str(r.get("anomaly_type"))].add(cause_key)
            rows_for_table.append(
                {
                    "log_id": r.get("log_id"),
                    "anomaly_type": r.get("anomaly_type"),
                    "cause_key": cause_key,
                    "root_cause_log_id": root_log,
                    "reason": reason,
                }
            )

        cause_df = pd.DataFrame(
            [{"cause_key": k, "count": v} for k, v in cause_counter.most_common(30)]
        )
        st.dataframe(cause_df, use_container_width=True, height=240)

        st.subheader("Correlation Graph")
        nodes = []
        edges = []
        for anomaly_type in sorted(type_to_causes.keys()):
            type_node = f"type::{anomaly_type}"
            nodes.append({"id": type_node, "label": anomaly_type, "color": "#ef4444"})
            for cause in sorted(type_to_causes[anomaly_type]):
                cause_node = f"cause::{cause}"
                count = cause_counter[cause]
                nodes.append({"id": cause_node, "label": f"{cause[:52]}{'...' if len(cause) > 52 else ''}", "color": "#f59e0b"})
                edges.append({"source": type_node, "target": cause_node, "color": "#fb923c", "value": count})

        html = build_graph_html(nodes, edges)
        components.html(html, height=560, scrolling=True)

        st.subheader("Detailed Group Results")
        st.dataframe(pd.DataFrame(rows_for_table), use_container_width=True, height=280)
else:
    st.info("Upload or paste logs to start ML-based group-level analysis.")
