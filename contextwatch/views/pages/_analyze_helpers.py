from __future__ import annotations

import json
import logging
import os
from typing import Optional

import plotly.graph_objects as go
import requests
import streamlit as st


logger = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
BATCH_TIMEOUT = int(os.environ.get("BATCH_READ_TIMEOUT_SECONDS", "300"))
CHUNK_SIZE = int(os.environ.get("BATCH_CHUNK_SIZE", "10"))


def parse_logs(raw: str) -> list[dict]:
    text = raw.strip()
    if not text:
        raise ValueError("Input is empty.")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if "logs" in parsed:
                return parsed["logs"]
            return [parsed]
        raise ValueError("JSON must be an array or object.")
    except json.JSONDecodeError:
        # Fall back to line-by-line JSONL parsing when full JSON parsing fails.
        parsed = None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = []
    for i, line in enumerate(lines, 1):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {i}: {e.msg}")
    if not result:
        raise ValueError("No valid JSON found.")
    return result


def ingest_batch_api(logs: list[dict], include_rca: bool) -> tuple[Optional[dict], Optional[str]]:
    """Chunk-upload logs and aggregate results."""
    if not logs:
        return {"batch_size": 0, "anomaly_count": 0, "normal_count": 0, "results": [], "graph": {"nodes": [], "edges": []}}, None

    all_results, anomaly_count, normal_count = [], 0, 0
    graph: dict = {"nodes": [], "edges": []}
    total_chunks = (len(logs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    bar = st.progress(0.0)
    status = st.empty()

    for idx, start in enumerate(range(0, len(logs), CHUNK_SIZE), 1):
        chunk = logs[start:start + CHUNK_SIZE]
        status.text(f"Processing chunk {idx}/{total_chunks} ({len(chunk)} logs)…")
        try:
            r = requests.post(
                f"{API_URL}/ingest/batch",
                json={"logs": chunk, "include_rca": include_rca},
                timeout=(10, BATCH_TIMEOUT),
            )
        except requests.RequestException as e:
            logger.error("Chunk %d upload failed", idx, exc_info=True)
            bar.empty()
            status.empty()
            return None, f"Chunk {idx} failed: {e}"

        if r.status_code != 200:
            bar.empty()
            status.empty()
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                logger.error("Failed to parse chunk %d error response JSON", idx, exc_info=True)
                detail = r.text
            return None, f"Chunk {idx} HTTP {r.status_code}: {detail}"

        data = r.json()
        all_results.extend(data.get("results", []))
        anomaly_count += int(data.get("anomaly_count", 0))
        normal_count += int(data.get("normal_count", 0))
        graph = data.get("graph", graph)
        bar.progress(min(1.0, idx / total_chunks))

    bar.empty()
    status.empty()
    return {
        "batch_size": len(logs),
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "results": all_results,
        "graph": graph,
    }, None


def render_graph(graph_data: dict) -> None:
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        st.info("No graph data yet. Run a batch first.")
        return

    try:
        import networkx as nx
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"], type=n.get("type", "Unknown"), **n.get("properties", {}))
        for e in edges:
            G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))

        pos = nx.spring_layout(G, seed=42)
        colors = {"Event": "#3b82f6", "Anomaly": "#ef4444", "Session": "#10b981", "Agent": "#f59e0b"}

        edge_x, edge_y = [], []
        for s, t in G.edges():
            x0, y0 = pos[s]
            x1, y1 = pos[t]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(width=1, color="#6b7280"), hoverinfo="none", showlegend=False))

        for ntype, color in colors.items():
            nids = [nid for nid, d in G.nodes(data=True) if d.get("type") == ntype]
            if not nids:
                continue
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in nids], y=[pos[n][1] for n in nids],
                mode="markers+text", name=ntype, text=[n.split("_")[0] for n in nids],
                textposition="top center",
                marker=dict(size=16, color=color, line=dict(width=2, color="white")),
            ))

        fig.update_layout(
            height=550, showlegend=True, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="white"), margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    except (ImportError, ValueError, RuntimeError, TypeError) as exc:
        logger.error("Graph render failed", exc_info=True)
        st.error(f"Graph render error: {exc}")


def severity(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.65:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def render_anomaly_details(anom_df) -> None:
    for _, row in anom_df.iterrows():
        with st.expander(f"{row['log_id']} — {row.get('anomaly_type', 'N/A')}"):
            st.code(str(row["log_id"]), language=None)
            left, right = st.columns(2)
            left.json(
                {
                    "anomaly_score": row.get("anomaly_score"),
                    "cosine_distance": row.get("cosine_distance"),
                    "confidence": row.get("confidence"),
                }
            )
            right.write(row.get("explanation", "No explanation"))

            marca = row.get("marca_trace")
            if isinstance(marca, dict) and marca:
                m1, m2, m3 = st.columns(3)
                m1.metric("Root cause", marca.get("root_cause", "N/A"))
                m2.metric("Component", marca.get("component", "N/A"))
                m3.metric("Severity", marca.get("severity", "N/A"))
                if marca.get("action"):
                    st.info(f"Suggested action: {marca.get('action')}")
                with st.expander("Raw MA-RCA JSON", expanded=False):
                    st.json(marca)

            judge_verdict = row.get("judge_verdict")
            if isinstance(judge_verdict, dict) and judge_verdict:
                j1, j2, j3 = st.columns(3)
                j1.metric("Verdict", judge_verdict.get("verdict", "N/A"))
                j2.metric("Confidence", judge_verdict.get("confidence", "N/A"))
                j3.metric("Severity", judge_verdict.get("severity", "N/A"))
                if judge_verdict.get("action"):
                    st.info(f"Suggested action: {judge_verdict.get('action')}")
                with st.expander("Raw judge verdict", expanded=False):
                    st.json(judge_verdict)
