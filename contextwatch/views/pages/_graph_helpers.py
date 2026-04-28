from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go
import streamlit as st


logger = logging.getLogger(__name__)


def _node_text(node_id: str) -> str:
    return str(node_id).split("_")[0]


def _node_value(node: dict[str, Any], key: str, default: str = "N/A") -> str:
    value = node.get(key)
    if value is None:
        value = node.get("properties", {}).get(key)
    if value is None:
        value = node.get("metadata", {}).get(key)
    return str(value) if value is not None else default


def _node_customdata(node: dict[str, Any]) -> list[str]:
    props = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
    return [
        str(node.get("id", "")),
        _node_value(node, "protocol"),
        _node_value(node, "anomaly_type"),
        _node_value(node, "score"),
        _node_value(node, "confidence"),
        str(props.get("agent_id", node.get("agent_id", "N/A"))),
        str(props.get("session_id", node.get("session_id", "N/A"))),
    ]


def render_searchable_graph(graph_data: dict[str, Any]) -> None:
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        st.info("No graph data yet. Run a batch first.")
        return

    query = st.text_input("Find anomaly id", key="graph_search_query", placeholder="Paste anomaly id or log id")
    focus_id = query.strip() if query else ""

    try:
        import networkx as nx
        graph = nx.DiGraph()
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                graph.add_node(node_id, **node)
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation", edge.get("type", "")))

        positions = nx.spring_layout(graph, seed=42)
        type_colors = {"Event": "#3b82f6", "Anomaly": "#ef4444", "Session": "#10b981", "Agent": "#f59e0b"}
        node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}

        focus_node_id = None
        if focus_id:
            for node_id, node in node_by_id.items():
                props = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
                labels = {
                    node_id,
                    str(node.get("label", "")),
                    str(props.get("log_id", "")),
                    str(node.get("log_id", "")),
                    str(props.get("anomaly_id", "")),
                }
                if focus_id in labels:
                    focus_node_id = node_id
                    break

        edge_x, edge_y = [], []
        for source, target in graph.edges():
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#6b7280"), hoverinfo="none", showlegend=False))

        for node_type, color in type_colors.items():
            filtered = [node_id for node_id, data in graph.nodes(data=True) if data.get("type") == node_type]
            if not filtered:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[positions[node_id][0] for node_id in filtered],
                    y=[positions[node_id][1] for node_id in filtered],
                    mode="markers+text",
                    name=node_type,
                    text=[_node_text(node_id) for node_id in filtered],
                    textposition="top center",
                    customdata=[_node_customdata(node_by_id.get(node_id, {"id": node_id})) for node_id in filtered],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Protocol: %{customdata[1]}<br>"
                        "Anomaly type: %{customdata[2]}<br>"
                        "Score: %{customdata[3]}<br>"
                        "Confidence: %{customdata[4]}<br>"
                        "<extra></extra>"
                    ),
                    marker=dict(size=18 if node_type == "Anomaly" else 14, color=color, line=dict(width=2, color="white")),
                )
            )

        if focus_node_id and focus_node_id in positions:
            fx, fy = positions[focus_node_id]
            fig.add_trace(
                go.Scatter(
                    x=[fx],
                    y=[fy],
                    mode="markers+text",
                    name="Focused anomaly",
                    text=[_node_text(focus_node_id)],
                    textposition="top center",
                    customdata=[_node_customdata(node_by_id.get(focus_node_id, {"id": focus_node_id}))],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Protocol: %{customdata[1]}<br>"
                        "Anomaly type: %{customdata[2]}<br>"
                        "Score: %{customdata[3]}<br>"
                        "Confidence: %{customdata[4]}<br>"
                        "<extra></extra>"
                    ),
                    marker=dict(size=24, color="#f97316", line=dict(width=3, color="#ffffff")),
                )
            )

        layout_kwargs: dict[str, Any] = dict(
            height=580,
            showlegend=True,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        if focus_node_id and focus_node_id in positions:
            fx, fy = positions[focus_node_id]
            layout_kwargs["xaxis"]["range"] = [fx - 0.35, fx + 0.35]
            layout_kwargs["yaxis"]["range"] = [fy - 0.35, fy + 0.35]
        fig.update_layout(**layout_kwargs)
        st.plotly_chart(fig, use_container_width=True)
        if focus_id and not focus_node_id:
            st.info(f'No anomaly graph node matched "{focus_id}".')
    except (ImportError, ValueError, RuntimeError, TypeError) as exc:
        logger.error("Graph render failed", exc_info=True)
        st.error(f"Graph render error: {exc}")
