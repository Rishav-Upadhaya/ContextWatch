from __future__ import annotations

from pyvis.network import Network


def build_graph_html(nodes: list[dict], edges: list[dict]) -> str:
    net = Network(height="550px", width="100%", directed=True)
    for node in nodes:
        net.add_node(node["id"], label=node.get("label", node["id"]), color=node.get("color", "#6baed6"))
    for edge in edges:
        net.add_edge(edge["source"], edge["target"], color=edge.get("color", "#999"))
    return net.generate_html(notebook=False)
