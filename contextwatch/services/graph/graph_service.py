"""Graph service — application layer for knowledge graph operations."""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from contextwatch.services.graph.knowledge_graph import KnowledgeGraph, NodeData


class GraphService:
    """Application service for graph-based root cause analysis."""

    def __init__(self, max_depth: int = 3):
        self._graph = KnowledgeGraph(directed=True)
        self._max_depth = max_depth

    # ------------------------------------------------------------------
    # Event & anomaly ingestion
    # ------------------------------------------------------------------
    def record_event(
        self,
        event_id: str,
        event_type: str,
        session_id: str,
        agent_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> NodeData:
        """Add an event node to the graph."""
        props = properties or {}
        props.update({
            "session_id": session_id,
            "agent_id": agent_id,
        })
        return self._graph.add_node(event_id, "Event", props)

    def record_anomaly(
        self,
        anomaly_id: str,
        event_id: str,
        anomaly_type: str,
        score: float,
        confidence: float,
        properties: Optional[Dict[str, Any]] = None,
    ) -> NodeData:
        """Link an anomaly to an event."""
        anomaly_node = self._graph.add_node(anomaly_id, "Anomaly", {
            "anomaly_type": anomaly_type,
            "score": score,
            "confidence": confidence,
            **(properties or {})
        })
        self._graph.add_edge(event_id, anomaly_id, "HAS_ANOMALY", weight=1.0)
        return anomaly_node

    # ------------------------------------------------------------------
    # Root cause analysis
    # ------------------------------------------------------------------
    def find_root_cause_candidates(self, anomaly_id: str) -> List[NodeData]:
        """Find potential root causes for an anomaly (up to max_depth)."""
        return self._graph.find_related_nodes(
            anomaly_id, relation_type="CAUSED_BY", max_hops=self._max_depth
        )

    def get_causal_chain(self, start_id: str, end_id: str) -> List[NodeData]:
        """Return the shortest causal chain between two nodes."""
        path, _ = self._graph.shortest_path(start_id, end_id)
        return path

    def get_session_events(self, session_id: str) -> List[NodeData]:
        """Get all events belonging to a session."""
        return self._graph.query_by_property("session_id", session_id)

    def get_agent_events(self, agent_id: str) -> List[NodeData]:
        """Get all events executed by an agent."""
        return self._graph.query_by_property("agent_id", agent_id)

    def serialize_graph(self) -> Dict[str, Any]:
        """Serialize the entire graph for persistence or transmission."""
        return self._graph.serialize()

    def load_graph(self, data: Dict[str, Any]) -> None:
        """Replace current graph with deserialized data."""
        self._graph = KnowledgeGraph.deserialize(data)

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._graph = KnowledgeGraph(directed=True)
