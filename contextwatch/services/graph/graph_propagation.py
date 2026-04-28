"""Graph Propagation: Diffusion-based influence scoring.

Models failure propagation as diffusion on the service dependency graph.
Computes influence scores to distinguish true root causes from correlated effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PropagationNode:
    """Node in the propagation model with influence metrics."""

    node_id: str
    node_type: str  # Service, Database, Queue, etc.
    is_anomalous: bool = False
    influence_score: float = 0.0  # 0-1, how much influence did this node have?
    distance_to_anomaly: int = -1  # Graph distance to anomalous event
    is_root_candidate: bool = False
    propagation_score: float = 0.0  # Diffusion-based influence
    metadata: Dict = field(default_factory=dict)


class GraphPropagation:
    """Model failure propagation as diffusion on service graph.

    Algorithm:
    1. Identify all anomalous nodes
    2. Compute outgoing influence from each node (how many downstream nodes does it affect?)
    3. Score nodes based on:
       - Temporal precedence (did it happen before anomalies?)
       - Structural influence (how central is it?)
       - Independent causation (is it truly root or just early symptom?)
    4. Rank by propagation score (true roots have high outgoing, low incoming)
    """

    def __init__(self, knowledge_graph: object):
        """Initialize propagation engine with knowledge graph.

        Args:
            knowledge_graph: Neo4j-backed KnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.nodes: Dict[str, PropagationNode] = {}
        self.adjacency: Dict[str, List[Tuple[str, str]]] = {}  # node_id -> [(neighbor, edge_type)]

    def build_from_graph(self, anomaly_node_ids: List[str]) -> None:
        """Build propagation model from knowledge graph.

        Args:
            anomaly_node_ids: List of anomalous event node IDs
        """
        # Fetch all nodes related to anomalies (BFS up to 5 hops)
        visited: Set[str] = set()
        frontier: List[Tuple[str, int]] = [(nid, 0) for nid in anomaly_node_ids]
        max_hops = 5

        while frontier:
            node_id, dist = frontier.pop(0)
            if node_id in visited or dist > max_hops:
                continue
            visited.add(node_id)

            # Fetch node from graph
            node = self.kg.get_node(node_id)
            if not node:
                continue

            # Create propagation node
            prop_node = PropagationNode(
                node_id=node_id,
                node_type=node.type,
                is_anomalous=node_id in anomaly_node_ids,
                distance_to_anomaly=dist,
                metadata=node.properties,
            )
            self.nodes[node_id] = prop_node

            # Add neighbors to frontier
            neighbors = self.kg.get_neighbors(node_id)
            if neighbors:
                self.adjacency[node_id] = neighbors
                for neighbor, edge_type in neighbors:
                    if neighbor.id not in visited:
                        frontier.append((neighbor.id, dist + 1))

    def compute_influence_scores(self) -> Dict[str, float]:
        """Compute outgoing/incoming influence for each node.

        Returns:
            Dict mapping node_id → influence_score (0-1)
        """
        influence = {}

        for node_id in self.nodes:
            outgoing = len(self.adjacency.get(node_id, []))
            incoming = sum(1 for neighbors in self.adjacency.values() if any(n[0].id == node_id for n in neighbors))

            # Influence = (outgoing + 1) / (incoming + 1)
            # Higher = more likely to be root (affects many, depends on few)
            influence_score = (outgoing + 1.0) / (incoming + 1.0)
            # Normalize to 0-1 range
            influence_score = min(1.0, influence_score / (outgoing + 1.0)) if outgoing > 0 else 0.5
            influence[node_id] = float(influence_score)
            self.nodes[node_id].influence_score = influence_score

        return influence

    def compute_propagation_scores(self) -> Dict[str, float]:
        """Compute diffusion-based propagation scores.

        Uses heat diffusion: influence flows outward from anomalies.
        Nodes that would cause the observed anomalies to occur score high.

        Algorithm:
        1. Initialize heat at anomalous nodes
        2. Propagate heat backward (against edge direction) to find sources
        3. Score nodes by accumulated heat (PageRank variant)
        """
        scores = {nid: 0.0 for nid in self.nodes}

        # Initialize anomalous nodes with heat
        heat = {nid: 1.0 if self.nodes[nid].is_anomalous else 0.0 for nid in self.nodes}

        # Diffuse backward (assumption: edges go downstream)
        # Real implementation would use matrix powers or iterative diffusion
        num_iterations = 5
        damping = 0.85

        for iteration in range(num_iterations):
            new_heat = {nid: heat[nid] * (1 - damping) for nid in heat}  # Decay

            # Transmit heat backward
            for node_id, neighbors in self.adjacency.items():
                if heat.get(node_id, 0) > 0:
                    for neighbor, edge_type in neighbors:
                        # Backward propagation: if A→B (A depends on B),
                        # anomaly in B will cause issues in A
                        neighbor_id = neighbor.id
                        if neighbor_id in new_heat:
                            new_heat[neighbor_id] += heat.get(node_id, 0) * damping / max(len(neighbors), 1)

            heat = new_heat

        # Normalize scores
        max_heat = max(heat.values()) if heat else 1.0
        propagation_scores = {nid: h / max_heat if max_heat > 0 else 0.0 for nid, h in heat.items()}

        for nid in self.nodes:
            self.nodes[nid].propagation_score = propagation_scores.get(nid, 0.0)

        return propagation_scores

    def rank_root_candidates(self) -> List[str]:
        """Rank nodes by likelihood of being root cause.

        Combines:
        - Temporal precedence (negative distance_to_anomaly)
        - Structural influence (high outgoing, low incoming)
        - Propagation score (heat diffusion ranking)

        Returns:
            Sorted list of node IDs (highest rank first)
        """
        self.compute_influence_scores()
        self.compute_propagation_scores()

        # Composite score
        candidates = []
        for node_id, node in self.nodes.items():
            # Factors:
            # 1. Is it a "root" type? (Services are better candidates than individual events)
            type_weight = 1.0 if node.node_type in ("Service", "Node", "Component") else 0.5

            # 2. Temporal precedence (happened earlier)
            temporal_weight = max(0.0, 1.0 - node.distance_to_anomaly * 0.1)

            # 3. Structural influence
            structural_weight = node.influence_score

            # 4. Propagation heat
            propagation_weight = node.propagation_score

            # Composite
            score = (
                0.2 * type_weight
                + 0.2 * temporal_weight
                + 0.3 * structural_weight
                + 0.3 * propagation_weight
            )

            candidates.append((node_id, score))
            node.is_root_candidate = True

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in candidates]

    def explain_propagation(self, root_id: str, anomaly_ids: List[str]) -> Dict:
        """Explain how failure propagates from root to anomalies.

        Args:
            root_id: Root cause node ID
            anomaly_ids: List of anomalous node IDs

        Returns:
            Explanation with path and influence chain
        """
        paths = []
        for anomaly_id in anomaly_ids:
            path = self._shortest_path(root_id, anomaly_id)
            if path:
                paths.append(path)

        return {
            "root_cause_id": root_id,
            "propagation_paths": paths,
            "root_influence_score": self.nodes[root_id].influence_score if root_id in self.nodes else 0.0,
            "root_propagation_score": self.nodes[root_id].propagation_score if root_id in self.nodes else 0.0,
        }

    def _shortest_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Find shortest path from start to end using BFS."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        visited = set()
        queue = [(start_id, [start_id])]

        while queue:
            node_id, path = queue.pop(0)
            if node_id == end_id:
                return path
            if node_id in visited:
                continue
            visited.add(node_id)

            for neighbor, edge_type in self.adjacency.get(node_id, []):
                neighbor_id = neighbor.id
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_statistics(self) -> Dict:
        """Return propagation analysis statistics."""
        if not self.nodes:
            return {"status": "no_nodes"}

        influence_scores = [n.influence_score for n in self.nodes.values()]
        propagation_scores = [n.propagation_score for n in self.nodes.values()]

        return {
            "num_nodes": len(self.nodes),
            "num_anomalous": sum(1 for n in self.nodes.values() if n.is_anomalous),
            "influence_scores": {
                "min": min(influence_scores) if influence_scores else 0.0,
                "max": max(influence_scores) if influence_scores else 0.0,
                "mean": sum(influence_scores) / len(influence_scores) if influence_scores else 0.0,
            },
            "propagation_scores": {
                "min": min(propagation_scores) if propagation_scores else 0.0,
                "max": max(propagation_scores) if propagation_scores else 0.0,
                "mean": sum(propagation_scores) / len(propagation_scores) if propagation_scores else 0.0,
            },
        }
