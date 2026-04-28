"""contextwatch.services.graph.knowledge_graph
==============================================
In-memory knowledge graph for anomaly context and causal traversal.

Stores typed nodes and weighted edges, then exposes BFS/DFS/path queries
used by RCA and graph propagation services.

Used by: contextwatch.services.reasoning.mar_cra
Depends on: dataclasses, typing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Optional, Tuple



@dataclass
class NodeData:
    """Data stored for a graph node."""
    id: str
    type: str  # Event, Session, Agent, Tool, Anomaly
    properties: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'NodeData':
        return NodeData(
            id=self.id,
            type=self.type,
            properties=self.properties.copy()
        )


@dataclass
class EdgeData:
    """Data stored for a graph edge."""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def copy(self) -> 'EdgeData':
        return EdgeData(
            source=self.source,
            target=self.target,
            relation=self.relation,
            properties=self.properties.copy(),
            weight=self.weight
        )


class KnowledgeGraph:
    """Represents log entities and relationships as a typed graph.

    Attributes:
        directed: Whether traversal follows directed edge semantics.
        _nodes: Node registry keyed by node ID.
        _edges: Flat edge list preserving insertion order.
        _outgoing: Outgoing adjacency index for fast traversal.
        _incoming: Incoming adjacency index for reverse traversal.
    """

    def __init__(self, directed: bool = True):
        """Initialize the knowledge graph.

        Parameters:
        -----------
        directed: bool
            Whether the graph is directed (True) or undirected (False)
        """
        self.directed = directed
        self._nodes: Dict[str, NodeData] = {}
        self._edges: List[EdgeData] = []
        self._outgoing: Dict[str, List[str]] = {}  # node_id -> list of target_ids
        self._incoming: Dict[str, List[str]] = {}  # node_id -> list of source_ids

    def add_node(self, node_id: str, node_type: str, properties: Optional[Dict[str, Any]] = None) -> NodeData:
        """Add or update a node in the graph.

        Parameters:
        -----------
        node_id: str
            Unique identifier for the node
        node_type: str
            Type of node (Event, Session, Agent, Tool, Anomaly)
        properties: dict, optional
            Additional node attributes

        Returns:
        --------
        The created/updated NodeData instance
        """
        if node_id not in self._nodes:
            node = NodeData(id=node_id, type=node_type, properties=properties or {})
            self._nodes[node_id] = node
            self._outgoing[node_id] = []
            self._incoming[node_id] = []
        else:
            # Update existing node
            if properties:
                self._nodes[node_id].properties.update(properties)
            node = self._nodes[node_id]
        return node

    def add_edge(self, source_id: str, target_id: str, relation: str,
                 weight: float = 1.0, properties: Optional[Dict[str, Any]] = None) -> EdgeData:
        """Add an edge between two nodes.

        Parameters:
        -----------
        source_id: str
            ID of source node
        target_id: str
            ID of target node
        relation: str
            Type of relationship (e.g., PART_OF, USED_TOOL, HAS_ANOMALY)
        weight: float
            Edge weight for pathfinding algorithms
        properties: dict, optional
            Additional edge attributes

        Returns:
        --------
        The created EdgeData instance
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            raise ValueError("Both source and target nodes must exist in the graph")

        edge = EdgeData(
            source=source_id,
            target=target_id,
            relation=relation,
            weight=weight,
            properties=properties or {}
        )
        self._edges.append(edge)

        # Update adjacency lists
        self._outgoing[source_id].append(target_id)
        self._incoming[target_id].append(source_id)

        # For undirected graphs, also add the reverse direction
        if not self.directed:
            self._incoming[source_id].append(target_id)
            self._outgoing[target_id].append(source_id)

        return edge

    def get_node(self, node_id: str) -> Optional[NodeData]:
        """Get node data by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeData]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_neighbors(self, node_id: str, outgoing: bool = True) -> List[Tuple[NodeData, str, float]]:
        """Get neighboring nodes with edge information.

        Parameters:
        -----------
        node_id: str
            The node to get neighbors for
        outgoing: bool
            If True, get outgoing neighbors; if False, get incoming neighbors

        Returns:
        --------
        List of tuples: (neighbor_node_data, relation, weight)
        """
        neighbors = []
        adj_list = self._outgoing[node_id] if outgoing else self._incoming[node_id]

        for neighbor_id in adj_list:
            # Find the edge
            for edge in self._edges:
                if outgoing and edge.source == node_id and edge.target == neighbor_id:
                    neighbors.append((self._nodes[neighbor_id], edge.relation, edge.weight))
                    break
                if not outgoing and edge.target == node_id and edge.source == neighbor_id:
                    neighbors.append((self._nodes[neighbor_id], edge.relation, edge.weight))
                    break
                if not self.directed and edge.source == neighbor_id and edge.target == node_id:
                    neighbors.append((self._nodes[neighbor_id], edge.relation, edge.weight))
                    break

        return neighbors

    def breadth_first_search(self, start_id: str, max_depth: int = 10) -> List[List[NodeData]]:
        """Perform BFS traversal from a starting node.

        Parameters:
        -----------
        start_id: str
            Starting node ID
        max_depth: int
            Maximum traversal depth

        Returns:
        --------
        List of levels, where each level is a list of nodes at that distance
        """
        if start_id not in self._nodes:
            return []

        visited: Set[str] = {start_id}
        levels: List[List[NodeData]] = [[self._nodes[start_id]]]

        current_level = [start_id]
        for depth in range(max_depth):
            next_level = []
            for node_id in current_level:
                for neighbor_data in self.get_neighbors(node_id, outgoing=True):
                    neighbor_id = neighbor_data[0].id
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.append(neighbor_id)

            if not next_level:
                break

            current_level = next_level
            levels.append([self._nodes[nid] for nid in current_level])

        return levels

    def depth_first_search(self, start_id: str, max_depth: int = 10) -> List[NodeData]:
        """Perform DFS traversal from a starting node.

        Returns:
        --------
        List of visited nodes in DFS order
        """
        if start_id not in self._nodes:
            return []

        visited: List[NodeData] = []
        visited_set: Set[str] = set()

        def dfs(node_id: str, depth: int):
            if depth > max_depth or node_id in visited_set:
                return
            visited_set.add(node_id)
            visited.append(self._nodes[node_id])
            for neighbor_data in self.get_neighbors(node_id, outgoing=True):
                neighbor_id = neighbor_data[0].id
                dfs(neighbor_id, depth + 1)

        dfs(start_id, 0)
        return visited

    def shortest_path(self, start_id: str, end_id: str) -> Tuple[List[NodeData], float]:
        """Find the shortest path between two nodes using Dijkstra's algorithm.

        Parameters:
        -----------
        start_id: str
            Source node ID
        end_id: str
            Destination node ID

        Returns:
        --------
        Tuple of (path_nodes, total_weight) or ([], float('inf')) if no path exists
        """
        if start_id not in self._nodes or end_id not in self._nodes:
            return ([], float('inf'))

        # Dijkstra's algorithm
        distances: Dict[str, float] = {node_id: float('inf') for node_id in self._nodes}
        predecessors: Dict[str, Optional[str]] = {node_id: None for node_id in self._nodes}
        distances[start_id] = 0.0

        # Simple priority queue implementation (min-priority queue)
        unvisited = set(self._nodes.keys())

        while unvisited:
            # Find unvisited node with minimum distance
            current = None
            min_distance = float('inf')
            for node_id in unvisited:
                if distances[node_id] < min_distance:
                    min_distance = distances[node_id]
                    current = node_id

            if current is None or distances[current] == float('inf'):
                break

            unvisited.remove(current)

            if current == end_id:
                break

            # Update distances to neighbors
            for neighbor_data in self.get_neighbors(current, outgoing=True):
                neighbor_id = neighbor_data[0].id
                edge_weight = neighbor_data[2]
                new_dist = distances[current] + edge_weight
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = current

        # Reconstruct path
        if distances[end_id] == float('inf'):
            return ([], float('inf'))

        path_nodes: List[NodeData] = []
        current = end_id
        while current is not None:
            path_nodes.insert(0, self._nodes[current])
            current = predecessors[current]

        return (path_nodes, distances[end_id])

    def find_related_nodes(self, node_id: str, relation_type: Optional[str] = None,
                          max_hops: int = 3) -> List[NodeData]:
        """Find nodes related to a given node within a hop limit.

        This is useful for root-cause analysis where we want to explore
        the causal chain up to a limited depth.

        Parameters:
        -----------
        node_id: str
            Starting node (usually an anomaly)
        relation_type: str, optional
            Filter by specific relationship type
        max_hops: int
            Maximum number of hops to traverse

        Returns:
        --------
        List of related nodes within the specified hop radius
        """
        if node_id not in self._nodes:
            return []

        visited: Set[str] = {node_id}
        frontier: List[Tuple[str, int]] = [(node_id, 0)]
        related: List[NodeData] = []

        while frontier:
            current_id, depth = frontier.pop(0)

            if depth >= max_hops:
                continue

            for neighbor_data in self.get_neighbors(current_id, outgoing=True):
                neighbor_id = neighbor_data[0].id
                neighbor_relation = neighbor_data[1]

                # Filter by relation type if specified
                if relation_type and neighbor_relation != relation_type:
                    continue

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    related.append(self._nodes[neighbor_id])
                    frontier.append((neighbor_id, depth + 1))

        return related

    def query_by_type(self, node_type: str) -> List[NodeData]:
        """Query all nodes of a specific type."""
        return [node for node in self._nodes.values() if node.type == node_type]

    def query_by_property(self, property_name: str, property_value: Any) -> List[NodeData]:
        """Query nodes by property value."""
        return [node for node in self._nodes.values()
                if node.properties.get(property_name) == property_value]

    def serialize(self) -> Dict[str, Any]:
        """Serialize graph to a dictionary for persistence."""
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "properties": n.properties}
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "weight": e.weight,
                    "properties": e.properties
                }
                for e in self._edges
            ]
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'KnowledgeGraph':
        """Create a graph from serialized data."""
        graph = cls(directed=True)
        for node_data in data["nodes"]:
            graph.add_node(
                node_id=node_data["id"],
                node_type=node_data["type"],
                properties=node_data["properties"]
            )
        for edge_data in data["edges"]:
            graph.add_edge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                relation=edge_data["relation"],
                weight=edge_data["weight"],
                properties=edge_data.get("properties")
            )
        return graph

"""End of KnowledgeGraph module."""
