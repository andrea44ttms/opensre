"""Core graph module for opensre.

Provides the base graph structure for building SRE automation workflows
as directed acyclic graphs (DAGs) of nodes and edges.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GraphNode:
    """Represents a single step/node in the SRE workflow graph."""

    def __init__(self, node_id: str, node_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            node_id: Unique identifier for this node.
            node_type: Type string matching a registered node_my_step handler.
            config: Optional configuration dict passed to the node at runtime.
        """
        self.node_id = node_id
        self.node_type = node_type
        self.config: Dict[str, Any] = config or {}
        self.metadata: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"GraphNode(id={self.node_id!r}, type={self.node_type!r})"


class Graph:
    """Directed acyclic graph representing an SRE automation workflow.

    Nodes are added individually; edges define execution dependencies.
    Execution order is determined via topological sort.
    """

    def __init__(self, graph_id: str):
        """
        Args:
            graph_id: Human-readable identifier for this workflow graph.
        """
        self.graph_id = graph_id
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [downstream node_ids]
        self._reverse_edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [upstream node_ids]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> "Graph":
        """Register a node in the graph. Returns self for chaining."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in graph '{self.graph_id}'")
        self._nodes[node.node_id] = node
        logger.debug("Added node %s to graph %s", node.node_id, self.graph_id)
        return self

    def add_edge(self, from_id: str, to_id: str) -> "Graph":
        """Add a directed edge from *from_id* to *to_id*. Returns self for chaining.

        Raises:
            KeyError: If either node is not registered.
            ValueError: If the edge would introduce a cycle.
        """
        for nid in (from_id, to_id):
            if nid not in self._nodes:
                raise KeyError(f"Node '{nid}' not found in graph '{self.graph_id}'")

        self._edges[from_id].append(to_id)
        self._reverse_edges[to_id].append(from_id)

        if self._has_cycle():
            # Roll back
            self._edges[from_id].remove(to_id)
            self._reverse_edges[to_id].remove(from_id)
            raise ValueError(
                f"Adding edge '{from_id}' -> '{to_id}' would introduce a cycle in graph '{self.graph_id}'"
            )

        logger.debug("Added edge %s -> %s in graph %s", from_id, to_id, self.graph_id)
        return self

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def _has_cycle(self) -> bool:
        """Return True if the graph currently contains a cycle (DFS-based)."""
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            in_stack.add(node_id)
            for neighbor in self._edges[node_id]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in in_stack:
                    return True
            in_stack.discard(node_id)
            return False

        for nid in self._nodes:
            if nid not in visited:
                if dfs(nid):
                    return True
        return False

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Return node IDs in a valid topological execution order (Kahn's algorithm).

        Raises:
            RuntimeError: If a cycle is detected (should not happen if add_edge is used).
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for downstream in self._edges[nid]:
                in_degree[downstream] += 1

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: List[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for downstream in self._edges[nid]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        if len(order) != len(self._nodes):
            raise RuntimeError(f"Cycle detected in graph '{self.graph_id}' during topological sort")

        return order

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by ID.

        Raises:
            KeyError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph '{self.graph_id}'")
        return self._nodes[node_id]

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the total number of directed edges in the graph."""
        return sum(len(neighbors) for neighbors in self._edges.values())

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.graph_id!r}, nodes={self.node_count()}, edges={self.edge_count()})"
        )
