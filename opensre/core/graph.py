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
            raise ValueError(f"Adding edge {from_id!r} -> {to_id!r} would create a cycle")

        logger.debug("Added edge %s -> %s in graph %s", from_id, to_id, self.graph_id)
        return self

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def topological_order(self) -> List[GraphNode]:
        """Return nodes in a valid topological execution order (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for downstream in self._edges[nid]:
                in_degree[downstream] += 1

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: List[GraphNode] = []

        while queue:
            current = queue.popleft()
            order.append(self._nodes[current])
            for downstream in self._edges[current]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        if len(order) != len(self._nodes):
            raise RuntimeError("Graph contains a cycle; topological sort failed.")

        return order

    def get_node(self, node_id: str) -> GraphNode:
        """Retrieve a node by its ID."""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"Node '{node_id}' not found in graph '{self.graph_id}'") from None

    @property
    def nodes(self) -> List[GraphNode]:
        """All registered nodes (insertion order)."""
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbour in self._edges[node]:
                if neighbour not in visited:
                    if dfs(neighbour):
                        return True
                elif neighbour in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        return any(dfs(n) for n in self._nodes if n not in visited)

    def __repr__(self) -> str:
        return f"Graph(id={self.graph_id!r}, nodes={len(self._nodes)}, edges={sum(len(v) for v in self._edges.values())})"
