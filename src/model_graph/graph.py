from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from .exceptions import (
    DuplicateEdgeError,
    DuplicateNodeError,
    InvalidEdgeError,
    UnknownNodeError,
)
from .model import Edge, Node


class Graph:
    """
    Mutable framework-independent directed graph.

    The graph maintains indexes for incoming and outgoing edges so that
    connectivity queries do not require scanning the entire edge collection.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

        self._outgoing: dict[str, list[Edge]] = {}
        self._incoming: dict[str, list[Edge]] = {}

        self._edge_keys: set[tuple[str, int, str, int]] = set()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Mapping[str, Node]:
        return MappingProxyType(self._nodes)

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(
            edge
            for edges in self._outgoing.values()
            for edge in edges
        )

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        if not node.id:
            raise ValueError("node.id must not be empty")

        if node.id in self._nodes:
            raise DuplicateNodeError(
                f"Node with ID {node.id!r} already exists"
            )

        self._nodes[node.id] = node
        self._outgoing[node.id] = []
        self._incoming[node.id] = []

    def get_node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise UnknownNodeError(
                f"Unknown node ID: {node_id!r}"
            ) from None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> Node:
        node = self.get_node(node_id)

        # Remove all incident edges first.
        for edge in list(self._outgoing[node_id]):
            self.remove_edge(edge)

        for edge in list(self._incoming[node_id]):
            self.remove_edge(edge)

        del self._outgoing[node_id]
        del self._incoming[node_id]
        del self._nodes[node_id]

        return node

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_key(edge: Edge) -> tuple[str, int, str, int]:
        return (
            edge.source_id,
            edge.source_port,
            edge.destination_id,
            edge.destination_port,
        )

    def add_edge(self, edge: Edge) -> None:
        if not self.has_node(edge.source_id):
            raise UnknownNodeError(
                f"Unknown source node: {edge.source_id!r}"
            )

        if not self.has_node(edge.destination_id):
            raise UnknownNodeError(
                f"Unknown destination node: {edge.destination_id!r}"
            )

        key = self._edge_key(edge)

        if key in self._edge_keys:
            raise DuplicateEdgeError(
                "An edge already exists for "
                f"{edge.source_id}:{edge.source_port} -> "
                f"{edge.destination_id}:{edge.destination_port}"
            )

        self._edge_keys.add(key)
        self._outgoing[edge.source_id].append(edge)
        self._incoming[edge.destination_id].append(edge)

    def remove_edge(self, edge: Edge) -> None:
        key = self._edge_key(edge)

        if key not in self._edge_keys:
            raise InvalidEdgeError("Edge does not exist in this graph")

        self._edge_keys.remove(key)

        self._outgoing[edge.source_id].remove(edge)
        self._incoming[edge.destination_id].remove(edge)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def outgoing_edges(self, node_id: str) -> tuple[Edge, ...]:
        self.get_node(node_id)
        return tuple(self._outgoing[node_id])

    def incoming_edges(self, node_id: str) -> tuple[Edge, ...]:
        self.get_node(node_id)
        return tuple(self._incoming[node_id])

    def outgoing_count(self, node_id: str) -> int:
        return len(self.outgoing_edges(node_id))

    def incoming_count(self, node_id: str) -> int:
        return len(self.incoming_edges(node_id))

    def has_edges(self, node_id: str) -> bool:
        return bool(
            self._outgoing[node_id] or self._incoming[node_id]
        )

    def consumers(self, node_id: str) -> tuple[Node, ...]:
        return tuple(
            self.get_node(edge.destination_id)
            for edge in self.outgoing_edges(node_id)
        )

    def producers(self, node_id: str) -> tuple[Node, ...]:
        return tuple(
            self.get_node(edge.source_id)
            for edge in self.incoming_edges(node_id)
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate all internal graph invariants.

        Raises InvalidEdgeError if the graph is inconsistent.
        """

        expected_keys: set[tuple[str, int, str, int]] = set()

        for node_id in self._nodes:
            if node_id not in self._outgoing:
                raise InvalidEdgeError(
                    f"Missing outgoing index for node {node_id!r}"
                )

            if node_id not in self._incoming:
                raise InvalidEdgeError(
                    f"Missing incoming index for node {node_id!r}"
                )

        for source_id, edges in self._outgoing.items():
            for edge in edges:
                if edge.source_id != source_id:
                    raise InvalidEdgeError(
                        "Outgoing edge is indexed under the wrong node"
                    )

                if edge.destination_id not in self._nodes:
                    raise InvalidEdgeError(
                        f"Edge references unknown destination "
                        f"{edge.destination_id!r}"
                    )

                expected_keys.add(self._edge_key(edge))

        for destination_id, edges in self._incoming.items():
            for edge in edges:
                if edge.destination_id != destination_id:
                    raise InvalidEdgeError(
                        "Incoming edge is indexed under the wrong node"
                    )

                if edge.source_id not in self._nodes:
                    raise InvalidEdgeError(
                        f"Edge references unknown source "
                        f"{edge.source_id!r}"
                    )

                key = self._edge_key(edge)

                if key not in expected_keys:
                    raise InvalidEdgeError(
                        "Incoming and outgoing edge indexes disagree"
                    )

        if expected_keys != self._edge_keys:
            raise InvalidEdgeError(
                "Edge key index is inconsistent with graph indexes"
            )