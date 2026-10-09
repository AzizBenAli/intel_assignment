from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TensorShape = tuple[int | None, ...]


@dataclass
class Node:
    """
    Framework-independent representation of an operation/node.

    `id` is the stable identity of the node inside the graph.
    `name` is the original model/framework name and does not need to be unique.
    """

    id: str
    name: str
    op_type: str

    attributes: dict[str, Any] = field(default_factory=dict)
    visualization: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    """
    Connection between an output port of one node and an input port of another.
    """

    source_id: str
    source_port: int

    destination_id: str
    destination_port: int

    shape: TensorShape | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.source_port < 0:
            raise ValueError("source_port must be >= 0")

        if self.destination_port < 0:
            raise ValueError("destination_port must be >= 0")