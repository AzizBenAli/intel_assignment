
from collections.abc import Iterable

from ..graph import Graph
from .base import TransformationPass


class NodeHighlightingPass(TransformationPass):
    """Mark nodes whose names match the requested names."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = set(names)

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.name in self.names:
                node.visualization["highlight"] = {
                    "color": "red",
                    "border_width": 3,
                }

        return graph