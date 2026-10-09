
from __future__ import annotations

from abc import ABC, abstractmethod

from model_graph import Graph


class TransformationPass(ABC):
    """Base interface for graph transformation passes."""

    @property
    def name(self) -> str:
        """Return the pass name used for configuration."""
        return type(self).__name__

    @abstractmethod
    def apply(self, graph: Graph) -> Graph:
        """Transform a graph and return the resulting graph."""
        raise NotImplementedError