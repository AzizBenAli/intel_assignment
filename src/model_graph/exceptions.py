class GraphError(Exception):
    """Base exception for graph-related errors."""


class DuplicateNodeError(GraphError):
    """Raised when a node ID already exists."""


class UnknownNodeError(GraphError):
    """Raised when a referenced node does not exist."""


class DuplicateEdgeError(GraphError):
    """Raised when the same port-to-port connection already exists."""


class InvalidEdgeError(GraphError):
    """Raised when an edge violates graph invariants."""