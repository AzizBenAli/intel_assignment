import pytest

from model_graph import Edge, Graph, Node
from model_graph.exceptions import DuplicateEdgeError


def add_basic_nodes(graph: Graph) -> None:
    graph.add_node(Node("src", "source", "Split"))
    graph.add_node(Node("dst", "destination", "Concat"))


def test_multiple_edges_between_same_nodes_are_supported() -> None:
    graph = Graph()
    add_basic_nodes(graph)

    graph.add_edge(
        Edge(
            source_id="src",
            source_port=0,
            destination_id="dst",
            destination_port=0,
        )
    )

    graph.add_edge(
        Edge(
            source_id="src",
            source_port=1,
            destination_id="dst",
            destination_port=1,
        )
    )

    assert len(graph.edges) == 2
    assert len(graph.outgoing_edges("src")) == 2
    assert len(graph.incoming_edges("dst")) == 2


def test_multiple_edges_preserve_port_information() -> None:
    graph = Graph()
    add_basic_nodes(graph)

    edge1 = Edge("src", 0, "dst", 2)
    edge2 = Edge("src", 1, "dst", 0)

    graph.add_edge(edge1)
    graph.add_edge(edge2)

    assert edge1 in graph.outgoing_edges("src")
    assert edge2 in graph.outgoing_edges("src")

    assert edge1 in graph.incoming_edges("dst")
    assert edge2 in graph.incoming_edges("dst")


def test_exact_duplicate_port_connection_is_rejected() -> None:
    graph = Graph()
    add_basic_nodes(graph)

    edge = Edge("src", 0, "dst", 0)

    graph.add_edge(edge)

    with pytest.raises(DuplicateEdgeError):
        graph.add_edge(edge)


def test_shape_and_dtype_are_optional() -> None:
    graph = Graph()
    add_basic_nodes(graph)

    edge = Edge(
        source_id="src",
        source_port=0,
        destination_id="dst",
        destination_port=0,
        shape=(1, None, 768),
        dtype="f32",
    )

    graph.add_edge(edge)

    stored = graph.outgoing_edges("src")[0]

    assert stored.shape == (1, None, 768)
    assert stored.dtype == "f32"


def test_scalar_shape_is_supported() -> None:
    edge = Edge(
        source_id="src",
        source_port=0,
        destination_id="dst",
        destination_port=0,
        shape=(),
        dtype="i32",
    )

    assert edge.shape == ()
    assert edge.dtype == "i32"


def test_negative_ports_are_rejected() -> None:
    with pytest.raises(ValueError):
        Edge("src", -1, "dst", 0)

    with pytest.raises(ValueError):
        Edge("src", 0, "dst", -1)