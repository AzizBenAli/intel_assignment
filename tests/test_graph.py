import pytest

from model_graph import Edge, Graph, Node
from model_graph.exceptions import (
    DuplicateNodeError,
    UnknownNodeError,
)


def make_node(node_id: str, name: str, op_type: str) -> Node:
    return Node(
        id=node_id,
        name=name,
        op_type=op_type,
    )


def test_add_and_get_nodes() -> None:
    graph = Graph()

    node = make_node("n1", "input", "Parameter")
    graph.add_node(node)

    assert len(graph) == 1
    assert graph.has_node("n1")
    assert graph.get_node("n1") is node


def test_node_ids_are_unique_but_names_do_not_have_to_be() -> None:
    graph = Graph()

    graph.add_node(make_node("n1", "duplicate", "Add"))
    graph.add_node(make_node("n2", "duplicate", "Add"))

    assert graph.get_node("n1").name == "duplicate"
    assert graph.get_node("n2").name == "duplicate"


def test_duplicate_node_id_is_rejected() -> None:
    graph = Graph()

    graph.add_node(make_node("n1", "first", "Add"))

    with pytest.raises(DuplicateNodeError):
        graph.add_node(make_node("n1", "second", "Add"))


def test_edge_connectivity() -> None:
    graph = Graph()

    graph.add_node(make_node("input", "input", "Parameter"))
    graph.add_node(make_node("add", "add", "Add"))
    graph.add_node(make_node("output", "output", "Result"))

    graph.add_edge(
        Edge(
            source_id="input",
            source_port=0,
            destination_id="add",
            destination_port=0,
        )
    )

    graph.add_edge(
        Edge(
            source_id="add",
            source_port=0,
            destination_id="output",
            destination_port=0,
        )
    )

    assert graph.outgoing_count("input") == 1
    assert graph.incoming_count("add") == 1
    assert graph.outgoing_count("add") == 1
    assert graph.incoming_count("output") == 1

    assert graph.producers("add")[0].id == "input"
    assert graph.consumers("add")[0].id == "output"


def test_unknown_nodes_are_rejected() -> None:
    graph = Graph()

    with pytest.raises(UnknownNodeError):
        graph.get_node("missing")


def test_remove_node_removes_incident_edges() -> None:
    graph = Graph()

    graph.add_node(make_node("a", "a", "Parameter"))
    graph.add_node(make_node("b", "b", "Add"))
    graph.add_node(make_node("c", "c", "Result"))

    edge_ab = Edge("a", 0, "b", 0)
    edge_bc = Edge("b", 0, "c", 0)

    graph.add_edge(edge_ab)
    graph.add_edge(edge_bc)

    graph.remove_node("b")

    assert not graph.has_node("b")
    assert graph.edges == ()
    assert graph.outgoing_count("a") == 0
    assert graph.incoming_count("c") == 0

    graph.validate()