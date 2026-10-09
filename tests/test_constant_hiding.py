
from model_graph import Edge, Graph, Node


def build_graph_with_constant() -> Graph:
    graph = Graph()

    graph.add_node(
        Node(id="input", name="input", op_type="Parameter")
    )
    graph.add_node(
        Node(id="constant", name="bias", op_type="Constant")
    )
    graph.add_node(
        Node(id="add", name="add", op_type="Add")
    )

    graph.add_edge(
        Edge(
            source_id="input",
            source_port=0,
            destination_id="add",
            destination_port=0,
            shape=(1, 4),
            dtype="f32",
        )
    )
    graph.add_edge(
        Edge(
            source_id="constant",
            source_port=0,
            destination_id="add",
            destination_port=1,
            shape=(4,),
            dtype="f32",
        )
    )

    return graph


from model_graph.transforms import ConstantHidingPass


def test_hides_constant_and_preserves_metadata() -> None:
    graph = build_graph_with_constant()

    transform = ConstantHidingPass()
    result = transform.apply(graph)

    # The Constant node should be removed.
    assert not result.has_node("constant")

    # The other nodes should remain.
    assert result.has_node("input")
    assert result.has_node("add")

    # The original input connection must remain intact.
    add_node = result.get_node("add")
    incoming = result.incoming_edges(add_node.id)

    assert len(incoming) == 1
    assert incoming[0].source_id == "input"
    assert incoming[0].destination_port == 0

    # Information about the hidden constant must be preserved.
    hidden = add_node.visualization["hidden_constants"]

    assert hidden == [
        {
            "input_port": 1,
            "dtype": "f32",
            "shape": [4],
        }
    ]

    result.validate()

def test_shared_constant_keeps_its_node_id() -> None:
    graph = build_graph_with_constant()

    graph.add_node(
        Node(id="multiply", name="multiply", op_type="Multiply")
    )

    graph.add_edge(
        Edge(
            source_id="constant",
            source_port=0,
            destination_id="multiply",
            destination_port=1,
            shape=(4,),
            dtype="f32",
        )
    )

    transform = ConstantHidingPass()
    result = transform.apply(graph)

    assert not result.has_node("constant")

    add_metadata = result.get_node("add").visualization["hidden_constants"]
    multiply_metadata = result.get_node("multiply").visualization["hidden_constants"]

    assert add_metadata[0]["shared_node_id"] == "constant"
    assert multiply_metadata[0]["shared_node_id"] == "constant"

    result.validate()

def test_preserves_unrelated_connections() -> None:
    graph = build_graph_with_constant()

    graph.add_node(
        Node(id="relu", name="relu", op_type="Relu")
    )

    graph.add_edge(
        Edge(
            source_id="add",
            source_port=0,
            destination_id="relu",
            destination_port=0,
            shape=(1, 4),
            dtype="f32",
        )
    )

    result = ConstantHidingPass().apply(graph)

    assert not result.has_node("constant")

    # The unrelated connection from Add to Relu must remain.
    outgoing = result.outgoing_edges("add")
    assert len(outgoing) == 1
    assert outgoing[0].destination_id == "relu"
    assert outgoing[0].source_port == 0
    assert outgoing[0].destination_port == 0

    result.validate()
