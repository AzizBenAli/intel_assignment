
from model_graph import Graph, Node
from model_graph.transforms import NodeHighlightingPass


def test_highlighting_marks_matching_node() -> None:
    graph = Graph()

    graph.add_node(Node(id="n1", name="input", op_type="Parameter"))
    graph.add_node(Node(id="n2", name="add", op_type="Add"))
    graph.add_node(Node(id="n3", name="output", op_type="Result"))

    transform = NodeHighlightingPass(names=["add"])
    result = transform.apply(graph)

    assert result is graph
    assert graph.get_node("n2").visualization["highlight"] == {
        "color": "red",
        "border_width": 3,
    }
    assert "highlight" not in graph.get_node("n1").visualization
    assert "highlight" not in graph.get_node("n3").visualization


def test_highlighting_supports_multiple_names() -> None:
    graph = Graph()
    graph.add_node(Node(id="n1", name="add", op_type="Add"))
    graph.add_node(Node(id="n2", name="relu", op_type="Relu"))
    graph.add_node(Node(id="n3", name="output", op_type="Result"))

    NodeHighlightingPass(names=["add", "relu"]).apply(graph)

    assert "highlight" in graph.get_node("n1").visualization
    assert "highlight" in graph.get_node("n2").visualization
    assert "highlight" not in graph.get_node("n3").visualization