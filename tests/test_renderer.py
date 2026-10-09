from model_graph.graph import Graph
from model_graph.model import Edge, Node
from model_graph.renderer import SVGRenderer


def test_renderer_creates_svg(tmp_path):
    graph = Graph()
    graph.add_node(Node(id="input", name="input", op_type="Parameter"))
    graph.add_node(Node(id="relu", name="relu", op_type="Relu"))

    graph.add_edge(
        Edge(
            source_id="input",
            source_port=0,
            destination_id="relu",
            destination_port=0,
            shape=(1, 3, 224, 224),
            dtype="f32",
        )
    )

    output_path = tmp_path / "graph.svg"
    SVGRenderer().render(graph, output_path)

    assert output_path.exists()
    svg = output_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "input" in svg
    assert "relu" in svg


def test_renderer_highlights_nodes(tmp_path):
    graph = Graph()
    graph.add_node(
        Node(
            id="relu",
            name="relu",
            op_type="Relu",
            visualization={
                "highlight": {"color": "red", "border_width": 3}
            },
        )
    )

    output_path = tmp_path / "highlight.svg"
    SVGRenderer().render(graph, output_path)

    svg = output_path.read_text(encoding="utf-8")
    assert 'stroke="red"' in svg
    assert "relu" in svg


def test_renderer_shows_hidden_constant_metadata(tmp_path):
    graph = Graph()
    graph.add_node(
        Node(
            id="add",
            name="add",
            op_type="Add",
            visualization={
                "hidden_constants": [
                    {
                        "input_port": 1,
                        "dtype": "f32",
                        "shape": [1024, 768],
                    }
                ]
            },
        )
    )

    output_path = tmp_path / "constants.svg"
    SVGRenderer().render(graph, output_path)

    svg = output_path.read_text(encoding="utf-8")
    assert "[1]: f32[1024, 768]" in svg


def test_renderer_shows_edge_port_and_tensor_metadata(tmp_path):
    graph = Graph()
    graph.add_node(Node(id="input", name="input", op_type="Parameter"))
    graph.add_node(Node(id="relu", name="relu", op_type="Relu"))

    graph.add_edge(
        Edge(
            source_id="input",
            source_port=0,
            destination_id="relu",
            destination_port=2,
            shape=(1, 3),
            dtype="f32",
        )
    )

    output_path = tmp_path / "ports.svg"
    SVGRenderer().render(graph, output_path)

    svg = output_path.read_text(encoding="utf-8")
    assert "0 → 2" in svg
    assert "f32[1, 3]" in svg
