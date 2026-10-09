
import openvino as ov
import pytest

from model_graph import OpenVINOParser


def build_test_model() -> ov.Model:
    parameter = ov.opset13.parameter(
        [1, 4],
        ov.Type.f32,
        name="input",
    )
    constant = ov.opset13.constant(
        [0.0, 0.0, 0.0, 0.0],
        ov.Type.f32,
        name="bias",
    )
    add = ov.opset13.add(
        parameter,
        constant,
        name="add",
    )
    result = ov.opset13.result(add, name="output")

    return ov.Model([result], [parameter], "test_model")


def test_parser_builds_nodes_and_edges() -> None:
    model = build_test_model()
    graph = OpenVINOParser().parse_model(model)

    # Parameter, Constant, Add, Result
    assert len(graph) == 4
    assert len(graph.edges) == 3

    nodes_by_name = {
        node.name: node for node in graph.nodes.values()
    }

    assert nodes_by_name["input"].op_type == "Parameter"
    assert nodes_by_name["bias"].op_type == "Constant"
    assert nodes_by_name["add"].op_type == "Add"
    assert nodes_by_name["output"].op_type == "Result"

    graph.validate()


def test_parser_preserves_input_ports_and_connections() -> None:
    graph = OpenVINOParser().parse_model(build_test_model())

    nodes_by_name = {
        node.name: node for node in graph.nodes.values()
    }
    add_node = nodes_by_name["add"]

    incoming = graph.incoming_edges(add_node.id)

    assert len(incoming) == 2
    assert {edge.destination_port for edge in incoming} == {0, 1}

    source_names = {
        graph.get_node(edge.source_id).name
        for edge in incoming
    }
    assert source_names == {"input", "bias"}


def test_parser_extracts_shape_and_dtype() -> None:
    graph = OpenVINOParser().parse_model(build_test_model())

    nodes_by_name = {
        node.name: node for node in graph.nodes.values()
    }
    add_node = nodes_by_name["add"]
    incoming = graph.incoming_edges(add_node.id)

    assert all(edge.shape is not None for edge in incoming)
    assert all(edge.dtype is not None for edge in incoming)

    shapes_by_source = {
        graph.get_node(edge.source_id).name: edge.shape
        for edge in incoming
    }
    assert shapes_by_source["input"] == (1, 4)
    assert shapes_by_source["bias"] == (4,)


def test_parser_preserves_dynamic_dimensions() -> None:
    parameter = ov.opset13.parameter(
        ov.PartialShape([-1, 4]),
        ov.Type.f32,
        name="dynamic_input",
    )
    result = ov.opset13.result(parameter, name="output")
    model = ov.Model([result], [parameter], "dynamic_model")

    graph = OpenVINOParser().parse_model(model)
    nodes_by_name = {
        node.name: node for node in graph.nodes.values()
    }
    output_node = nodes_by_name["output"]

    incoming = graph.incoming_edges(output_node.id)

    assert len(incoming) == 1
    assert incoming[0].shape == (None, 4)


def test_parser_rejects_missing_model_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.xml"

    with pytest.raises(FileNotFoundError):
        OpenVINOParser().parse(missing_path)