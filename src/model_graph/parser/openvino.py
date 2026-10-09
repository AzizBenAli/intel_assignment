
from __future__ import annotations

from pathlib import Path
from typing import Any

import openvino as ov

from ..graph import Graph
from ..model import Edge, Node


class OpenVINOParser:
    """Convert an OpenVINO model into our framework-independent graph."""

    def __init__(self) -> None:
        self._core = ov.Core()

    def parse(self, model_path: str | Path) -> Graph:
        """Load an OpenVINO model from disk and convert it to our graph."""
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {path}")

        model = self._core.read_model(path)
        return self.parse_model(model)

    def parse_model(self, model: ov.Model) -> Graph:
        """Convert an already-loaded OpenVINO model into our graph."""
        graph = Graph()
        operations = list(model.get_ordered_ops())

        # OpenVINO's instance IDs identify operations independently
        # of their friendly names.
        operation_to_id = {
            operation.get_instance_id(): f"n_{index:06d}"
            for index, operation in enumerate(operations)
        }

        # First create every node so all edge endpoints will exist.
        for operation in operations:
            node_id = operation_to_id[operation.get_instance_id()]

            graph.add_node(
                Node(
                    id=node_id,
                    name=operation.get_friendly_name(),
                    op_type=operation.get_type_name(),
                    attributes=self._extract_attributes(operation),
                )
            )

        # Then translate each OpenVINO input connection into an edge.
        for operation in operations:
            destination_id = operation_to_id[
                operation.get_instance_id()
            ]

            for input_port in range(operation.get_input_size()):
                input_value = operation.input(input_port)
                source_output = input_value.get_source_output()
                source_operation = source_output.get_node()

                source_id = operation_to_id[
                    source_operation.get_instance_id()
                ]

                graph.add_edge(
                    Edge(
                        source_id=source_id,
                        source_port=source_output.get_index(),
                        destination_id=destination_id,
                        destination_port=input_port,
                        shape=self._extract_shape(source_output),
                        dtype=self._extract_dtype(source_output),
                    )
                )

        graph.validate()
        return graph

    @staticmethod
    def _extract_shape(value: Any) -> tuple[int | None, ...] | None:
        """Return known dimensions and use None for dynamic dimensions."""
        try:
            partial_shape = value.get_partial_shape()
            dimensions: list[int | None] = []

            for dimension in partial_shape:
                if dimension.is_static:
                    dimensions.append(dimension.get_length())
                else:
                    dimensions.append(None)

            return tuple(dimensions)
        except (RuntimeError, TypeError, AttributeError):
            # Includes shapes that cannot be represented by our current
            # tuple-based shape type, such as a fully dynamic rank.
            return None

    @staticmethod
    def _extract_dtype(value: Any) -> str | None:
        try:
            return value.get_element_type().get_type_name()
        except (RuntimeError, TypeError, AttributeError):
            return None

    @staticmethod
    def _extract_attributes(operation: Any) -> dict[str, Any]:
        """Extract small, useful metadata without retaining OpenVINO objects."""
        attributes: dict[str, Any] = {
            "friendly_name": operation.get_friendly_name(),
        }

        try:
            attributes["input_count"] = operation.get_input_size()
        except (RuntimeError, AttributeError):
            pass

        try:
            attributes["output_count"] = operation.get_output_size()
        except (RuntimeError, AttributeError):
            pass

        return attributes