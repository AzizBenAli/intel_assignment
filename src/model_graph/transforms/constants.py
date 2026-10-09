from ..graph import Graph
from .base import TransformationPass


class ConstantHidingPass(TransformationPass):
    """Hide Constant nodes while preserving their metadata."""

    def apply(self, graph: Graph) -> Graph:
        constant_ids = [
            node.id
            for node in graph.nodes.values()
            if node.op_type == "Constant"
        ]

        for constant_id in constant_ids:
            constant = graph.get_node(constant_id)
            consumers = graph.outgoing_edges(constant_id)

            # Save the Constant's information on each consumer.
            is_shared = len(consumers) > 1

            for edge in consumers:
                destination = graph.get_node(edge.destination_id)

                metadata = {
                    "input_port": edge.destination_port,
                    "dtype": edge.dtype,
                    "shape": (
                        list(edge.shape)
                        if edge.shape is not None
                        else None
                    ),
                }

                if is_shared:
                    metadata["shared_node_id"] = constant.id

                hidden = destination.visualization.setdefault(
                    "hidden_constants", []
                )

                if metadata not in hidden:
                    hidden.append(metadata)

                # Remove only the Constant-to-consumer edge
                # after saving its information.
                graph.remove_edge(edge)

            # Remove the Constant only if no edges remain.
            if not graph.has_edges(constant_id):
                graph.remove_node(constant_id)

        graph.validate()
        return graph
