from pathlib import Path

import graphviz

from ..graph import Graph

class SVGRenderer:
    """Render a framework-independent graph as an SVG image."""

    def render(self, graph: Graph, output_path: str | Path) -> Path:
        output_path = Path(output_path)

        # Graphviz creates a top-to-bottom graph.
        diagram = graphviz.Digraph(
            name="model_graph",
            format="svg",
            graph_attr={
                "rankdir": "TB",
                "nodesep": "0.5",
                "ranksep": "0.8",
                "bgcolor": "white",
            },
            node_attr={
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "white",
                "fontname": "Arial",
                "fontsize": "10",
                "margin": "0.15",
            },
            edge_attr={
                "fontname": "Arial",
                "fontsize": "9",
                "arrowsize": "0.7",
            },
        )

        # Draw every node.
        for node in graph.nodes.values():
            label = f"{node.name}\\n({node.op_type})"

            visualization = node.visualization
            node_attrs = {}

            highlight = visualization.get("highlight")
            if highlight:
                node_attrs["color"] = highlight.get("color", "red")
                node_attrs["penwidth"] = str(
                    highlight.get("border_width", 3)
                )

            hidden_constants = visualization.get("hidden_constants", [])
            for constant in hidden_constants:
                shape = constant.get("shape")
                dtype = constant.get("dtype") or "unknown"

                if shape is None:
                    shape_text = "?"
                else:
                    shape_text = ", ".join(
                        "?" if dim is None else str(dim)
                        for dim in shape
                    )

                port = constant["input_port"]
                label += f"\\n[{port}]: {dtype}[{shape_text}]"

                shared_id = constant.get("shared_node_id")
                if shared_id:
                    label += f" (shared: {shared_id})"

            diagram.node(
                node.id,
                label=label,
                **node_attrs,
            )

        # Draw graph connections and show port numbers.
        for edge in graph.edges:
            edge_label = (
                f"{edge.source_port} → {edge.destination_port}"
            )

            if edge.dtype or edge.shape is not None:
                dtype = edge.dtype or "unknown"

                if edge.shape is None:
                    shape_text = "?"
                else:
                    shape_text = ", ".join(
                        "?" if dim is None else str(dim)
                        for dim in edge.shape
                    )

                edge_label += f"\\n{dtype}[{shape_text}]"

            diagram.edge(
                edge.source_id,
                edge.destination_id,
                label=edge_label,
            )

        # Graphviz appends ".svg", so remove an existing suffix first.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_stem = (
            output_path.with_suffix("")
            if output_path.suffix.lower() == ".svg"
            else output_path
        )

        rendered_path = Path(
            diagram.render(
                filename=str(output_stem),
                cleanup=True,
            )
        )

        return rendered_path
