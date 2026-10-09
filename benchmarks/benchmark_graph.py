from time import perf_counter

from model_graph import Edge, Graph, Node
from model_graph.transforms import (
    ConstantHidingPass,
    NodeHighlightingPass,
    PassPipeline,
)


def build_graph(node_count: int = 5000) -> Graph:
    graph = Graph()

    # Step 1: Add every node before creating any connections.
    for index in range(node_count):
        node_id = f"node_{index:05d}"
        op_type = "Constant" if index % 10 == 1 else "Relu"

        graph.add_node(
            Node(
                id=node_id,
                name=f"operation_{index}",
                op_type=op_type,
            )
        )

    # Step 2: Connect regular operations in a chain.
    previous_operation_id = None

    for index in range(node_count):
        node_id = f"node_{index:05d}"
        is_constant = index % 10 == 1

        if is_constant:
            # Constants feed the next regular operation on input port 1.
            consumer_index = index + 1
            if consumer_index < node_count:
                graph.add_edge(
                    Edge(
                        source_id=node_id,
                        source_port=0,
                        destination_id=f"node_{consumer_index:05d}",
                        destination_port=1,
                        shape=(1, 128),
                        dtype="f32",
                    )
                )
        else:
            if previous_operation_id is not None:
                graph.add_edge(
                    Edge(
                        source_id=previous_operation_id,
                        source_port=0,
                        destination_id=node_id,
                        destination_port=0,
                        shape=(1, 128),
                        dtype="f32",
                    )
                )

            previous_operation_id = node_id

    graph.validate()
    return graph


def main() -> None:
    node_count = 5000

    start = perf_counter()
    graph = build_graph(node_count)
    build_seconds = perf_counter() - start

    pipeline = PassPipeline()
    pipeline.add(NodeHighlightingPass(["operation_100"]))
    pipeline.add(ConstantHidingPass())

    start = perf_counter()
    transformed_graph = pipeline.run(graph)
    transform_seconds = perf_counter() - start

    print("Model Graph Benchmark")
    print("=====================")
    print(f"Initial nodes:       {node_count}")
    print(f"Final nodes:         {len(transformed_graph.nodes)}")
    print(f"Final edges:         {len(transformed_graph.edges)}")
    print(f"Graph build time:    {build_seconds:.4f} seconds")
    print(f"Transform time:      {transform_seconds:.4f} seconds")


if __name__ == "__main__":
    main()
