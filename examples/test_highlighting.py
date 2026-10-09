from model_graph import Graph, Node
from model_graph.transforms import NodeHighlightingPass, PassPipeline

# 1. Build a small graph.
graph = Graph()
graph.add_node(Node(id="n1", name="input", op_type="Parameter"))
graph.add_node(Node(id="n2", name="add", op_type="Add"))
graph.add_node(Node(id="n3", name="output", op_type="Result"))

# 2. Configure the pipeline.
pipeline = PassPipeline()
pipeline.add(NodeHighlightingPass(names=["add"]))

# 3. Run the pipeline.
transformed_graph = pipeline.run(graph)

# 4. Inspect the result.
for node in transformed_graph.nodes.values():
    print(node.name, node.visualization)