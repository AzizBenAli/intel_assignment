from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

# Create an input tensor with shape [1, 4].
input_node = ops.parameter(
    [1, 4],
    dtype=ov.Type.f32,
    name="input",
)

# Create a constant tensor with four fixed values.
constant_node = ops.constant(
    np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    name="bias",
)

# Add the input and constant together.
add_node = ops.add(
    input_node,
    constant_node,
    name="add",
)

# Assemble the operations into an OpenVINO model.
model = ov.Model(
    [add_node],
    [input_node],
    "tiny_test_model",
)

# Save the model as an XML file and its associated binary weights.
xml_path = output_dir / "tiny_test_model.xml"
bin_path = output_dir / "tiny_test_model.bin"
ov.serialize(model, str(xml_path), str(bin_path))

print(f"Model created: {xml_path}")
print(f"Weights created: {bin_path}")
