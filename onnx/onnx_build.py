import onnx
from onnx import helper, TensorProto

input_name = "input"
output_name = "output"

# Input shape: 1x300x80
input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 300, 80])
output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 300, 80])

# Unsqueeze on axis 0 (becomes 1x1x300x80)
unsq = helper.make_node(
    "Unsqueeze",
    inputs=[input_name],
    outputs=["unsq_out"],
    axes=[0],  # opset <13 uses attr; keep simple
    name="Unsqueeze_0",
)

# Squeeze same axis to return to 1x300x80
sq = helper.make_node(
    "Squeeze",
    inputs=["unsq_out"],
    outputs=["sq_out"],
    axes=[0],
    name="Squeeze_0",
)

sigmoid = helper.make_node(
    "Sigmoid",
    inputs=["sq_out"],
    outputs=[output_name],
    name="Sigmoid_0",
)

graph = helper.make_graph(
    nodes=[unsq, sq, sigmoid],
    name="UnsqSqSigmoidGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
onnx.checker.check_model(model)
onnx.save(model, "unsq_sq_sigmoid.onnx")
print("Saved: unsq_sq_sigmoid.onnx")