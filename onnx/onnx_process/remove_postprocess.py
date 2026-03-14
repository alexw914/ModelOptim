import onnx
from onnx import helper, TensorProto

model_path = "model.onnx"
model = onnx.load(model_path)
graph = model.graph

# node want be removed
target_nodes = {"/baseModel/head_module/Reshape", "/baseModel/head_module/Reshape_2", "/baseModel/head_module/Reshape_4",
                "/Transpose", "/Transpose_1", "/Transpose_2"}

affected_nodes = set(target_nodes)
known_outputs = set()

for node in graph.node:
    if any(inp in known_outputs for inp in node.input) or node.name in affected_nodes:
        affected_nodes.add(node.name)
        known_outputs.update(node.output)

new_nodes = [node for node in graph.node if node.name not in affected_nodes]
graph.ClearField("node") 
graph.node.extend(new_nodes)

new_outputs = [output for output in graph.output if output.name not in known_outputs]
graph.ClearField("output") 
graph.output.extend(new_outputs) 

# 3 concat node
concat_nodes = [
    helper.make_node("Concat", inputs=["/baseModel/head_module/reg_preds.0/reg_preds.0.2/Conv_output_0", "/baseModel/head_module/cls_preds.0/cls_preds.0.2/Conv_output_0"], outputs=["concat_output_1"], axis=1),
    helper.make_node("Concat", inputs=["/baseModel/head_module/reg_preds.1/reg_preds.1.2/Conv_output_0", "/baseModel/head_module/cls_preds.1/cls_preds.1.2/Conv_output_0"], outputs=["concat_output_2"], axis=1),
    helper.make_node("Concat", inputs=["/baseModel/head_module/reg_preds.2/reg_preds.2.2/Conv_output_0", "/baseModel/head_module/cls_preds.2/cls_preds.2.2/Conv_output_0"], outputs=["concat_output_3"], axis=1),
]
graph.node.extend(concat_nodes)

# 3 output
add_outputs = [
    helper.make_tensor_value_info("concat_output_1", TensorProto.FLOAT, [1, 66, 128, 128]),
    helper.make_tensor_value_info("concat_output_2", TensorProto.FLOAT, [1, 66, 64, 64]),
    helper.make_tensor_value_info("concat_output_3", TensorProto.FLOAT, [1, 66, 32, 32]),
]
graph.output.extend(add_outputs)

onnx.save(model, "modified_model.onnx")
print("Generate modified_model.onnx success!")
