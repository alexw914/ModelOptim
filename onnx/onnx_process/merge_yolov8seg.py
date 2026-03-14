import onnx
from onnx import helper
import os

input_model = "/Users/krisw/Downloads/jjx_seg_zhlh_yolov8_0_0_0_1_1280_fp32.onnx"

dirname = os.path.dirname(input_model)
basename = os.path.basename(input_model)
output_model = os.path.join(dirname, "modified." + basename)

model = onnx.load(input_model)
graph = model.graph

orig_outputs = list(graph.output)
assert len(orig_outputs) >= 7

new_output_names = ["concat_out_0", "concat_out_1", "concat_out_2"]

concat_nodes = []
for i in range(3):
    node = helper.make_node(
        "Concat",
        inputs=[orig_outputs[i].name, orig_outputs[i+3].name],
        outputs=[new_output_names[i]],
        axis=1
    )
    concat_nodes.append(node)

# 删除前6个输出，只保留第七个
for i in reversed(range(6)):
    del graph.output[i]

graph.node.extend(concat_nodes)

# 为新 concat 输出创建 ValueInfo，并设置正确 shape
for i, name in enumerate(new_output_names):
    shape1 = [dim.dim_value for dim in orig_outputs[i].type.tensor_type.shape.dim]
    shape2 = [dim.dim_value for dim in orig_outputs[i+3].type.tensor_type.shape.dim]
    
    # axis=1 拼接 C 维
    concat_shape = shape1.copy()
    concat_shape[1] = shape1[1] + shape2[1]
    
    new_vi = helper.make_tensor_value_info(
        name,
        orig_outputs[i].type.tensor_type.elem_type,
        concat_shape
    )
    graph.output.insert(-1, new_vi)

# 保留原第七个输出（已经在 graph.output 中）
# 不用重复添加

onnx.save(model, output_model)
print(f"修改后的模型已保存到 {output_model}")
