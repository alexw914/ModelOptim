import onnx
import onnx_graphsurgeon as gs

# 1. 加载模型
onnx_path = "v0015_obbczs_detection_qdq_int8_reduce_input.onnx"
model = onnx.load(onnx_path)
graph = gs.import_onnx(model)

# 2. 目标 Concat 节点名称列表
target_concat_names = [
    "/model.22/Concat",
    "/model.22/Concat_1",
    "/model.22/Concat_2"
]

# 3. 遍历所有节点
modified_count = 0
for node in graph.nodes:
    if node.name in target_concat_names and node.op == "Concat":
        print(f"\n✅ Found target Concat node: {node.name}")
        print("🔹 Original inputs:", [inp.name for inp in node.inputs])

        # 4. 修改拼接顺序（示例：输入[0,2,1]）
        if len(node.inputs) >= 3:
            node.inputs = [node.inputs[0], node.inputs[2], node.inputs[1]]
        else:
            print("⚠️ Warning: node has fewer than 3 inputs, skipping reorder.")
            continue

        print("✅ Modified inputs:", [inp.name for inp in node.inputs])
        modified_count += 1

# 5. 检查修改数量
print(f"\n🎯 Total Concat nodes modified: {modified_count}")

# 6. 导出修改后的模型
graph.cleanup()
onnx.save(gs.export_onnx(graph), "modified." + onnx_path)
print("✅ Model saved to modified.onnx")

