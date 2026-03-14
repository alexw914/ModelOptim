import onnx
import onnx_graphsurgeon as gs

# 加载模型
onnx_model = onnx.load("sim_helmetcolor_cls_v0_0_0_0_int8.onnx")

# 用 onnx-graphsurgeon 导入计算图
graph = gs.import_onnx(onnx_model)

# 手动修改 opset 信息
# 设置主域 opset 为 11（"" 是主域）
onnx_model.opset_import[0].version = 11

# 如果原模型中有其他非空域，也保留其 opset 版本
for imp in onnx_model.opset_import:
    if imp.domain != "":
        imp.version = 11  # 或其他你希望的版本

# 保存修改后的模型
onnx.save(onnx_model, "opt11_model.onnx")

# 验证新模型
onnx.checker.check_model("opt11_model.onnx")
print("✅ 转换完成并验证成功，保存为 opt11_model.onnx")
