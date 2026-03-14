import torch
import torch.onnx
import torchvision.models as models
from model import MobileNetV2Output

state_dict = torch.load("best_val.pt", map_location='cpu')
model = MobileNetV2Output()
model.load_state_dict(state_dict)
model.eval()

# 创建一个虚拟输入
dummy_input = torch.randn(1, 3, 448, 448, device='cpu')

# 定义输入和输出的名称
input_names = ["input"]
output_names = ["output"]

# 导出模型
torch.onnx.export(model, dummy_input, "project_anjian_nangongcheng_belt_cls_v0_1_0_7.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11, do_constant_folding=True, dynamic_axes=None)

