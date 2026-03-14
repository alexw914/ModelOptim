import torch
from models import *
import utils

pth_file = "runs/checkpoint-best.pth"
output_file = "runs/best.onnx"

# MobileNetV3_Small
# net = MobileNetV3_Small(num_classes=2)
net = se_resnet_18()
checkpoint = utils.load_checkpoint(pth_file, map_location="cpu")
net.load_state_dict(checkpoint['model'])

net.to('cpu')
net.eval()

input_names = ["image"]
output_names = ["class"]
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(net, dummy_input, output_file, 
                  input_names=input_names,
                output_names=output_names,
                opset_version=12)
