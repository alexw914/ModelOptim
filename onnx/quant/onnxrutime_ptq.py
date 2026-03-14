import os, random
import cv2
import numpy as np
import onnx, onnxsim
import onnxruntime as ort
from onnxruntime.quantization import quantize_static,quantize_dynamic, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod

def get_input_shape_from_model(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    return input_shape[2], input_shape[3]

def get_ort_session(model_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    used_providers = [p for p in providers if p in available]
    print(f"使用推理设备: {used_providers[0]}")
    return ort.InferenceSession(model_path, providers=used_providers)

def letterbox(img, new_shape=(224, 224), color=(114, 114, 114)):
    shape = img.shape[:2]  # 当前图像高度、宽度
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

def preprocess(image_path, input_shape=(224, 224), mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
    input_img, r, dw, dh = letterbox(img, input_shape)
    if to_rgb:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    input_img = (input_img - mean) / std
    input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
    input_img = np.expand_dims(input_img, axis=0)  # 增加 batch 维度
    return input_img, r, dw, dh, img

# ===============================
# 创建 ORT 校准数据集生成器
# ===============================
class MyDataReader(CalibrationDataReader):
    def __init__(self, input_name, data_dir, input_size=(1280,1280), num_samples=100):
        self.input_name = input_name
        self.input_size = input_size
        self.image_files = [os.path.join(data_dir, f)
                            for f in os.listdir(data_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
        random.seed(1000)
        random.shuffle(self.image_files)
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            # 注意这里取 preprocess 返回的第一个值 input_img
            self.enum_data = iter([{self.input_name: preprocess(p, self.input_size)[0]}
                                   for p in self.image_files])
        return next(self.enum_data, None)

# ===============================
# 静态量化
# ===============================
fp32_model_path = 'personhead/v1.1.50.yolov8.person.head.fp32_12801280.onnx'
int8_model_path = 'personhead/v1.1.50.yolov8.person.head.int8_12801280.onnx'
calibration_data_dir = r'/home/wulei/wkspace/airiacv_predict/projects/person_head/data/webshare/project_anjian/personhead/datasets/val'

# 先简化 ONNX（可选）
model_onnx = onnx.load(fp32_model_path)
model_onnx, check = onnxsim.simplify(model_onnx)
onnx.save(model_onnx, fp32_model_path)

ort_session = get_ort_session(fp32_model_path)
input_name = ort_session.get_inputs()[0].name
output_names = [o.name for o in ort_session.get_outputs()]
input_shape = get_input_shape_from_model(fp32_model_path)
dr = MyDataReader(input_name, calibration_data_dir, input_size=input_shape, num_samples=1000)


import onnx

SUPPORTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}

def collect_quantizable_nodes(onnx_path):
    model = onnx.load(onnx_path)

    nodes_to_quantize = []

    for node in model.graph.node:
        if node.op_type not in SUPPORTED_OPS:
            continue

        # ORT 要求 node.name 非空
        if not node.name:
            continue

        nodes_to_quantize.append(node.name)

    return nodes_to_quantize
nodes_to_quantize = collect_quantizable_nodes(fp32_model_path)


# 执行静态量化
quantize_static(
    model_input=fp32_model_path,
    model_output=int8_model_path,
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,                        # QDQ 模式，兼容 TRT
    per_channel=True,                                    # 权重量化使用 per-channel
    reduce_range=False,                                  # 保持完整量化范围
    activation_type=QuantType.QInt8,                     # 激活量化类型
    weight_type=QuantType.QInt8,                         # 权重量化类型
    nodes_to_quantize=None,                 # 默认量化网络可量化节点
    calibrate_method=CalibrationMethod.MinMax            # min-max 校准
)

# quantize_dynamic(
#     model_input=fp32_model_path,
#     model_output=int8_model_path,
#     op_types_to_quantize=['Conv', 'Gemm', 'MatMul', 'ConvTranspose'],
#     per_channel=True,
#     reduce_range=False,
#     weight_type=QuantType.QInt8,
#     nodes_to_quantize=None,
#     nodes_to_exclude=None,
#     use_external_data_format=False,
#     extra_options=None,
# )

print(f"[DONE] INT8 ONNX 已生成: {int8_model_path}")
