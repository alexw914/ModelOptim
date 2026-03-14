import random
import shutil

import onnx2tf
import numpy as np
import os
import cv2
import onnxruntime as ort
from sympy.core.random import shuffle


def resize(img, new_shape):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    resize_shape = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, dsize=(resize_shape[1], resize_shape[0]))
    return img

def pad(img, new_shape):
    h, w = img.shape[:2]
    if h == new_shape[0] and w == new_shape[1]:
        return img

    top = (new_shape[0] - h) // 2
    left = (new_shape[1] - w) // 2
    bottom = new_shape[0] - h - top
    right = new_shape[1] - w - left
    value = (0, 0, 0)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return img

# 定义数据预处理函数
def preprocess_image(image_path, input_size=(224, 224)):
    filename = os.path.basename(image_path)
    new_name = filename + ".npy"
    dir_name = os.path.dirname(image_path)
    new_dir = dir_name + "_npy"
    # new_dir = r"C:\Users\Lenovo\Desktop\qichechebahand_det_data_v1_npy"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    new_path = f'{new_dir}/{new_name}'
    if os.path.exists(new_path):
        return new_path
    img = cv2.imread(image_path)
    new_shape = input_size
    img = resize(img, new_shape)
    img = pad(img, new_shape)

    # img = Image.open(image_path).convert('RGB')
    # img = img.resize(target_size)
    # img_array = np.array(img).astype(np.float32)
    # 归一化处理，根据模型要求调整
    # img_array = img / 255.0
    # # 添加批次维度 (H,W,C) -> (B,H,W,C)
    # img_array = np.expand_dims(img_array, axis=0)
    # img = np.ascontiguousarray(img.transpose(2, 0, 1))[None, :, :, :].astype(np.float32)
    img = np.ascontiguousarray(img)[None, :, :, :].astype(np.float32) # (B,H,W,C)

    np.save(f"{new_path}",img)
    return new_path


# 创建校准数据集生成器
def calibration_dataset_generator(data_dir,input_size, num_samples=1000):
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files = image_files[:num_samples]
    random.seed(1000)
    random.shuffle(image_files)
    out = []
    for image_path in image_files:
        out_path = preprocess_image(image_path,input_size)
        # if os.path.basename(out_path) =="zhlh_996.jpg.npy":
        #     continue
        # 返回字典，键为模型输入名称（需根据实际模型修改）
        # out.append([{"input_op_name": input_name}, {"numpy_file_path": out_path}, {"mean": np.array([0., 0., 0.])}, {"std": np.array([255., 255., 255.])}])

        # out.append([input_name, out_path, np.array([[[[0.]],[[ 0.]], [[0.]]]]),np.array([[[[255.]], [[255.]],[[ 255.]]]])])
        out.append([input_name, out_path, np.array([[[[0., 0.,0.]]]]),np.array([[[[255.,255., 255.]]]])])
    return out

onnx_path = 'personhead_detect_yolov8_v0_0_0_0.onnx'
# 必须要优化结构
model_onnx = onnx.load(onnx_path)
model_onnx, check = onnxsim.simplify(model_onnx)
onnx.save(model_onnx, onnx_path)

input_size = (1280,1280)
# 定义数据集目录
calibration_data_dir = r'\\172.16.60.45\syhg_personal_tmp\zhangyanhua\数据\person_head_detection\data\test_data'
ort_session = ort.InferenceSession(onnx_path,
                                       providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                  'CPUExecutionProvider'])

input_name = ort_session.get_inputs()[0].name
out_dir = r'C:\Users\Lenovo\Desktop\lianghua_onnx2tf'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
# if os.path.exists(out_dir):
#     os.remove(out_dir)
# 调用onnx2tf进行量化，使用自定义校准数据集
onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=out_dir,
    quant_type="per-tensor",
    output_integer_quantized_tflite =True,
    # quantization=True,
    # quantization_method='integer_quantization',  # 或 'weight_only_quantization'
    custom_input_op_name_np_data_path=calibration_dataset_generator(calibration_data_dir,input_size),
    # 其他可选参数
)
dirname = os.path.basename(out_dir)
if os.path.exists(f'./{dirname}'):
    shutil.rmtree(f'./{dirname}')
shutil.move(out_dir, './')


#############################################
# 量化前 onnx 必须要先onnxsim.simplify 优化onnx模型结构
# model_onnx = onnx.load(onnx_path)
# model_onnx, check = onnxsim.simplify(model_onnx)
# onnx.save(model_onnx, onnx_path)
#############################################

# 将tf转onnx
def tflite2onnx(model_path, out_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_name = input_details[0]['name']  # 'input_0_f.1'
    output_details = interpreter.get_output_details()
    names = []
    for output in output_details:
        names.append(output['name'])
    # names = [output_details[0]['name'],output_details[1]['name'],output_details[2]['name']]
    names.sort()
    output_name = ','.join(names)
    # output_name = f"{names[0]},{names[1]},{names[2]}"  # '1673'
    cmd = [
        sys.executable, '-m', 'tf2onnx.convert', '--opset', '13',
        '--tflite', model_path,
        '--output', out_path,
        '--outputs', output_name,
        '--outputs-as-nchw', output_name,
        '--inputs', input_name,
        '--inputs-as-nchw', input_name
    ]
    print(cmd)
    subprocess.run(cmd, check=True)
    print(f'量化的onnx模型保存成功，{out_path}')
out_dir = 'lianghua_onnx2tf'
model_path = f'{out_dir}/{basename}_integer_quant.tflite'
out_path = f'{out_dir}/{basename}_int8_lianghua.onnx'
tflite2onnx(model_path, out_path)



