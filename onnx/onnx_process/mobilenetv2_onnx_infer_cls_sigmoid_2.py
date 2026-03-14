import os
import shutil
import cv2
import numpy as np
import onnxruntime

target_sz = (224, 224)

def sigmoid_(x):
    return 1 / (1 + np.exp(-x))

import cv2
import numpy as np
import onnxruntime


def resize_keep_ratio(img, target_size):
    """等比 resize，返回 resized_img, scale_ratio"""
    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale

# def resize_keep_ratio(img, target_size, interpolation=cv2.INTER_LINEAR):
#     """
#     对齐 mmcv Resize(size=target_size, keep_ratio=True, mode='normal')
#     """
#     h, w = img.shape[:2]
#     target_h, target_w = target_size
#
#     scale_h = target_h / h
#     scale_w = target_w / w
#     scale = min(scale_h, scale_w)
#
#     new_h = int(h * scale + 0.5)
#     new_w = int(w * scale + 0.5)
#
#     resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
#     return resized, scale



def impad(img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant'):
    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }

    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)
    return img

def letterbox(img, target_size, border_value=(0, 0, 0)):
    """与 YOLO/MMDeploy 完全一致的 LetterBox 实现"""
    target_h, target_w = target_size
    h, w = img.shape[:2]

    # 计算 padding（上下左右对称）

    dh = (target_h - h) / 2
    dw = (target_h - w) / 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = impad(
        img, padding=(left, top, right, bottom), pad_val=border_value)



    return img





def get_onnx_model(onnx_path):
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return session
def onnx_infer(session, img_path, input_size=(192, 192),border_value=(114,114,114)):
    # ---- LoadImageFromFile ----
    img = cv2.imread(img_path)  # BGR
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- Resize keep_ratio ----
    resized, _ = resize_keep_ratio(img, input_size)

    # ---- LetterBox ----
    boxed = letterbox(resized, input_size, border_value=border_value)

    # ---- Normalize ----
    # mean=[0,0,0], std=[255,255,255]
    boxed = boxed.astype(np.float32) / 255.0

    # ---- Transpose to CHW ----
    boxed = np.transpose(boxed, (2, 0, 1))

    # ---- Newaxis ----
    img_input = np.expand_dims(boxed, axis=0)  # (1,3,H,W)

    # ---- ONNX Inference ----

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})[0]

    return outputs

def classify_and_move(img_path, pred_idx, base_output_dir):
    cls_name = labels[pred_idx]
    save_dir = os.path.join(base_output_dir, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    basename = os.path.basename(img_path)
    dst_path = os.path.join(save_dir, basename)

    # 避免覆盖
    if os.path.exists(dst_path):
        name, ext = os.path.splitext(basename)
        dst_path = os.path.join(save_dir, f"{name}_dup{ext}")

    # shutil.copy(img_path, dst_path)
    # print(f"移动: {img_path}  ->  {dst_path}")


# ----------------------------------------------
labels = ['no_do', 'do']   # 二分类
onnx_path = r"/Users/krisw/Downloads/project_anjian_call_v0_0_0_1_int8.onnx"
img_path = (r"/Users/krisw/Downloads/call_crop")
save_root = r"/Users/krisw/Downloads/call_crop_result"
# ----------------------------------------------
target_sz = (192, 192)
session=get_onnx_model(onnx_path)
for img_file in sorted(os.listdir(img_path)):
    print(img_file)
    img_ = os.path.join(img_path, img_file)

    outputs = onnx_infer(session, img_,target_sz,border_value=(0,0,0))

    # outputs shape = (1,1) → 单值 sigmoid
    if outputs.shape[-1] == 1:
        prob = sigmoid_(outputs[0][0])
        probs = np.array([1-prob, prob])  # 转为两类概率
    else:
        # outputs = (1,2)
        logits = outputs[0]
        probs = sigmoid_(logits)

    pred_idx = int(probs.argmax())

    # print("---------------------")
    print(outputs)
    print("probs:", probs)
    # print("预测结果:", labels[pred_idx], " 概率:", probs[pred_idx])
    # if pred_idx==1:
    #     print("预测结果:", labels[pred_idx], " 概率:", probs[pred_idx])


    classify_and_move(img_, pred_idx, save_root)
