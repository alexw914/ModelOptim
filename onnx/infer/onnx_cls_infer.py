import onnxruntime as ort
import numpy as np
import cv2
import os
import json
import argparse
from tqdm import tqdm

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
    return img

def preprocess(image_path, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
    img = letterbox(img, (192, 192), (0, 0, 0))
#    cv2.imwrite(os.path.join("input", os.path.basename(image_path)), img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 增加 batch 维度
    return img

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def infer_folder(model_path, image_dir, output_json, activation='none'):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = ort.InferenceSession(model_path,sess_options,
     providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    results = []

    for filename in tqdm(sorted(os.listdir(image_dir))):
        image_path = os.path.join(image_dir, filename)
        if not os.path.isfile(image_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            input_tensor = preprocess(image_path,mean=[0, 0, 0], std=[255, 255, 255])
            output = ort_session.run([output_name], {input_name: input_tensor})[0]
            if activation == 'softmax':
                cls_preds = softmax(output)
            elif activation == 'sigmoid':
                cls_preds = sigmoid(output)
            else:
                cls_preds = output

            results.append({
                "output": output.tolist(),
                "cls_preds": cls_preds.tolist(),
                "image_filename": filename
            })

        except Exception as e:
            print(f"跳过文件 {filename}，因错误: {e}")

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"推理完成，结果已保存到 {output_json}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch ONNX inference to JSON.")
    parser.add_argument("model", help="ONNX 模型路径")
    parser.add_argument("input_dir", help="图像所在文件夹")
    parser.add_argument("--activation", choices=["none", "softmax", "sigmoid"], default="softmax",
                        help="是否对输出应用 softmax 或 sigmoid")
    args = parser.parse_args()

    infer_folder(args.model, args.input_dir, output_json="cls_res.json", activation=args.activation)
