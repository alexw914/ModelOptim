import onnxruntime as ort
import numpy as np
import cv2
import os
import json
import argparse
from tqdm import tqdm

NUM_BINS = 16 
CONF_THRESH = 0.01
NMS_THRESH = 0.1
STRIDES = [8, 16, 32]

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

def preprocess(image_path, input_shape=(224, 224), mean=[0, 0, 0], std=[255, 255, 255]):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
    input_img, r, dw, dh = letterbox(img, input_shape)

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    input_img = (input_img - mean) / std

    input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW

    input_img = np.expand_dims(input_img, axis=0)  # 增加 batch 维度
    return input_img, r, dw, dh, img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def dfl_decode(logits):
    prob = softmax(logits, axis=-1)
    proj = np.arange(NUM_BINS, dtype=np.float32)
    return np.sum(prob * proj, axis=-1)

def decode_head(output, stride, scale, dw, dh):
    C, H, W = output.shape
    output = output.reshape(C, -1).T
    bbox_dfl = output[:, :NUM_BINS * 4].reshape(-1, 4, NUM_BINS)
    bbox = dfl_decode(bbox_dfl) * stride
    cls_logits = output[:, NUM_BINS * 4:]
    probs = sigmoid(cls_logits)

    y, x = np.divmod(np.arange(H * W), W)
    grid_x = (x * stride + stride / 2)
    grid_y = (y * stride + stride / 2)

    x1 = (grid_x - bbox[:, 0] - dw) / scale
    y1 = (grid_y - bbox[:, 1] - dh) / scale
    x2 = (grid_x + bbox[:, 2] - dw) / scale
    y2 = (grid_y + bbox[:, 3] - dh) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    final_boxes, final_scores, final_labels = [], [], []
    for cls_id in range(probs.shape[1]):
        cls_scores = probs[:, cls_id]
        mask = cls_scores > CONF_THRESH
        if np.any(mask):
            final_boxes.append(boxes[mask])
            final_scores.append(cls_scores[mask])
            final_labels.append(np.full(np.sum(mask), cls_id, dtype=np.int32))

    if final_boxes:
        return np.concatenate(final_boxes), np.concatenate(final_scores), np.concatenate(final_labels)
    else:
        return np.zeros((0, 4)), np.array([]), np.array([])

def nms_boxes(boxes, scores, nms_thresh=0.5):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= nms_thresh)[0] + 1]
    return np.array(keep, dtype=np.int32)

def draw_detections(image, boxes, scores, labels, class_names=None, save_path=None):

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            score = scores[i]
            label = labels[i]
            class_name = class_names[label] if class_names else str(label)
            color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{class_name}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)

def infer_folder(model_path, input_dir, output_dir="none", output_json="res.json"):

    ort_session = get_ort_session(model_path)
    input_name = ort_session.get_inputs()[0].name
    output_names = [o.name for o in ort_session.get_outputs()]
    input_shape = get_input_shape_from_model(model_path)

    results = []

    for filename in tqdm(sorted(os.listdir(input_dir))):
        image_path = os.path.join(input_dir, filename)
        if not os.path.isfile(image_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            input_tensor, scale, dw, dh, img = preprocess(image_path, input_shape, mean=[0, 0, 0], std=[255, 255, 255])
            outputs = ort_session.run(output_names, {input_name: input_tensor})

            all_boxes, all_scores, all_labels = [], [], []
            for i, output in enumerate(outputs):
                output = np.squeeze(output)
                stride = STRIDES[i]
                boxes, scores, labels = decode_head(output, stride, scale, dw, dh)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            keep = nms_boxes(boxes, scores, NMS_THRESH)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if output_dir != "none":
                draw_detections(img, boxes, scores, labels, save_path=os.path.join(output_dir, filename))

            results.append({
                "image": filename,
                "detections": [{
                    "bbox": box.tolist(),
                    "score": float(score),
                    "label": int(label)
                } for box, score, label in zip(boxes, scores, labels)]
            })
        except Exception as e:
            print(f"跳过文件 {filename}，因错误: {e}")

    if output_json != "none":
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"推理完成，结果已保存到 {output_json}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch ONNX inference to JSON.")
    parser.add_argument("model_path", help="ONNX 模型路径")
    parser.add_argument("input_dir", help="图像文件夹")
    parser.add_argument("--output_dir", default="./vis", help="输出检测结果图像保存目录")

    args = parser.parse_args()

    infer_folder(args.model_path, args.input_dir, output_dir=args.output_dir, output_json="yolov8_res.json")
