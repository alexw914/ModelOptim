import onnxruntime as ort
import numpy as np
import cv2
import os
import json
import argparse
from tqdm import tqdm

NUM_BINS = 16
CONF_THRESH = 0.3
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

def regularize_boxes(rboxes):
    cx, cy, w, h, theta = np.split(rboxes, 5, axis=1)
    swap_mask = w > h
    w_, h_ = np.where(swap_mask, h, w), np.where(swap_mask, w, h)
    theta_ = np.where(swap_mask, (theta + np.pi / 2) % np.pi, theta)
    return np.concatenate([cx, cy, w_, h_, theta_], axis=1)

def decode_head_obb(output, stride, scale, dw, dh):
    C, H, W = output.shape
    N = H * W
    reg_max = NUM_BINS

    num_box_channels = reg_max * 4
    num_theta_channels = 1
    num_cls_channels = C - num_box_channels - num_theta_channels
    output = output.reshape(C, -1)
    
    # DFL解码
    dfl_raw = output[:num_box_channels, :].T.reshape(N, 4, reg_max)  # (N, 4, reg_max)
    prob = softmax(dfl_raw, axis=-1)
    proj = np.arange(reg_max, dtype=np.float32)
    offsets = np.sum(prob * proj, axis=-1)  # (N, 4)

    # 角度解码
    theta= output[num_box_channels, :]  # (N,)
    theta = (sigmoid(theta) - 0.25) * np.pi  # [-pi/4, 3pi/4] in radians

    cos = np.cos(theta)
    sin = np.sin(theta)

    lt_x = offsets[:, 0]  # left
    lt_y = offsets[:, 1]  # top
    rb_x = offsets[:, 2]  # right
    rb_y = offsets[:, 3]  # bottom

    xf = (rb_x - lt_x) / 2
    yf = (rb_y - lt_y) / 2

    yv, xv = np.divmod(np.arange(N), W)
    grid_x = xv + 0.5
    grid_y = yv + 0.5

    # 仿射计算 center
    cx = (xf * cos - yf * sin + grid_x) * stride
    cy = (xf * sin + yf * cos + grid_y) * stride

    # size 计算为 (lt + rb) * stride
    w = (rb_x + lt_x) * stride
    h = (rb_y + lt_y) * stride

    # 还原 letterbox 坐标
    cx = (cx - dw) / scale
    cy = (cy - dh) / scale
    w /= scale
    h /= scale

    boxes = np.stack([cx, cy, w, h, theta], axis=1)
#    boxes = regularize_boxes(boxes)

    # 解码分类分支
    cls_logits = output[num_box_channels + 1:, :].T  # (N, num_cls)
    scores = sigmoid(cls_logits)

    # 阈值过滤
    final_boxes, final_scores, final_labels = [], [], []
    for cls_id in range(scores.shape[1]):
        cls_scores = scores[:, cls_id]
        mask = cls_scores > CONF_THRESH
        if np.any(mask):
            final_boxes.append(boxes[mask])
            final_scores.append(cls_scores[mask])
            final_labels.append(np.full(np.sum(mask), cls_id, dtype=np.int32))

    if final_boxes:
        return np.concatenate(final_boxes), np.concatenate(final_scores), np.concatenate(final_labels)
    else:
        return np.zeros((0, 5)), np.array([]), np.array([])


def rotated_iou_numpy(box1, box2):
    """
    计算两个旋转框的 IoU，box 为 [cx, cy, w, h, theta]，theta 单位为角度
    """
    rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

    int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    if int_pts is None or len(int_pts) < 3:
        return 0.0

    inter_area = cv2.contourArea(int_pts)
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def rotated_nms(boxes, scores, iou_thresh=0.1):
    """
    NumPy 实现的旋转框 NMS
    boxes: (N, 5) -> cx, cy, w, h, angle (单位为度)
    scores: (N,)
    返回保留索引
    """
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = np.array([
            rotated_iou_numpy(boxes[i], boxes[j])
            for j in order[1:]
        ])
        order = order[1:][ious <= iou_thresh]

    return np.array(keep, dtype=np.int32)

def draw_obb(image, boxes, scores, labels, class_names=None, save_path=None):
    for i, box in enumerate(boxes):
        cx, cy, w, h, theta = box
        score = scores[i]
        label = labels[i]
        class_name = class_names[label] if class_names else str(label)
        color = (0, 0, 255)

        rect = ((cx, cy), (w, h), theta)
        box_pts = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box_pts], 0, color, 2)

        text = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (int(cx), int(cy - th - 4)), (int(cx + tw), int(cy)), color, -1)
        cv2.putText(image, text, (int(cx), int(cy - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

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
            input_tensor, scale, dw, dh, img = preprocess(image_path, input_shape)
            outputs = ort_session.run(output_names, {input_name: input_tensor})

            all_boxes, all_scores, all_labels = [], [], []

            for i, output in enumerate(outputs):
                output = np.squeeze(output)
                stride = STRIDES[i]
                boxes, scores, labels = decode_head_obb(output, stride, scale, dw, dh)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            # OBB NMS
            if len(boxes) > 0:
                # 注意: theta 需为角度
                boxes[:, 4] = np.degrees(boxes[:, 4])
                keep = rotated_nms(boxes, scores, NMS_THRESH)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            if output_dir != "none":
                draw_obb(img, boxes, scores, labels, save_path=os.path.join(output_dir, filename))

            results.append({
                "image": filename,
                "detections": [  # xywhtheta
                    {
                        "bbox": box.tolist(),
                        "score": float(score),
                        "label": int(label)
                    } for box, score, label in zip(boxes, scores, labels)
                ]
            })
        except Exception as e:
            print(f"跳过文件 {filename}，因错误: {e}")

    if output_json != "none":
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"推理完成，结果已保存到 {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 OBB ONNX inference.")
    parser.add_argument("model_path", help="ONNX 模型路径")
    parser.add_argument("input_dir", help="图像文件夹路径")
    parser.add_argument("--output_dir", default="./vis", help="检测结果图像输出目录")
    args = parser.parse_args()

    infer_folder(args.model_path, args.input_dir, output_dir=args.output_dir, output_json="yolov8obb_res.json")
