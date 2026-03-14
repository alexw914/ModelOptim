import onnxruntime as ort
import numpy as np
import cv2
import os
import json
import argparse
from tqdm import tqdm

CONF_THRESH = 0.2
NMS_THRESH = 0.2
MASK_THRESH = 0.5

def get_input_shape_from_model(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    return input_shape[2], input_shape[3]

def get_ort_session(model_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    used_providers = [p for p in providers if p in available]
    print(f"使用推理设备: {used_providers[0]}")
    return ort.InferenceSession(model_path, providers=used_providers)

def letterbox(img, new_shape=(224, 224), color=(114, 114, 114)):
    shape = img.shape[:2]
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

    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)
    return input_img, r, dw, dh, img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def _flatten_det_output(output):
    output = np.squeeze(output)
    if output.ndim == 2:
        if output.shape[0] in (37,):
            return output.T
        if output.shape[1] in (37,):
            return output
    if output.ndim == 3:
        if output.shape[0] in (37,):
            c, h, w = output.shape
            return output.reshape(c, -1).T
        if output.shape[2] in (37,):
            h, w, c = output.shape
            return output.reshape(-1, c)
    raise ValueError(f"无法解析检测输出形状: {output.shape}")

def decode_detections(det_output, input_shape, scale, dw, dh, conf_thresh=CONF_THRESH):
    data = _flatten_det_output(det_output)
    if data.shape[1] < 5:
        return np.zeros((0, 4)), np.array([]), np.zeros((0, 0))

    boxes = data[:, :4].astype(np.float32)
    conf = data[:, 4].astype(np.float32)
    mask_coeffs = data[:, 5:].astype(np.float32)

    input_h, input_w = input_shape
    if boxes.max() <= 1.5:
        boxes[:, 0] *= input_w
        boxes[:, 2] *= input_w
        boxes[:, 1] *= input_h
        boxes[:, 3] *= input_h

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1 = (x1 - dw) / scale
    y1 = (y1 - dh) / scale
    x2 = (x2 - dw) / scale
    y2 = (y2 - dh) / scale
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = conf > conf_thresh
    if not np.any(keep):
        return np.zeros((0, 4)), np.array([]), np.zeros((0, mask_coeffs.shape[1]))

    return boxes[keep], conf[keep], mask_coeffs[keep]

def process_masks(mask_coeffs, proto_mask, boxes, scale, dw, dh, input_shape=(1280, 1280), mask_thresh=MASK_THRESH):
    num_dets = boxes.shape[0]
    masks = []

    if num_dets == 0:
        return masks

    proto_mask = np.squeeze(proto_mask).astype(np.float32)
    mask_coeffs = mask_coeffs.astype(np.float32)

    if proto_mask.ndim != 3:
        raise ValueError(f"proto_mask 维度异常: {proto_mask.shape}")

    mask_dim, _, _ = proto_mask.shape
    input_h, input_w = input_shape

    w0 = int(round((input_w - 2 * dw) / scale))
    h0 = int(round((input_h - 2 * dh) / scale))

    proto_resized = np.zeros((mask_dim, input_h, input_w), dtype=np.float32)
    for i in range(mask_dim):
        proto_resized[i] = cv2.resize(proto_mask[i], (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    x0 = int(round(dw))
    x1 = int(round(dw + w0 * scale))
    y0 = int(round(dh))
    y1 = int(round(dh + h0 * scale))
    proto_eff = proto_resized[:, y0:y1, x0:x1]

    h_eff, w_eff = proto_eff.shape[1], proto_eff.shape[2]
    proto_flat = proto_eff.reshape(mask_dim, -1)

    masks_flat = mask_coeffs @ proto_flat
    masks_flat = sigmoid(masks_flat)
    masks_eff = masks_flat.reshape(num_dets, h_eff, w_eff)

    for i, box in enumerate(boxes):
        mask_i = masks_eff[i]
        mask_orig = cv2.resize(mask_i, (w0, h0), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_orig > mask_thresh).astype(np.uint8)

        x1_box, y1_box, x2_box, y2_box = box.astype(int)
        x1_box = np.clip(x1_box, 0, w0)
        y1_box = np.clip(y1_box, 0, h0)
        x2_box = np.clip(x2_box, 0, w0)
        y2_box = np.clip(y2_box, 0, h0)

        mask_canvas = np.zeros((h0, w0), dtype=np.uint8)
        mask_canvas[y1_box:y2_box, x1_box:x2_box] = mask_bin[y1_box:y2_box, x1_box:x2_box]
        masks.append(mask_canvas)

    return masks

def draw_detections(image, boxes, scores, labels=None, masks=None, save_path=None):
    overlay = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            score = scores[i]
            if labels is not None and len(labels) > i:
                color = colors[int(labels[i]) % len(colors)]
            else:
                color = colors[i % len(colors)]

            if masks is not None and i < len(masks):
                mask = masks[i].astype(bool)
                colored_mask = np.zeros_like(image, dtype=np.uint8)
                colored_mask[mask] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            text = f"{score:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
            cv2.putText(overlay, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)

def infer_folder(model_path, input_dir, output_dir="none", output_json="res.json", conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH, mask_thresh=MASK_THRESH):
    ort_session = get_ort_session(model_path)
    input_name = ort_session.get_inputs()[0].name
    output_names = [o.name for o in ort_session.get_outputs()]
    input_shape = get_input_shape_from_model(model_path)

    results = []

    for filename in tqdm(sorted(os.listdir(input_dir))):
        image_path = os.path.join(input_dir, filename)
        if not os.path.isfile(image_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_tensor, scale, dw, dh, img = preprocess(image_path, input_shape, mean=[0, 0, 0], std=[255, 255, 255])
        outputs = ort_session.run(output_names, {input_name: input_tensor})

        if len(outputs) != 2:
            raise ValueError(f"预期2个输出，但得到{len(outputs)}个")

        det_output = outputs[0]
        proto_mask = outputs[1]

        print(det_output)
        boxes, scores, mask_coeffs = decode_detections(det_output, input_shape, scale, dw, dh, conf_thresh=conf_thresh)

        if boxes.shape[0] == 0:
            final_boxes, final_scores, final_masks = boxes, scores, []
        else:
            keep = nms_boxes(boxes, scores, nms_thresh)
            final_boxes, final_scores = boxes[keep], scores[keep]
            final_mask_coeffs = mask_coeffs[keep]
            final_masks = process_masks(final_mask_coeffs, proto_mask, final_boxes, scale, dw, dh, input_shape, mask_thresh=mask_thresh)

        if output_dir != "none":
            labels = np.zeros((final_boxes.shape[0],), dtype=np.int32)
            draw_detections(img, final_boxes, final_scores, labels, final_masks, save_path=os.path.join(output_dir, filename))

        results.append({
            "image": filename,
            "detections": [{
                "bbox": box.tolist(),
                "score": float(score),
                "label": 0
            } for box, score in zip(final_boxes, final_scores)]
        })

    if output_json != "none":
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"推理完成，结果已保存到 {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ONNX inference for YOLOv8-seg (general export).")
    parser.add_argument("model_path", help="ONNX 模型路径")
    parser.add_argument("input_dir", help="图像文件夹")
    parser.add_argument("--output_dir", default="./vis", help="输出检测结果图像保存目录")
    parser.add_argument("--output_json", default="yolov8_seg_res.json", help="输出 JSON 文件")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度阈值")
    parser.add_argument("--nms", type=float, default=NMS_THRESH, help="NMS 阈值")
    parser.add_argument("--mask_thresh", type=float, default=MASK_THRESH, help="mask 二值化阈值")
    args = parser.parse_args()
    infer_folder(
        args.model_path,
        args.input_dir,
        output_dir=args.output_dir,
        output_json=args.output_json,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        mask_thresh=args.mask_thresh,
    )
