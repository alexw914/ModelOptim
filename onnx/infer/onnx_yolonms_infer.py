import cv2
import numpy as np
import onnxruntime as ort
import os

def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=color)
    assert img_padded.shape[:2] == new_shape, f"{img_padded.shape[:2]} vs {new_shape}"
    return img_padded, r, left, top

def preprocess_image(image_path, img_size=640):
    img0 = cv2.imread(image_path)
    assert img0 is not None, f"Image {image_path} not found"
    img, ratio, pad_x, pad_y = letterbox(img0, new_shape=(img_size, img_size), color=(114,114,114))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = img_norm.transpose(2,0,1)
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch, img0, ratio, pad_x, pad_y

def draw_boxes(img0, boxes, ratio, pad_x, pad_y, class_names=None):
    h0, w0 = img0.shape[:2]
    r = ratio
    for (x1,y1,x2,y2,conf,cls) in boxes:
        xp1 = int((x1 - pad_x) / r)
        yp1 = int((y1 - pad_y) / r)
        xp2 = int((x2 - pad_x) / r)
        yp2 = int((y2 - pad_y) / r)
        xp1 = max(0, min(xp1, w0-1))
        yp1 = max(0, min(yp1, h0-1))
        xp2 = max(0, min(xp2, w0-1))
        yp2 = max(0, min(yp2, h0-1))

        color = (0, 255, 0)
        cv2.rectangle(img0, (xp1, yp1), (xp2, yp2), color, 2)

        label = f"{class_names[cls] if class_names else cls}:{conf:.2f}"
        cv2.putText(img0, label, (xp1, yp1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img0

def inference_single_image(sess, input_name, output_name, image_path,
                           img_size=640, conf_threshold=0.25, class_names=None, save_path="result.jpg"):
    img_batch, img0, ratio, pad_x, pad_y = preprocess_image(image_path, img_size)
    outputs = sess.run([output_name], {input_name: img_batch})
    out = outputs[0]
    
    print(out[0])
    boxes = []
    for x1, y1, x2, y2, conf, cls in out[0]:
        if conf >= conf_threshold:
            boxes.append((x1,y1,x2,y2,conf,int(cls)))

    img_drawn = draw_boxes(img0, boxes, ratio, pad_x, pad_y, class_names)
    cv2.imwrite(save_path, img_drawn)
    print(f"Saved result → {save_path}")

def inference_dir(model_path, image_dir, out_dir="results",
                  img_size=640, conf_threshold=0.25, class_names=None):

    os.makedirs(out_dir, exist_ok=True)

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    exts = {".jpg", ".png", ".jpeg", ".bmp"}

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1].lower() not in exts:
            continue

        img_path = os.path.join(image_dir, filename)
        save_path = os.path.join(out_dir, filename)

        inference_single_image(sess, input_name, output_name, img_path,
                               img_size, conf_threshold, class_names, save_path)

if __name__ == "__main__":
    model_path = "yolo11n.onnx"
    image_dir = "/Users/krisw/Downloads/test_video/test/jjx_frames"     # 输入文件夹
    out_dir = "results"      # 输出文件夹

    class_names = {
        0:"person",
        1:"bicycle",
        2:"car",
        3:"motorbike",
        4:"aeroplane",
        5:"bus",
        6:"train",
        7:"truck",
        8:"boat",
        9:"traffic light",
        10:"fire hydrant",
        11:"stop sign",
        12:"parking meter",
        13:"bench",
        14:"bird",
        15:"cat",
        16:"dog",
        17:"horse",
        18:"sheep",
        19:"cow",
        20:"elephant",
        21:"bear",
        22:"zebra",
        23:"giraffe",
        24:"backpack",
        25:"umbrella",
        26:"handbag",
        27:"tie",
        28:"suitcase",
        29:"frisbee",
        30:"skis",
        31:"snowboard",
        32:"sports ball",
        33:"kite",
        34:"baseball bat",
        35:"baseball glove",
        36:"skateboard",
        37:"surfboard",
        38:"tennis racket",
        39:"bottle",
        40:"wine glass",
        41:"cup",
        42:"fork",
        43:"knife",
        44:"spoon",
        45:"bowl",
        46:"banana",
        47:"apple",
        48:"sandwich",
        49:"orange",
        50:"broccoli",
        51:"carrot",
        52:"hot dog",
        53:"pizza",
        54:"donut",
        55:"cake",
        56:"chair",
        57:"sofa",
        58:"pottedplant",
        59:"bed",
        60:"diningtable",
        61:"toilet",
        62:"tvmonitor",
        63:"laptop",
        64:"mouse",
        65:"remote",
        66:"keyboard",
        67:"cell phone",
        68:"microwave",
        69:"oven",
        70:"toaster",
        71:"sink",
        72:"refrigerator",
        73:"book",
        74:"clock",
        75:"vase",
        76:"scissors",
        77:"teddy bear",
        78:"hair drier",
        79:"toothbrush"
    }


    inference_dir(model_path, image_dir, out_dir,
                  img_size=1280, conf_threshold=0.25, class_names=class_names)
