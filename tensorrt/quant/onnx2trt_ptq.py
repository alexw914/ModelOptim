import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import onnxruntime as ort

def get_input_shape_from_model(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    return input_shape[2], input_shape[3]

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

def preprocess(image_path, input_shape=(224, 224),
        mean=[0, 0, 0], std=[255, 255, 255],
        # mean=[123.675, 116.280, 130.530], std=[58.395, 57.120, 57.375],
        to_rgb=True):
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

class Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, imgpath, cache_file, inputsize=[1280, 1280], to_rgb=True):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.batch_size = 1
        self.channel = 3
        self.to_rgb = to_rgb
        self.height, self.width = inputsize
        self.imgs = [os.path.join(imgpath, f) for f in os.listdir(imgpath) if f.endswith(".jpg")]
        np.random.shuffle(self.imgs)
        if len(self.imgs) == 0:
            raise RuntimeError("No calibration images found in {}".format(imgpath))
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs)
        self.data_size = self.batch_size * self.channel * self.height * self.width * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        if self.batch_idx >= self.max_batch_idx:
            return None
        batch_file = self.imgs[self.batch_idx]
        img = preprocess(batch_file, input_shape=(self.height, self.width), to_rgb=self.to_rgb)[0]
        img = np.ascontiguousarray(img, dtype=np.float32)
        cuda.memcpy_htod(self.device_input, img)
        self.batch_idx += 1
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()


def get_engine(onnx_file_path, engine_file_path, cali_img_path, to_rgb=True, mode='FP32'):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    def build_engine():
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:

            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                for i in range(parser.num_errors):
                    print(parser.get_error(i).desc())
                    return None

            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * 1024 * 1024)

            h, w = get_input_shape_from_model(onnx_file_path)
            if mode.lower() == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif mode.lower() == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                calibrator = Calibrator(cali_img_path, onnx_file_path.replace('.onnx', '.cache'), inputsize=[h, w], to_rgb=to_rgb)
                config.int8_calibrator = calibrator

            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            min_shape = (1, 3, h, w)
            opt_shape = (1, 3, h, w)
            max_shape = (1, 3, h, w)
            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)

            engine = builder.build_serialized_network(network, config)

            if engine is None:
                print("ERROR: Engine build failed!")
                return None

            with open(engine_file_path, "wb") as f:
                f.write(engine)
            print("Engine build and saved:", engine_file_path)
            return engine

    if os.path.exists(engine_file_path):
        print("Reading engine from file:", engine_file_path)
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                print("ERROR: Failed to deserialize existing engine, rebuilding...")
                return build_engine()
            return engine
    else:
        return build_engine()

def process_all_onnx():
    base_dirs = ["rgb_onnx","bgr_onnx"]
    output_dirs = {"rgb_onnx": "t4_int8_trt", "rgb_onnx": "t4_int8_trt"}

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"❌ 路径不存在：{base_dir}")
            continue

        os.makedirs(output_dirs[base_dir], exist_ok=True)
        to_rgb = base_dir.startswith("rgb")

        onnx_files = [f for f in os.listdir(base_dir) if f.endswith(".onnx")]
        if not onnx_files:
            print(f"⚠️ 目录 {base_dir} 下没有找到 ONNX 文件")
            continue

        for onnx_file in onnx_files:
            onnx_path = os.path.join(base_dir, onnx_file)
            engine_path = os.path.join(output_dirs[base_dir], onnx_file.replace(".onnx", ".trt"))
            if os.path.exists(engine_path):
                continue
            # 提取模型名前缀
            prefix = onnx_file.split("_")[0]
            cali_img_path = os.path.join("cali_data", prefix, "val")

            if not os.path.exists(cali_img_path):
                print(f"⚠️ 未找到校准图片路径：{cali_img_path}，跳过 {onnx_file}")
                continue

            print("=" * 80)
            print(f"🔧 开始转换: {onnx_path}")
            print(f"📁 校准路径: {cali_img_path}")
            print(f"🎨 颜色空间: {'RGB' if to_rgb else 'BGR'}")
            print(f"💾 输出路径: {engine_path}")
            print("=" * 80)

            get_engine(onnx_file_path=onnx_path, engine_file_path=engine_path, cali_img_path=cali_img_path, to_rgb=to_rgb, mode="int8")

if __name__ == "__main__":
    process_all_onnx()