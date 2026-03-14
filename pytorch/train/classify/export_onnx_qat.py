import argparse
from pathlib import Path

import numpy as np
import torch

import utils
from datasets import build_dataset
from main import get_args_parser
from models import *

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization import quant_utils


def build_parser():
    parser = argparse.ArgumentParser(
        "Export QAT checkpoint to ONNX",
        parents=[get_args_parser()],
    )
    parser.add_argument(
        "--checkpoint_path",
        default="runs_qat/checkpoint-best.pth",
        type=str,
        help="QAT checkpoint path",
    )
    parser.add_argument(
        "--export_kind",
        default="qdq",
        choices=["fp32", "qdq", "int8", "int4"],
        help="ONNX export kind",
    )
    parser.add_argument(
        "--output_file",
        default="",
        type=str,
        help="Output ONNX file path; empty means auto-generate under runs_qat",
    )
    parser.add_argument(
        "--calibration_samples",
        default=128,
        type=int,
        help="Number of eval images to use for static calibration",
    )
    parser.add_argument(
        "--calibration_batch_size",
        default=1,
        type=int,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--opset_version",
        default=0,
        type=int,
        help="Override ONNX opset; 0 means auto-select",
    )
    parser.add_argument(
        "--int4_block_size",
        default=128,
        type=int,
        help="Block size for ORT int4 weight-only quantization",
    )
    parser.add_argument(
        "--int4_symmetric",
        default=True,
        type=str2bool,
        help="Use signed int4 instead of uint4",
    )
    parser.set_defaults(
        output_dir="runs_qat",
        eval=True,
        batch_size=1,
        num_workers=0,
        model="senet18",
        nb_classes=0,
    )
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def create_model(args):
    if args.model == "mobilenet_v3_small":
        return MobileNetV3_Small(num_classes=args.nb_classes)
    if args.model == "mobilenet_v3_large":
        return MobileNetV3_Large(num_classes=args.nb_classes)
    if args.model == "senet18":
        return se_resnet_18(num_classes=args.nb_classes)
    if args.model == "senet34":
        return se_resnet_34(num_classes=args.nb_classes)
    raise ValueError(f"Unsupported model: {args.model}")


def extract_model_state(checkpoint):
    for key in ("model", "module"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def load_checkpoint_model(model, checkpoint_path):
    checkpoint = utils.load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_model = extract_model_state(checkpoint)
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in checkpoint_model.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"Skip {len(skipped)} checkpoint keys that do not match the export model")
    utils.load_state_dict(model, filtered_state)


def infer_output_file(args):
    if args.output_file:
        return Path(args.output_file)
    suffix = {
        "fp32": "fp32",
        "qdq": "qdq",
        "int8": "int8",
        "int4": "int4",
    }[args.export_kind]
    return Path(args.output_dir) / f"{args.model}_{suffix}.onnx"


def export_fp32_onnx(model, output_path, input_size, opset_version):
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Saved fp32 ONNX to {output_path}")


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, max_batches):
        self.data_loader = data_loader
        self.max_batches = max_batches
        self.enum_data = None
        self.rewind()

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        count = 0
        batches = []
        for samples, _ in self.data_loader:
            batches.append({"image": samples.numpy().astype(np.float32)})
            count += 1
            if count >= self.max_batches:
                break
        self.enum_data = iter(batches)


def build_calibration_reader(args):
    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    max_batches = max(1, args.calibration_samples // args.calibration_batch_size)
    return ImageCalibrationDataReader(data_loader, max_batches)


def export_int8_or_qdq(args, fp32_path, output_path):
    reader = build_calibration_reader(args)
    if args.export_kind == "qdq":
        quant_format = QuantFormat.QDQ
        activation_type = QuantType.QInt8
        weight_type = QuantType.QInt8
    else:
        quant_format = QuantFormat.QOperator
        activation_type = QuantType.QUInt8
        weight_type = QuantType.QInt8

    quantize_static(
        str(fp32_path),
        str(output_path),
        reader,
        quant_format=quant_format,
        per_channel=True,
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": args.export_kind == "qdq",
            "WeightSymmetric": True,
        },
    )
    print(f"Saved {args.export_kind} ONNX to {output_path}")


def export_int4(fp32_path, output_path, args):
    try:
        from onnxruntime.quantization import matmul_bnb4_quantizer
    except ImportError as exc:
        raise RuntimeError(
            "Current onnxruntime environment does not expose int4 export helpers. "
            "QDQ and int8 export remain available."
        ) from exc

    if not hasattr(quant_utils, "load_model_with_shape_infer"):
        raise RuntimeError("Current onnxruntime package does not provide int4 shape-inference helpers")

    quant_format = getattr(quant_utils, "QuantFormat", QuantFormat)
    print("Exporting int4 ONNX with ORT weight-only quantization. Only MatMul nodes are quantized.")
    quant_config = matmul_bnb4_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=args.int4_block_size,
        is_symmetric=args.int4_symmetric,
        accuracy_level=4,
        quant_format=quant_format.QOperator,
        op_types_to_quantize=("MatMul",),
        quant_axes=(("MatMul", 0),),
    )
    model = quant_utils.load_model_with_shape_infer(Path(fp32_path))
    quantizer = matmul_bnb4_quantizer.MatMulBnb4Quantizer(
        model,
        nodes_to_exclude=None,
        nodes_to_include=None,
        algo_config=quant_config,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output_path), True)
    print(f"Saved int4 ONNX to {output_path}")


def main(args):
    output_path = infer_output_file(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.export_kind == "int4":
        opset_version = args.opset_version or 21
    else:
        opset_version = args.opset_version or 17

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    print(f"Detected {args.nb_classes} classes from dataset")
    del dataset_train

    model = create_model(args)
    load_checkpoint_model(model, args.checkpoint_path)
    model.to("cpu")
    model.eval()

    if args.export_kind == "fp32":
        export_fp32_onnx(model, output_path, args.input_size, opset_version)
        return

    fp32_path = output_path.with_name(output_path.stem + "_fp32.onnx")
    export_fp32_onnx(model, fp32_path, args.input_size, opset_version)

    if args.export_kind in ("qdq", "int8"):
        export_int8_or_qdq(args, fp32_path, output_path)
    elif args.export_kind == "int4":
        export_int4(fp32_path, output_path, args)
    else:
        raise ValueError(args.export_kind)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
