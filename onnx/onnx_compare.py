# compare_onnx_outputs.py
import sys
import numpy as np
import onnx
import onnxruntime as ort

def load_input_info(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    if len(graph.input) == 0:
        raise RuntimeError("Model has no inputs.")
    inp = graph.input[0]
    shape = []
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            # dynamic dim -> default to 1
            shape.append(1)
    dtype = inp.type.tensor_type.elem_type
    return inp.name, shape, dtype

def ort_dtype_to_np(onnx_dtype):
    # most common cases
    from onnx import TensorProto
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.FLOAT16: np.float16,
    }
    if onnx_dtype not in mapping:
        raise RuntimeError(f"Unsupported dtype: {onnx_dtype}")
    return mapping[onnx_dtype]

def run_model(model_path, input_name, input_data):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {input_name: input_data})
    return outputs

def main(model_a, model_b, seed=0, atol=1e-5, rtol=1e-4):
    name_a, shape_a, dtype_a = load_input_info(model_a)
    name_b, shape_b, dtype_b = load_input_info(model_b)

    if shape_a != shape_b:
        raise RuntimeError(f"Input shapes differ: {shape_a} vs {shape_b}")
    if dtype_a != dtype_b:
        raise RuntimeError(f"Input dtypes differ: {dtype_a} vs {dtype_b}")

    np_dtype = ort_dtype_to_np(dtype_a)
    rng = np.random.default_rng(seed)
    input_data = rng.standard_normal(shape_a).astype(np_dtype)

    out_a = run_model(model_a, name_a, input_data)
    out_b = run_model(model_b, name_b, input_data)

    if len(out_a) != len(out_b):
        raise RuntimeError(f"Output count differs: {len(out_a)} vs {len(out_b)}")

    all_ok = True
    for i, (oa, ob) in enumerate(zip(out_a, out_b)):
        ok = np.allclose(oa, ob, atol=atol, rtol=rtol)
        print(f"Output[{i}] equal: {ok}  max_abs_diff={np.max(np.abs(oa - ob))}")
        all_ok = all_ok and ok

    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_onnx_outputs.py model_a.onnx model_b.onnx")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))