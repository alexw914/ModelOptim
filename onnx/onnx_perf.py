import argparse
import json
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort


def _fix_dim(dim):
    if dim is None:
        return 1
    if isinstance(dim, str):
        return 1
    if isinstance(dim, int):
        return dim if dim > 0 else 1
    return 1


def _dtype_from_ort(ort_type):
    # ort_type example: "tensor(float)" / "tensor(int64)" / "tensor(uint8)"
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    if "double" in ort_type:
        return np.float64
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "int16" in ort_type:
        return np.int16
    if "int8" in ort_type:
        return np.int8
    if "uint64" in ort_type:
        return np.uint64
    if "uint32" in ort_type:
        return np.uint32
    if "uint16" in ort_type:
        return np.uint16
    if "uint8" in ort_type:
        return np.uint8
    if "bool" in ort_type:
        return np.bool_
    return np.float32


def build_dummy_inputs(session):
    feeds = {}
    for inp in session.get_inputs():
        shape = tuple(_fix_dim(d) for d in inp.shape)
        dtype = _dtype_from_ort(inp.type)
        if dtype == np.bool_:
            data = (np.random.rand(*shape) > 0.5)
        elif np.issubdtype(dtype, np.integer):
            data = np.random.randint(0, 2, size=shape, dtype=dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        feeds[inp.name] = data
    return feeds


def summarize_outputs(outputs, output_infos):
    print("Output summary:")
    for out, info in zip(outputs, output_infos):
        name = info.name
        if hasattr(out, "shape"):
            shape = out.shape
        else:
            shape = "n/a"
        dtype = getattr(out, "dtype", type(out))
        line = f"{name:40s} shape={shape} dtype={dtype}"
        if isinstance(out, np.ndarray) and np.issubdtype(out.dtype, np.number) and out.size > 0:
            line += f" min={out.min():.6f} max={out.max():.6f} mean={out.mean():.6f}"
        print(line)


def parse_profile(profile_file, topk=20):
    with open(profile_file, "r") as f:
        prof = json.load(f)

    nodes = [e for e in prof if e.get("cat") == "Node"]
    stat = defaultdict(lambda: [0.0, 0])
    for n in nodes:
        name = n.get("name", "unknown")
        dur = n.get("dur", 0.0)  # us
        stat[name][0] += dur
        stat[name][1] += 1

    avg_stats = []
    for name, (total_dur, cnt) in stat.items():
        avg_ms = (total_dur / cnt) / 1000.0 if cnt else 0.0
        avg_stats.append((name, avg_ms, cnt))

    avg_stats.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {topk} operators by average time:")
    for name, avg_ms, cnt in avg_stats[:topk]:
        print(f"{name:80s}  avg={avg_ms:.3f} ms  runs={cnt}")


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime perf and operator profiling")
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs (default: 2)")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5)")
    parser.add_argument("--topk", type=int, default=20, help="Top-K operators to show (default: 20)")
    args = parser.parse_args()

    # Warmup session without profiling to avoid polluting operator stats
    warm_sess = ort.InferenceSession(args.model)
    feeds = build_dummy_inputs(warm_sess)
    for _ in range(max(args.warmup, 0)):
        warm_sess.run(None, feeds)

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    session = ort.InferenceSession(args.model, sess_options)
    feeds = build_dummy_inputs(session)

    times = []
    last_outputs = None
    for _ in range(max(args.runs, 1)):
        start = time.perf_counter()
        last_outputs = session.run(None, feeds)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    avg_ms = sum(times) / len(times)
    print(f"Avg inference time over {len(times)} runs: {avg_ms:.3f} ms")

    if last_outputs is not None:
        summarize_outputs(last_outputs, session.get_outputs())

    profile_file = session.end_profiling()
    print("Profile saved:", profile_file)
    parse_profile(profile_file, topk=args.topk)


if __name__ == "__main__":
    main()
