"""C++ TRT runner regression + benchmark vs Python TRT path.

Tests:
  1. Engine load (.engine file) + introspection
  2. Single inference: output shape match
  3. Benchmark: 1000 iters, p50/p99/max — compare to Python TRT API call

Run on Jetson (after scripts/build_cpp.sh):
    PYTHONPATH=src python3 tests/test_trt_runner_cpp.py \\
        --engine src/perception/CUDA_Stream/yolo26s-lower6-v2.engine

Optionally compare to Python TRT (TRTRunner from CUDA_Stream):
    PYTHONPATH=src python3 tests/test_trt_runner_cpp.py \\
        --engine path/to/engine --compare-python
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help=".engine file path")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--compare-python", action="store_true",
                    help="Also benchmark Python TRT (CUDA_Stream/trt_runner.py)")
    args = ap.parse_args()

    if not Path(args.engine).is_file():
        print(f"ERROR: engine not found: {args.engine}")
        return 1

    # Try import C++ extension
    try:
        from perception.realtime import hwalker_trt_runner as cpp_trt
    except ImportError as e:
        print(f"ERROR: hwalker_trt_runner import failed: {e}")
        print("Run scripts/build_cpp.sh on Jetson first.")
        return 2

    try:
        import torch
    except ImportError:
        print("ERROR: torch required (PyTorch GPU tensors as buffers)")
        return 3

    print("=" * 70)
    print("Test 1: C++ TrtRunner load + introspection")
    print("=" * 70)
    runner = cpp_trt.TrtRunner(args.engine)
    print(f"  engine_path:  {runner.engine_path}")
    print(f"  input_name:   {runner.input_name}")
    print(f"  output_name:  {runner.output_name}")
    print(f"  input_shape:  {runner.input_shape}")
    print(f"  output_shape: {runner.output_shape}")
    print(f"  input_bytes:  {runner.input_bytes}")
    print(f"  output_bytes: {runner.output_bytes}")
    print("  ✓ Engine loaded")
    print()

    # Allocate GPU buffers (torch)
    device = torch.device("cuda:0")
    # Use known shape; fp16 (half) for TRT FP16 engine
    in_shape = tuple(s if s > 0 else 1 for s in runner.input_shape)
    out_shape = tuple(s if s > 0 else 1 for s in runner.output_shape)

    # Engine dtype detection: bytes / element_count
    in_count = 1
    for d in in_shape:
        in_count *= d
    in_dtype_size = runner.input_bytes // max(in_count, 1)
    in_dtype = torch.float16 if in_dtype_size == 2 else torch.float32

    out_count = 1
    for d in out_shape:
        out_count *= d
    out_dtype_size = runner.output_bytes // max(out_count, 1)
    out_dtype = torch.float16 if out_dtype_size == 2 else torch.float32

    input_tensor = torch.zeros(in_shape, dtype=in_dtype, device=device).contiguous()
    output_tensor = torch.zeros(out_shape, dtype=out_dtype, device=device).contiguous()

    print("=" * 70)
    print("Test 2: Single inference (sync) — output shape check")
    print("=" * 70)
    ok = runner.infer_sync(input_tensor.data_ptr(), output_tensor.data_ptr())
    print(f"  infer_sync: {'OK' if ok else 'FAIL'}")
    print(f"  output finite: {torch.isfinite(output_tensor.float()).all().item()}")
    print(f"  output shape: {tuple(output_tensor.shape)}")
    print()

    print("=" * 70)
    print(f"Test 3: Benchmark ({args.iters} iterations)")
    print("=" * 70)
    # Warmup
    for _ in range(50):
        runner.infer(input_tensor.data_ptr(), output_tensor.data_ptr(), 0)
    torch.cuda.synchronize()

    # Async benchmark (closer to production: enqueue + sync once at end)
    times_async_us = np.zeros(args.iters, dtype=np.float64)
    for i in range(args.iters):
        t0 = time.perf_counter_ns()
        runner.infer(input_tensor.data_ptr(), output_tensor.data_ptr(), 0)
        torch.cuda.synchronize()  # measure full latency
        t1 = time.perf_counter_ns()
        times_async_us[i] = (t1 - t0) / 1000.0

    arr = times_async_us[200:]  # skip warmup
    def _stats(a):
        return (float(np.percentile(a, 50)), float(np.percentile(a, 95)),
                float(np.percentile(a, 99)), float(a.max()), float(a.mean()))
    p50, p95, p99, mx, mn = _stats(arr)
    print(f"  C++ TrtRunner.infer + synchronize:")
    print(f"    p50:  {p50:7.2f} us = {p50/1000.0:.3f} ms")
    print(f"    p95:  {p95:7.2f} us = {p95/1000.0:.3f} ms")
    print(f"    p99:  {p99:7.2f} us = {p99/1000.0:.3f} ms")
    print(f"    max:  {mx:7.2f} us")
    print(f"    mean: {mn:7.2f} us")
    print()

    # Optional: compare to Python TRT
    if args.compare_python:
        print("=" * 70)
        print("Test 4: Python TRT (CUDA_Stream/trt_runner.py) — comparison")
        print("=" * 70)
        try:
            sys.path.insert(0, "src")
            from perception.CUDA_Stream.trt_runner import TRTRunner as PyTRT
            py_runner = PyTRT(args.engine)

            # Use same input/output tensors
            in_dict = {py_runner.input_names[0]: input_tensor}
            out_dict = {py_runner.output_names[0]: output_tensor}

            for _ in range(50):
                py_runner.run(in_dict, out_dict)
            torch.cuda.synchronize()

            py_times_us = np.zeros(args.iters, dtype=np.float64)
            for i in range(args.iters):
                t0 = time.perf_counter_ns()
                py_runner.run(in_dict, out_dict)
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                py_times_us[i] = (t1 - t0) / 1000.0

            arr_py = py_times_us[200:]
            p50p, p95p, p99p, mxp, mnp = _stats(arr_py)
            print(f"  Python TRTRunner.run + synchronize:")
            print(f"    p50:  {p50p:7.2f} us = {p50p/1000.0:.3f} ms")
            print(f"    p95:  {p95p:7.2f} us = {p95p/1000.0:.3f} ms")
            print(f"    p99:  {p99p:7.2f} us = {p99p/1000.0:.3f} ms")
            print(f"    max:  {mxp:7.2f} us")
            print()
            print(f"  Gain (Python − C++) p50: {p50p - p50:+.2f} us = {(p50p - p50)/1000.0:+.3f} ms")
            print(f"  Gain (Python − C++) p99: {p99p - p99:+.2f} us = {(p99p - p99)/1000.0:+.3f} ms")
        except Exception as e:
            print(f"  Python comparison skipped: {e}")
        print()

    print("=" * 70)
    print("All tests complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
