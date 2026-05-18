"""TRTPoseEngine with use_cpp_trt=True — regression + benchmark vs Python TRT.

TDD for Sprint 1 Phase 2 Week 2: pipeline integration of C++ TrtRunner.

Tests:
  1. Load with use_cpp_trt=True — engine + adapter init OK
  2. predict() output shape + finite check
  3. Output value consistency: Python TRT vs C++ TRT on same image (synth) — Cosine ≥ 0.999
  4. Benchmark predict() E2E (preprocess + TRT + postprocess) — Python vs C++

Run on Jetson (after build_cpp.sh + commit 5b59265+):
    PYTHONPATH=src:src/perception/benchmarks python3 tests/test_trt_pose_engine_cpp.py \\
        --engine src/perception/CUDA_Stream/yolo26s-lower6-v2.engine

Note: Track A 의 production engine = src/perception/models/yolo26s-lower6-v2-640.engine
      Track B 의 production engine = src/perception/CUDA_Stream/yolo26s-lower6-v2.engine
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-py-compare", action="store_true",
                    help="Skip Python TRT comparison (faster, only C++ benchmark)")
    args = ap.parse_args()

    if not Path(args.engine).is_file():
        print(f"ERROR: engine not found: {args.engine}")
        return 1

    # Allow benchmarks path
    sys.path.insert(0, "src/perception/benchmarks")
    sys.path.insert(0, "src")

    try:
        from trt_pose_engine import TRTPoseEngine
    except ImportError as e:
        print(f"ERROR: TRTPoseEngine import failed: {e}")
        return 2

    try:
        import torch
    except ImportError:
        print("ERROR: torch required")
        return 3

    # ──────────────────────────────────────────────────────────────────────
    # Test 1: C++ adapter load
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Test 1: TRTPoseEngine(use_cpp_trt=True) load")
    print("=" * 70)
    eng_cpp = TRTPoseEngine(args.engine, imgsz=args.imgsz, use_cpp_trt=True)
    eng_cpp.load()
    print("  ✓ C++ adapter loaded")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # Test 2: predict() shape + finite
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Test 2: predict() shape + finite check")
    print("=" * 70)
    rng = np.random.default_rng(42)
    test_img = rng.integers(0, 256, size=(600, 960, 3), dtype=np.uint8)
    result_cpp = eng_cpp.predict(test_img)
    print(f"  C++ result type: {type(result_cpp).__name__}")
    # PoseResult is a dataclass-like object; introspect generically
    if hasattr(result_cpp, '__dict__'):
        fields = list(vars(result_cpp).keys())[:10]
        print(f"  result fields (first 10): {fields}")
    if hasattr(result_cpp, 'detected'):
        print(f"  detected: {result_cpp.detected}")
    elif hasattr(result_cpp, 'valid'):
        print(f"  valid: {result_cpp.valid}")
    # Output tensor (pre-allocated, written by infer)
    out = eng_cpp._output_tensor
    print(f"  output_tensor shape: {tuple(out.shape)}")
    print(f"  output_tensor finite: {torch.isfinite(out).all().item()}")
    print(f"  output_tensor mean abs: {out.abs().mean().item():.4f}")
    print("  ✓ predict() ran OK")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # Test 3: Output value consistency Python vs C++
    # ──────────────────────────────────────────────────────────────────────
    if not args.skip_py_compare:
        print("=" * 70)
        print("Test 3: Output consistency (Python TRT vs C++ TRT)")
        print("=" * 70)
        eng_py = TRTPoseEngine(args.engine, imgsz=args.imgsz, use_cpp_trt=False)
        eng_py.load()

        # Capture raw output_tensor (before postprocess)
        # Run same image through both, compare _output_tensor
        eng_py.predict(test_img)
        out_py = eng_py._output_tensor.detach().cpu().numpy().flatten()
        eng_cpp.predict(test_img)
        out_cpp = eng_cpp._output_tensor.detach().cpu().numpy().flatten()

        # Cosine similarity (both vectors)
        norm_py = np.linalg.norm(out_py)
        norm_cpp = np.linalg.norm(out_cpp)
        if norm_py > 0 and norm_cpp > 0:
            cos = float(np.dot(out_py, out_cpp) / (norm_py * norm_cpp))
            print(f"  Output cosine similarity: {cos:.6f}")
            print(f"  Max abs diff:             {np.max(np.abs(out_py - out_cpp)):.6f}")
            print(f"  Mean abs diff:            {np.mean(np.abs(out_py - out_cpp)):.6f}")
            if cos >= 0.999:
                print(f"  ✓ Consistency PASS (cos ≥ 0.999)")
            else:
                print(f"  ⚠ Consistency LOW — outputs differ")
        else:
            print("  ⚠ Cannot compare — one or both outputs zero norm")
        print()

    # ──────────────────────────────────────────────────────────────────────
    # Test 4: Benchmark E2E predict() — Python vs C++
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Test 4: Benchmark predict() E2E ({args.iters} iters)")
    print("=" * 70)

    # Use a realistic walking image (synthetic)
    bench_img = rng.integers(0, 256, size=(600, 960, 3), dtype=np.uint8)

    # Warmup C++
    for _ in range(20):
        eng_cpp.predict(bench_img)
    torch.cuda.synchronize()

    times_cpp_us = np.zeros(args.iters, dtype=np.float64)
    for i in range(args.iters):
        t0 = time.perf_counter_ns()
        eng_cpp.predict(bench_img)
        t1 = time.perf_counter_ns()
        times_cpp_us[i] = (t1 - t0) / 1000.0

    arr_cpp = times_cpp_us[100:]
    p50_c, p95_c, p99_c, max_c = (
        float(np.percentile(arr_cpp, 50)),
        float(np.percentile(arr_cpp, 95)),
        float(np.percentile(arr_cpp, 99)),
        float(arr_cpp.max()),
    )

    print(f"  C++ TRT path (preprocess + TRT + postprocess):")
    print(f"    p50:  {p50_c:7.2f} us = {p50_c/1000.0:.3f} ms")
    print(f"    p95:  {p95_c:7.2f} us = {p95_c/1000.0:.3f} ms")
    print(f"    p99:  {p99_c:7.2f} us = {p99_c/1000.0:.3f} ms")
    print(f"    max:  {max_c:7.2f} us")
    print()

    if not args.skip_py_compare:
        # Python TRT benchmark
        for _ in range(20):
            eng_py.predict(bench_img)
        torch.cuda.synchronize()

        times_py_us = np.zeros(args.iters, dtype=np.float64)
        for i in range(args.iters):
            t0 = time.perf_counter_ns()
            eng_py.predict(bench_img)
            t1 = time.perf_counter_ns()
            times_py_us[i] = (t1 - t0) / 1000.0

        arr_py = times_py_us[100:]
        p50_p, p95_p, p99_p, max_p = (
            float(np.percentile(arr_py, 50)),
            float(np.percentile(arr_py, 95)),
            float(np.percentile(arr_py, 99)),
            float(arr_py.max()),
        )
        print(f"  Python TRT path:")
        print(f"    p50:  {p50_p:7.2f} us = {p50_p/1000.0:.3f} ms")
        print(f"    p95:  {p95_p:7.2f} us = {p95_p/1000.0:.3f} ms")
        print(f"    p99:  {p99_p:7.2f} us = {p99_p/1000.0:.3f} ms")
        print(f"    max:  {max_p:7.2f} us")
        print()
        print(f"  Gain (Python − C++):")
        print(f"    p50: {(p50_p - p50_c):+.2f} us = {(p50_p - p50_c)/1000.0:+.3f} ms")
        print(f"    p99: {(p99_p - p99_c):+.2f} us = {(p99_p - p99_c)/1000.0:+.3f} ms")

    print()
    print("All tests complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
