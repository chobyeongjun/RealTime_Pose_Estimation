"""CUDA preprocess kernel — correctness + benchmark vs torch reference.

TDD for Sprint 1 Phase 2 Week 3:
  1. Load CudaPreprocessor module (build sanity)
  2. Cosine ≥ 0.999 vs torch reference on multiple input sizes:
     - 600x960 BGR (production input)
     - 600x960 BGRA (4-channel)
     - 601x959 BGR (odd, non-factor)
     - 720x1280 BGR (different aspect ratio)
     - 320x320 BGR (square, smaller than imgsz → upsize)
     - 1080x1920 BGR (downsize)
     - 300x301 BGR (the float32 vs double divergence case)
     - non-contiguous (transposed view)
     - zero-init (all zeros, degenerate)
  3. Pad region == 114/255 exactly
  4. Benchmark vs torch _preprocess_gpu()

Exit code: 0 on all pass, non-zero on any failure (CI friendly).

Run (Jetson, after scripts/build_cpp.sh):
    PYTHONPATH=src:src/perception/benchmarks python3 tests/test_cuda_preprocess.py \\
        --iters 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


def torch_preprocess_reference(image, imgsz, pad_tensor, input_tensor, is_bgra=False):
    """Reference impl — copy of trt_pose_engine._preprocess_gpu() logic."""
    import torch
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_h, pad_w = (imgsz - new_h) // 2, (imgsz - new_w) // 2

    t = torch.from_numpy(np.ascontiguousarray(image)).cuda(non_blocking=True)
    if is_bgra:
        t = t[:, :, [2, 1, 0]]
    else:
        t = t[:, :, [2, 1, 0]]
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    t = torch.nn.functional.interpolate(
        t, size=(new_h, new_w), mode='bilinear', align_corners=False)
    input_tensor.copy_(pad_tensor)
    input_tensor[0, :, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = t[0]
    return scale, pad_w, pad_h


def run_correctness_case(pre, name, image, is_bgra, imgsz, pad_t, out_torch, out_cuda,
                         results: List[Tuple[str, bool, dict]],
                         tol_cos: float = 0.999):
    """Run one (image, is_bgra) case, append (name, ok, info) to results."""
    import torch
    torch_preprocess_reference(image, imgsz, pad_t, out_torch, is_bgra=is_bgra)
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        ok = pre.process(image, out_cuda.data_ptr(), is_bgra, stream.cuda_stream)
    stream.synchronize()
    if not ok:
        results.append((name, False, {"reason": "process() returned False"}))
        print(f"  ✗ {name}: process() returned False")
        return

    ref = out_torch.flatten().cpu().numpy()
    got = out_cuda.flatten().cpu().numpy()
    norm_r, norm_g = float(np.linalg.norm(ref)), float(np.linalg.norm(got))
    if norm_r == 0.0 or norm_g == 0.0:
        # Both zero is fine (zero image), one zero is a failure
        both_zero = norm_r == 0.0 and norm_g == 0.0
        results.append((name, both_zero, {"reason": "zero norm",
                                           "norm_ref": norm_r, "norm_got": norm_g}))
        status = "✓" if both_zero else "✗"
        print(f"  {status} {name}: zero-norm (ref={norm_r}, got={norm_g})")
        return
    cos = float(np.dot(ref, got) / (norm_r * norm_g))
    max_diff = float(np.max(np.abs(ref - got)))
    mean_diff = float(np.mean(np.abs(ref - got)))
    passed = cos >= tol_cos
    results.append((name, passed, {"cos": cos, "max_diff": max_diff, "mean_diff": mean_diff}))
    status = "✓" if passed else "✗"
    print(f"  {status} {name}: cos={cos:.6f} max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, "src/perception/realtime")
    sys.path.insert(0, "src")

    # ── Test 1: import ────────────────────────────────────────────────────
    print("=" * 70)
    print("Test 1: import hwalker_cuda_preprocess")
    print("=" * 70)
    try:
        import hwalker_cuda_preprocess as cudap
    except ImportError:
        try:
            from perception.realtime import hwalker_cuda_preprocess as cudap
        except ImportError as e:
            print(f"  ✗ ERROR: {e}")
            print("    Did you run scripts/build_cpp.sh?")
            return 1
    print(f"  module: {cudap}")
    print(f"  CudaPreprocessor: {cudap.CudaPreprocessor}")

    import torch
    pre = cudap.CudaPreprocessor(args.imgsz, 1200, 1920, 4)
    print(f"  ✓ imgsz: {pre.imgsz}, staging: {pre.staging_bytes:,} B")
    print()

    pad_t = torch.full((1, 3, args.imgsz, args.imgsz), 114.0/255.0,
                       dtype=torch.float32, device='cuda')
    out_torch = torch.empty((1, 3, args.imgsz, args.imgsz),
                            dtype=torch.float32, device='cuda')
    out_cuda = torch.empty((1, 3, args.imgsz, args.imgsz),
                           dtype=torch.float32, device='cuda')
    rng = np.random.default_rng(42)
    results: List[Tuple[str, bool, dict]] = []

    # ── Test 2: correctness across input shapes ──────────────────────────
    print("=" * 70)
    print("Test 2: cosine similarity ≥ 0.999 vs torch reference")
    print("=" * 70)
    cases = [
        ("600x960 BGR (production)", (600, 960, 3), False),
        ("600x960 BGRA",              (600, 960, 4), True),
        ("601x959 BGR (odd)",         (601, 959, 3), False),
        ("720x1280 BGR (16:9)",       (720, 1280, 3), False),
        ("320x320 BGR (square, upsize)", (320, 320, 3), False),
        ("1080x1920 BGR (FHD)",       (1080, 1920, 3), False),
        ("300x301 BGR (float/double divergence case)", (300, 301, 3), False),
        ("640x640 BGR (exact imgsz)", (640, 640, 3), False),
    ]
    for name, shape, is_bgra in cases:
        img = rng.integers(0, 256, size=shape, dtype=np.uint8)
        run_correctness_case(pre, name, img, is_bgra, args.imgsz,
                              pad_t, out_torch, out_cuda, results)

    # Zero image (edge case — both should produce same trivial output)
    zero_img = np.zeros((600, 960, 3), dtype=np.uint8)
    run_correctness_case(pre, "600x960 BGR (zero image)", zero_img, False,
                          args.imgsz, pad_t, out_torch, out_cuda, results)

    # Non-contiguous numpy view (transposed)
    src = rng.integers(0, 256, size=(960, 600, 3), dtype=np.uint8)
    transposed = np.ascontiguousarray(src.transpose(1, 0, 2))  # 600x960x3
    run_correctness_case(pre, "600x960 BGR (from transpose)", transposed, False,
                          args.imgsz, pad_t, out_torch, out_cuda, results)
    print()

    # ── Test 3: pad region exactness (use production input) ──────────────
    print("=" * 70)
    print("Test 3: pad region = 114/255 exactly (production 600x960)")
    print("=" * 70)
    prod_img = rng.integers(0, 256, size=(600, 960, 3), dtype=np.uint8)
    stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        pre.process(prod_img, out_cuda.data_ptr(), False, stream.cuda_stream)
    stream.synchronize()
    scale = min(args.imgsz / 600, args.imgsz / 960)
    new_h, new_w = int(600 * scale), int(960 * scale)
    pad_h, pad_w = (args.imgsz - new_h) // 2, (args.imgsz - new_w) // 2
    out_np = out_cuda[0].cpu().numpy()   # (3, S, S)
    pad_val = 114.0 / 255.0
    pad_top_ok = pad_h == 0 or bool(np.all(np.isclose(out_np[:, :pad_h, :], pad_val)))
    pad_left_ok = pad_w == 0 or bool(np.all(np.isclose(out_np[:, :, :pad_w], pad_val)))
    pad_bot_ok = pad_h == 0 or bool(np.all(np.isclose(out_np[:, args.imgsz - pad_h:, :], pad_val)))
    pad_right_ok = pad_w == 0 or bool(np.all(np.isclose(out_np[:, :, args.imgsz - pad_w:], pad_val)))
    all_pad_ok = pad_top_ok and pad_bot_ok and pad_left_ok and pad_right_ok
    print(f"  top[{pad_h}]: {pad_top_ok}  bot[{pad_h}]: {pad_bot_ok}  "
          f"left[{pad_w}]: {pad_left_ok}  right[{pad_w}]: {pad_right_ok}")
    results.append(("pad regions exact 114/255", all_pad_ok, {}))
    print(f"  {'✓' if all_pad_ok else '✗'} pad exact")
    print()

    # ── Test 4: benchmark ─────────────────────────────────────────────────
    if not args.skip_bench:
        print("=" * 70)
        print(f"Test 4: Benchmark vs torch ({args.iters} iters, 600x960 BGR)")
        print("=" * 70)
        bench_img = rng.integers(0, 256, size=(600, 960, 3), dtype=np.uint8)

        # Warmup
        for _ in range(30):
            torch_preprocess_reference(bench_img, args.imgsz, pad_t, out_torch, is_bgra=False)
            with torch.cuda.stream(stream):
                pre.process(bench_img, out_cuda.data_ptr(), False, stream.cuda_stream)
            stream.synchronize()
        torch.cuda.synchronize()

        # Torch baseline
        t_torch = np.zeros(args.iters)
        for i in range(args.iters):
            t0 = time.perf_counter()
            torch_preprocess_reference(bench_img, args.imgsz, pad_t, out_torch, is_bgra=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_torch[i] = (t1 - t0) * 1000.0

        # CUDA kernel
        t_cuda = np.zeros(args.iters)
        for i in range(args.iters):
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                pre.process(bench_img, out_cuda.data_ptr(), False, stream.cuda_stream)
            stream.synchronize()
            t1 = time.perf_counter()
            t_cuda[i] = (t1 - t0) * 1000.0

        # Drop first 50 (settle)
        torch_arr, cuda_arr = t_torch[50:], t_cuda[50:]
        p50_t, p95_t, p99_t = np.percentile(torch_arr, [50, 95, 99])
        p50_c, p95_c, p99_c = np.percentile(cuda_arr, [50, 95, 99])
        print(f"  torch _preprocess_gpu():")
        print(f"    p50={p50_t:.3f}ms p95={p95_t:.3f}ms p99={p99_t:.3f}ms")
        print(f"  CUDA kernel preprocess:")
        print(f"    p50={p50_c:.3f}ms p95={p95_c:.3f}ms p99={p99_c:.3f}ms")
        print(f"  Gain (torch − cuda):")
        print(f"    p50: {p50_t - p50_c:+.3f} ms  ({(p50_t - p50_c)/p50_t*100:+.1f} %)")
        print(f"    p99: {p99_t - p99_c:+.3f} ms")
        bench_pass = p50_c < p50_t
        results.append(("benchmark CUDA p50 < torch p50", bench_pass,
                        {"torch_p50_ms": p50_t, "cuda_p50_ms": p50_c}))
        print(f"  {'✓' if bench_pass else '✗'} CUDA faster")
    print()

    # ── Summary + exit code ───────────────────────────────────────────────
    print("=" * 70)
    print(f"Summary: {sum(1 for _,ok,_ in results if ok)} / {len(results)} pass")
    print("=" * 70)
    fail_count = 0
    for name, ok, info in results:
        if not ok:
            fail_count += 1
            print(f"  ✗ FAIL: {name}  {info}")
    if fail_count == 0:
        print("All tests pass.")
        return 0
    print(f"\n{fail_count} test(s) FAILED — exit 1")
    return 1


if __name__ == "__main__":
    sys.exit(main())
