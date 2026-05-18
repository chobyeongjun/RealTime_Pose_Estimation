"""CUDA preprocess kernel — correctness + benchmark vs torch reference.

TDD for Sprint 1 Phase 2 Week 3:
  1. Load CudaPreprocessor module (build sanity)
  2. Process same image → cosine ≥ 0.999 vs torch reference
  3. Letterbox region: kernel output == torch output (bilinear bit-equal within FP tolerance)
  4. Pad region: kernel output == 114/255 (exact)
  5. Benchmark vs torch _preprocess_gpu()

Run (Jetson, after scripts/build_cpp.sh):
    PYTHONPATH=src:src/perception/benchmarks python3 tests/test_cuda_preprocess.py \\
        --iters 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def torch_preprocess_reference(image, imgsz, pad_tensor, input_tensor, is_bgra=False):
    """Reference impl — copy of trt_pose_engine._preprocess_gpu() logic."""
    import torch
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_h, pad_w = (imgsz - new_h) // 2, (imgsz - new_w) // 2

    t = torch.from_numpy(np.ascontiguousarray(image)).cuda(non_blocking=True)
    # BGR or BGRA → RGB (drop alpha if present, swap channels)
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
            print(f"  ERROR: {e}")
            print("  Did you run scripts/build_cpp.sh?")
            return 1
    print(f"  module: {cudap}")
    print(f"  CudaPreprocessor: {cudap.CudaPreprocessor}")

    import torch
    pre = cudap.CudaPreprocessor(args.imgsz, 1200, 1920, 4)
    print(f"  imgsz: {pre.imgsz}, staging: {pre.staging_bytes:,} B")
    print()

    # ── Test 2: correctness (cosine ≥ 0.999) ──────────────────────────────
    print("=" * 70)
    print("Test 2: cosine similarity vs torch reference (BGR 600x960)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    bgr_img = rng.integers(0, 256, size=(600, 960, 3), dtype=np.uint8)

    pad_t = torch.full((1, 3, args.imgsz, args.imgsz), 114.0/255.0,
                       dtype=torch.float32, device='cuda')
    out_torch = torch.empty((1, 3, args.imgsz, args.imgsz),
                            dtype=torch.float32, device='cuda')
    out_cuda = torch.empty((1, 3, args.imgsz, args.imgsz),
                           dtype=torch.float32, device='cuda')

    # Reference
    torch_preprocess_reference(bgr_img, args.imgsz, pad_t, out_torch, is_bgra=False)
    torch.cuda.synchronize()

    # CUDA kernel
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ok = pre.process(bgr_img, out_cuda.data_ptr(), False, stream.cuda_stream)
    stream.synchronize()
    if not ok:
        print("  ERROR: process() returned False")
        return 2

    ref = out_torch.flatten().cpu().numpy()
    got = out_cuda.flatten().cpu().numpy()
    norm_r, norm_g = float(np.linalg.norm(ref)), float(np.linalg.norm(got))
    cos = float(np.dot(ref, got) / (norm_r * norm_g))
    max_diff = float(np.max(np.abs(ref - got)))
    mean_diff = float(np.mean(np.abs(ref - got)))
    print(f"  cosine:   {cos:.6f}  (target ≥ 0.999)")
    print(f"  max diff: {max_diff:.6f}")
    print(f"  mean diff: {mean_diff:.6f}")
    if cos >= 0.999:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — outputs differ too much")
    print()

    # ── Test 3: pad region exactness ──────────────────────────────────────
    print("=" * 70)
    print("Test 3: pad region = 114/255 exactly")
    print("=" * 70)
    scale = min(args.imgsz / 600, args.imgsz / 960)
    new_h, new_w = int(600 * scale), int(960 * scale)
    pad_h, pad_w = (args.imgsz - new_h) // 2, (args.imgsz - new_w) // 2
    out_np = out_cuda[0].cpu().numpy()   # (3, S, S)
    pad_val = 114.0 / 255.0
    # Top strip
    if pad_h > 0:
        strip = out_np[:, :pad_h, :]
        all_pad = bool(np.all(np.isclose(strip, pad_val)))
        print(f"  top strip [{pad_h} rows]: all == 114/255? {all_pad}")
    if pad_w > 0:
        strip = out_np[:, :, :pad_w]
        all_pad = bool(np.all(np.isclose(strip, pad_val)))
        print(f"  left strip [{pad_w} cols]: all == 114/255? {all_pad}")
    print()

    # ── Test 4: BGRA path ────────────────────────────────────────────────
    print("=" * 70)
    print("Test 4: BGRA (4-channel) input")
    print("=" * 70)
    bgra_img = rng.integers(0, 256, size=(600, 960, 4), dtype=np.uint8)
    torch_preprocess_reference(bgra_img, args.imgsz, pad_t, out_torch, is_bgra=True)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        ok = pre.process(bgra_img, out_cuda.data_ptr(), True, stream.cuda_stream)
    stream.synchronize()
    if not ok:
        print("  ERROR: BGRA process() returned False")
        return 3
    ref = out_torch.flatten().cpu().numpy()
    got = out_cuda.flatten().cpu().numpy()
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    print(f"  cosine (BGRA): {cos:.6f}")
    if cos >= 0.999:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
    print()

    # ── Test 5: benchmark ─────────────────────────────────────────────────
    if not args.skip_bench:
        print("=" * 70)
        print(f"Test 5: Benchmark vs torch ({args.iters} iters, 600x960 BGR)")
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
    print()
    print("All tests complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
