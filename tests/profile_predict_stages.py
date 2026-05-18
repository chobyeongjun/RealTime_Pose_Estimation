"""Measure predict() stage breakdown: preprocess / infer / postprocess.

Premise validation for Sprint 1 Phase 2 Week 3 (CUDA preprocess kernel):
  If preprocess_ms > 1.0 ms → custom CUDA kernel ROI 충분
  If preprocess_ms < 0.5 ms → kernel 작업 abort, Sprint 2 EKF 로 pivot

Run (Jetson):
    PYTHONPATH=src:src/perception/benchmarks python3 tests/profile_predict_stages.py \\
        --engine src/perception/models/yolo26s-lower6-v2-640.engine \\
        --iters 300

Output: p50/p95/p99 for each stage, breakdown summary.
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
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not Path(args.engine).is_file():
        print(f"ERROR: engine not found: {args.engine}")
        return 1

    sys.path.insert(0, "src/perception/benchmarks")
    sys.path.insert(0, "src")

    from trt_pose_engine import TRTPoseEngine
    import torch

    print("=" * 70)
    print("Profile predict() stage breakdown")
    print("=" * 70)
    eng = TRTPoseEngine(args.engine, imgsz=args.imgsz, use_cpp_trt=False)
    eng.load()

    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(600, 960, 3), dtype=np.uint8)

    # Warmup (no profiling — production hot path)
    for _ in range(30):
        eng.predict(img)
    torch.cuda.synchronize()

    pre_ms = np.zeros(args.iters)
    inf_ms = np.zeros(args.iters)
    post_ms = np.zeros(args.iters)
    e2e_ms = np.zeros(args.iters)

    for i in range(args.iters):
        prof = {}
        t0 = time.perf_counter()
        eng.predict(img, _profile=prof)
        t1 = time.perf_counter()
        pre_ms[i] = prof.get('preprocess_ms', 0.0)
        inf_ms[i] = prof.get('infer_ms', 0.0)
        post_ms[i] = prof.get('postprocess_ms', 0.0)
        e2e_ms[i] = (t1 - t0) * 1000.0

    # Drop first 50 (settle)
    pre, inf, post, e2e = pre_ms[50:], inf_ms[50:], post_ms[50:], e2e_ms[50:]

    print()
    print(f"  Sample size: {len(pre)} iters (after warmup)")
    print()
    print(f"  {'stage':<20} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, arr in [('preprocess', pre), ('infer', inf), ('postprocess', post), ('e2e (sum)', e2e)]:
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        mx = arr.max()
        print(f"  {name:<20} {p50:>7.3f}m {p95:>7.3f}m {p99:>7.3f}m {mx:>7.3f}m")
    print()
    print(f"  preprocess p50 = {np.percentile(pre, 50):.3f} ms")
    print()

    # Decision
    pre_p50 = float(np.percentile(pre, 50))
    print("=" * 70)
    print("Week 3 (CUDA preprocess kernel) ROI verdict")
    print("=" * 70)
    if pre_p50 >= 1.0:
        print(f"  ✓ preprocess {pre_p50:.3f} ms ≥ 1.0 ms — kernel ROI 충분")
        print(f"    expected gain (kernel halves preprocess): ~{pre_p50*0.5:.2f} ms")
    elif pre_p50 >= 0.5:
        print(f"  ⚠ preprocess {pre_p50:.3f} ms ∈ [0.5, 1.0) — marginal")
        print(f"    consider opportunity cost vs Sprint 2 EKF")
    else:
        print(f"  ✗ preprocess {pre_p50:.3f} ms < 0.5 ms — kernel ROI 없음")
        print(f"    ABORT Week 3, pivot to Sprint 2 EKF iteration")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
