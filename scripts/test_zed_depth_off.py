#!/usr/bin/env python3
"""ZED depth_mode=NONE smoke test — Sprint 1 의 E path 의 진짜 gain 측정.

Test 3 (prior) 가 dense depth marginal cost 1.36ms 만 발견. 의문:
  - ZED SDK 의 grab() 가 *내부* 에 dense depth compute 포함 가능
  - depth_mode=NONE 으로 open 시 grab time 줄어드는가?

이 script:
  1. depth_mode=PERFORMANCE: grab() time 측정 (baseline)
  2. depth_mode=NONE:        grab() time 측정 (no depth compute)
  3. depth_mode=NONE + LEFT_UNRECTIFIED + RIGHT_UNRECTIFIED retrieve
     (sparse stereo 의 진짜 input)
  4. depth_mode=NONE + LEFT_UNRECTIFIED_GRAY + RIGHT_UNRECTIFIED_GRAY
     (가장 raw + lightweight)

→ depth_mode=NONE 으로 grab() 가 1-2ms 로 줄어들면 E path 진짜 gain = 4-5ms.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed.sl not available. Run on Jetson.", file=sys.stderr)
    sys.exit(1)


def measure_mode(
    init_params: sl.InitParameters,
    label: str,
    retrieve_views: list,   # [(view, mem), ...]
    n_frames: int = 100,
) -> dict:
    """Open camera with given init params, measure grab() + retrieve timing."""
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"  {label}: OPEN FAILED: {err}")
        return {}

    rt = sl.RuntimeParameters()
    mats = [sl.Mat() for _ in retrieve_views]

    grab_times = []
    retrieve_times = []
    total_times = []
    success = 0
    eof = False

    for _ in range(n_frames):
        # Measure grab
        t0 = time.perf_counter()
        err_g = zed.grab(rt)
        t1 = time.perf_counter()

        if err_g == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            eof = True
            break
        if err_g != sl.ERROR_CODE.SUCCESS:
            continue

        # Measure retrieves
        t_retrieve_start = time.perf_counter()
        for (view, mem), mat in zip(retrieve_views, mats):
            zed.retrieve_image(mat, view, mem)
        t2 = time.perf_counter()

        grab_times.append((t1 - t0) * 1000.0)
        retrieve_times.append((t2 - t_retrieve_start) * 1000.0)
        total_times.append((t2 - t0) * 1000.0)
        success += 1

    zed.close()

    if not grab_times:
        return {'label': label, 'error': 'no frames captured'}

    grab_arr = np.array(grab_times)
    retr_arr = np.array(retrieve_times)
    total_arr = np.array(total_times)

    result = {
        'label': label,
        'n': success,
        'grab_p50': float(np.percentile(grab_arr, 50)),
        'grab_p99': float(np.percentile(grab_arr, 99)),
        'retrieve_p50': float(np.percentile(retr_arr, 50)),
        'retrieve_p99': float(np.percentile(retr_arr, 99)),
        'total_p50': float(np.percentile(total_arr, 50)),
        'total_p99': float(np.percentile(total_arr, 99)),
    }
    print(f"  {label}:")
    print(f"     grab() :           p50={result['grab_p50']:6.3f}ms  p99={result['grab_p99']:6.3f}ms")
    print(f"     retrieve(s):       p50={result['retrieve_p50']:6.3f}ms  p99={result['retrieve_p99']:6.3f}ms")
    print(f"     total (grab+retr): p50={result['total_p50']:6.3f}ms  p99={result['total_p99']:6.3f}ms")
    print(f"     N={success}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--svo', default=None, help='Replay from SVO')
    ap.add_argument('--frames', type=int, default=100)
    args = ap.parse_args()

    # Check DEPTH_MODE.NONE 가능 여부
    print("=" * 70)
    print("Available DEPTH_MODE options:")
    print("=" * 70)
    modes = [m for m in dir(sl.DEPTH_MODE) if not m.startswith('_') and m.isupper()]
    for m in modes:
        print(f"  {m}")
    has_none = 'NONE' in modes
    print(f"\n  DEPTH_MODE.NONE available: {'✓' if has_none else '✗'}")
    print()

    if not has_none:
        print("ERROR: DEPTH_MODE.NONE not available. Cannot test depth-off mode.")
        return 1

    # Build init params 공통 부분
    def make_init(depth_mode):
        init = sl.InitParameters()
        if args.svo:
            init.set_from_svo_file(args.svo)
            init.svo_real_time_mode = False
        else:
            init.camera_resolution = sl.RESOLUTION.SVGA
            init.camera_fps = 120
        init.depth_mode = depth_mode
        init.coordinate_units = sl.UNIT.METER
        return init

    print(f"Source: {'SVO ' + args.svo if args.svo else 'Live SVGA@120'}")
    print(f"Frames per test: {args.frames}")
    print()

    # ─── Test 1: PERFORMANCE depth (baseline) ────────────────────────────
    print("=" * 70)
    print("Test A: PERFORMANCE depth (baseline) + RGB GPU retrieve")
    print("=" * 70)
    r1 = measure_mode(
        make_init(sl.DEPTH_MODE.PERFORMANCE),
        'PERFORMANCE + RGB GPU',
        [(sl.VIEW.LEFT, sl.MEM.GPU)],
        n_frames=args.frames,
    )
    print()

    # ─── Test 2: NONE depth (no compute) ──────────────────────────────────
    print("=" * 70)
    print("Test B: NONE depth + RGB GPU retrieve")
    print("=" * 70)
    r2 = measure_mode(
        make_init(sl.DEPTH_MODE.NONE),
        'NONE + RGB GPU',
        [(sl.VIEW.LEFT, sl.MEM.GPU)],
        n_frames=args.frames,
    )
    print()

    # ─── Test 3: NONE depth + LEFT_UNRECTIFIED + RIGHT_UNRECTIFIED ───────
    print("=" * 70)
    print("Test C: NONE depth + LEFT_UNRECTIFIED + RIGHT_UNRECTIFIED GPU")
    print("=" * 70)
    r3 = measure_mode(
        make_init(sl.DEPTH_MODE.NONE),
        'NONE + L+R UNRECTIFIED GPU',
        [(sl.VIEW.LEFT_UNRECTIFIED, sl.MEM.GPU),
         (sl.VIEW.RIGHT_UNRECTIFIED, sl.MEM.GPU)],
        n_frames=args.frames,
    )
    print()

    # ─── Test 4: NONE + LEFT_UNRECTIFIED_GRAY + RIGHT_UNRECTIFIED_GRAY ───
    print("=" * 70)
    print("Test D: NONE depth + LEFT_UNRECTIFIED_GRAY + RIGHT_UNRECTIFIED_GRAY GPU")
    print("=" * 70)
    r4 = measure_mode(
        make_init(sl.DEPTH_MODE.NONE),
        'NONE + L+R UNRECTIFIED_GRAY GPU',
        [(sl.VIEW.LEFT_UNRECTIFIED_GRAY, sl.MEM.GPU),
         (sl.VIEW.RIGHT_UNRECTIFIED_GRAY, sl.MEM.GPU)],
        n_frames=args.frames,
    )
    print()

    # ─── Comparison ───────────────────────────────────────────────────────
    print("=" * 70)
    print("Summary — E path 의 진짜 gain 분석")
    print("=" * 70)
    if r1 and r2 and r3 and r4:
        baseline = r1['total_p50']
        no_depth = r2['total_p50']
        e_path_rectified = r3['total_p50']
        e_path_raw = r4['total_p50']

        print(f"  Baseline (PERFORMANCE + RGB):      {baseline:6.3f}ms")
        print(f"  No depth (NONE + RGB):             {no_depth:6.3f}ms  Δ={no_depth-baseline:+.3f}ms")
        print(f"  E path rect (NONE + L+R rect):     {e_path_rectified:6.3f}ms  Δ={e_path_rectified-baseline:+.3f}ms")
        print(f"  E path raw (NONE + L+R unrect):    {e_path_raw:6.3f}ms  Δ={e_path_raw-baseline:+.3f}ms")
        print()
        print(f"  Depth compute 시간 (baseline - no_depth): {baseline - no_depth:+.3f}ms")
        print(f"  → 만약 양수 = depth compute 가 grab() 안 에 있음 → E path 가 그만큼 gain")
        print(f"  → 만약 음수 또는 0 = SDK 가 depth 안 compute 함 → E path gain limited to 1.36ms")
        print()
        print(f"  E path 의 최대 gain (baseline - e_path_raw): {baseline - e_path_raw:+.3f}ms")
    else:
        print("  Some tests failed — see above")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
