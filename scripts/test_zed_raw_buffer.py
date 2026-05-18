#!/usr/bin/env python3
"""ZED SDK RawBuffer + GPU API smoke test — Sprint 1 의 E path go/no-go gate.

Tests:
  1. sl.VIEW.* 의 사용 가능 한 view 목록 (LEFT_GRAY, RIGHT_GRAY 등)
  2. retrieve_image with MEM.GPU 작동 + timing (RGB, LEFT_GRAY, RIGHT_GRAY)
  3. Dense depth retrieve 의 cost = E path 가 제거 가능 한 latency

Run on Jetson:
    cd ~/realtime-vision-control
    python3 scripts/test_zed_raw_buffer.py                  # live camera SVGA@120
    python3 scripts/test_zed_raw_buffer.py --svo recordings/walking_20260518_115340/walking_20260518_115340.svo2
    python3 scripts/test_zed_raw_buffer.py --frames 200     # 더 긴 측정

Output:
    - sl.VIEW.* 의 모든 attribute list
    - 각 (view, mem) 조합 의 timing (p50, p99)
    - Dense depth 의 marginal cost (E path 가 제거 가능)
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
    print("ERROR: pyzed.sl not available. Run on Jetson with ZED SDK installed.", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: sl.VIEW.* attributes
# ─────────────────────────────────────────────────────────────────────────────

def list_view_options() -> list[str]:
    """List all sl.VIEW.* attributes + check E path requirements."""
    print("=" * 70)
    print("Test 1: sl.VIEW.* attributes")
    print("=" * 70)
    views = [v for v in dir(sl.VIEW) if not v.startswith('_') and v.isupper()]
    print("All available views:")
    for v in sorted(views):
        print(f"  {v}")

    print("\nE path (sparse stereo) requirements:")
    needed = ['LEFT_GRAY', 'RIGHT_GRAY',
              'LEFT_UNRECTIFIED', 'RIGHT_UNRECTIFIED',
              'LEFT_UNRECTIFIED_GRAY', 'RIGHT_UNRECTIFIED_GRAY']
    for v in needed:
        has = v in views
        mark = '✓' if has else '✗ NOT AVAILABLE'
        print(f"  {v}: {mark}")
    print()
    return views


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: retrieve_image API matrix
# ─────────────────────────────────────────────────────────────────────────────

def _get_view_attr(name: str):
    return getattr(sl.VIEW, name, None)


def test_retrieve_apis(zed, n_frames: int = 100, available_views: list[str] = None):
    """Test each (view, MEM) combination."""
    print("=" * 70)
    print(f"Test 2: retrieve_image API matrix ({n_frames} frames per config)")
    print("=" * 70)
    rt = sl.RuntimeParameters()

    configs = [
        ('LEFT (RGB) CPU',    'LEFT',         sl.MEM.CPU),
        ('LEFT (RGB) GPU',    'LEFT',         sl.MEM.GPU),
        ('LEFT_GRAY CPU',     'LEFT_GRAY',    sl.MEM.CPU),
        ('LEFT_GRAY GPU',     'LEFT_GRAY',    sl.MEM.GPU),
        ('RIGHT (RGB) GPU',   'RIGHT',        sl.MEM.GPU),
        ('RIGHT_GRAY GPU',    'RIGHT_GRAY',   sl.MEM.GPU),
    ]

    results = {}
    for label, view_name, mem in configs:
        view = _get_view_attr(view_name)
        if view is None or (available_views and view_name not in available_views):
            results[label] = ('SKIP (view N/A)', None)
            continue

        m = sl.Mat()
        times_ms = []
        success = 0
        eof = False
        for i in range(n_frames):
            err_g = zed.grab(rt)
            if err_g == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                eof = True
                break
            if err_g != sl.ERROR_CODE.SUCCESS:
                continue
            t0 = time.perf_counter()
            err_r = zed.retrieve_image(m, view, mem)
            t1 = time.perf_counter()
            if err_r == sl.ERROR_CODE.SUCCESS:
                success += 1
                times_ms.append((t1 - t0) * 1000.0)

        if times_ms:
            arr = np.array(times_ms)
            results[label] = ('OK', {
                'count': success,
                'mean':  float(arr.mean()),
                'p50':   float(np.percentile(arr, 50)),
                'p99':   float(np.percentile(arr, 99)),
                'max':   float(arr.max()),
            })
        else:
            results[label] = ('FAIL', None)

        if eof:
            break

    for label, (status, stats) in results.items():
        if stats:
            print(f"  {label:25s}: {status}  "
                  f"p50={stats['p50']:6.3f}ms  p99={stats['p99']:6.3f}ms  "
                  f"max={stats['max']:6.3f}ms  N={stats['count']}")
        else:
            print(f"  {label:25s}: {status}")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Dense depth marginal cost
# ─────────────────────────────────────────────────────────────────────────────

def test_depth_cost(zed, init_params, n_frames: int = 100, svo_path: Optional[str] = None):
    """Measure marginal cost of retrieve_measure DEPTH (E path 가 제거 가능)."""
    print("=" * 70)
    print(f"Test 3: Dense depth marginal cost ({n_frames} frames)")
    print("=" * 70)

    rt = sl.RuntimeParameters()
    rgb_mat = sl.Mat()
    depth_mat = sl.Mat()

    # Mode A: grab + retrieve RGB only (no dense depth)
    times_rgb_only = []
    eof = False
    for _ in range(n_frames):
        t0 = time.perf_counter()
        err = zed.grab(rt)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            eof = True
            break
        if err != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(rgb_mat, sl.VIEW.LEFT, sl.MEM.GPU)
        t1 = time.perf_counter()
        times_rgb_only.append((t1 - t0) * 1000.0)

    # Reset SVO if needed
    if svo_path is not None and eof:
        zed.close()
        zed = sl.Camera()
        zed.open(init_params)

    # Mode B: grab + retrieve RGB + DEPTH (dense)
    times_with_depth = []
    eof = False
    for _ in range(n_frames):
        t0 = time.perf_counter()
        err = zed.grab(rt)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            eof = True
            break
        if err != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(rgb_mat, sl.VIEW.LEFT, sl.MEM.GPU)
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.GPU)
        t1 = time.perf_counter()
        times_with_depth.append((t1 - t0) * 1000.0)

    # Reset SVO again for Mode C
    if svo_path is not None and eof:
        zed.close()
        zed = sl.Camera()
        zed.open(init_params)

    # Mode C: grab + retrieve LEFT_GRAY + RIGHT_GRAY (proposed E path)
    times_left_right_gray = []
    left_gray_view = _get_view_attr('LEFT_GRAY')
    right_gray_view = _get_view_attr('RIGHT_GRAY')
    has_gray = left_gray_view is not None and right_gray_view is not None
    if has_gray:
        l_mat = sl.Mat()
        r_mat = sl.Mat()
        for _ in range(n_frames):
            t0 = time.perf_counter()
            err = zed.grab(rt)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(l_mat, left_gray_view, sl.MEM.GPU)
            zed.retrieve_image(r_mat, right_gray_view, sl.MEM.GPU)
            t1 = time.perf_counter()
            times_left_right_gray.append((t1 - t0) * 1000.0)

    # Print results
    def _stats(arr):
        if not arr:
            return None
        a = np.array(arr)
        return {
            'p50': float(np.percentile(a, 50)),
            'p99': float(np.percentile(a, 99)),
            'n':   len(arr),
        }

    A = _stats(times_rgb_only)
    B = _stats(times_with_depth)
    C = _stats(times_left_right_gray) if has_gray else None

    if A:
        print(f"  Mode A (RGB GPU only):           p50={A['p50']:6.3f}ms  p99={A['p99']:6.3f}ms  N={A['n']}")
    if B:
        print(f"  Mode B (RGB + DEPTH GPU):        p50={B['p50']:6.3f}ms  p99={B['p99']:6.3f}ms  N={B['n']}")
    if C:
        print(f"  Mode C (LEFT_GRAY + RIGHT_GRAY): p50={C['p50']:6.3f}ms  p99={C['p99']:6.3f}ms  N={C['n']}")

    if A and B:
        depth_cost_p50 = B['p50'] - A['p50']
        depth_cost_p99 = B['p99'] - A['p99']
        print()
        print(f"  >> Dense depth marginal cost:    p50={depth_cost_p50:+.3f}ms  p99={depth_cost_p99:+.3f}ms")
        print(f"  >> E path 가 제거 가능 한 latency: ~{depth_cost_p50:.2f}ms (per frame)")

    if A and C:
        gray_extra = C['p50'] - A['p50']
        print(f"  >> E path 의 추가 cost (right gray retrieve): p50={gray_extra:+.3f}ms")
        if A and B and C:
            net_gain = (B['p50'] - C['p50'])
            print(f"  >> E path 의 net latency 변화 (Mode B → Mode C): {net_gain:+.3f}ms")
            if net_gain > 0:
                print(f"  >> ✓ E path 가 latency 감소 ({net_gain:.2f}ms gain)")
            else:
                print(f"  >> ✗ E path 가 latency 증가 ({-net_gain:.2f}ms loss)")
    print()

    return {
        'A_rgb_only': A,
        'B_with_depth': B,
        'C_left_right_gray': C,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--svo', default=None, help='Replay from SVO file')
    ap.add_argument('--frames', type=int, default=100, help='Frames per test')
    args = ap.parse_args()

    # List views first (no camera needed)
    views = list_view_options()

    # Open camera
    init = sl.InitParameters()
    if args.svo:
        init.set_from_svo_file(args.svo)
        init.svo_real_time_mode = False
        print(f"Source: SVO file {args.svo}")
    else:
        init.camera_resolution = sl.RESOLUTION.SVGA
        init.camera_fps = 120
        print("Source: Live camera SVGA @ 120fps")
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: zed.open() failed: {err}", file=sys.stderr)
        return 2

    info = zed.get_camera_information()
    cfg = info.camera_configuration
    print(f"Camera: {cfg.resolution.width}x{cfg.resolution.height} @ {cfg.fps}fps, "
          f"depth_mode=PERFORMANCE")
    print()

    # Test 2
    test_retrieve_apis(zed, n_frames=args.frames, available_views=views)

    # Reset for Test 3 (if SVO, restart from beginning)
    if args.svo:
        zed.close()
        zed = sl.Camera()
        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ERROR: zed.open() failed on restart: {err}", file=sys.stderr)
            return 2

    # Test 3
    test_depth_cost(zed, init, n_frames=args.frames, svo_path=args.svo)

    zed.close()

    print("=" * 70)
    print("smoke test 완료")
    print("=" * 70)
    print()
    print("결과 paste 형식:")
    print("  1. Test 1: LEFT_GRAY/RIGHT_GRAY 의 availability")
    print("  2. Test 2: retrieve_image timing matrix")
    print("  3. Test 3: Dense depth marginal cost + E path 의 latency 변화")
    return 0


if __name__ == '__main__':
    sys.exit(main())
