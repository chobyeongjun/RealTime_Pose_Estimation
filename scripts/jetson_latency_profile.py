#!/usr/bin/env python3
"""Jetson vision sensor latency 분해 — 진정 측정.

사용자 의지 (2026-05-12): "vision sensor 60ms 전체를 raw 로 reduce, 모든 항목 다."

이 script 는 ZED SDK 의 각 단계 latency 를 *진정 측정* 한다 (guess 가 아닌).
30s run → 각 단계의 p50/p95/p99 + breakdown.

측정 항목:
    A) ZED grab() 자체 — sensor → driver buffer
    B) retrieve_image (LEFT) — driver buffer → sl.Mat
    C) retrieve_measure (DEPTH) — driver buffer → sl.Mat
    D) sl.Mat → numpy copy time
    E) sensor latency = ZED CURRENT timestamp - ZED IMAGE timestamp
       (둘 모두 CLOCK_REALTIME epoch ns — 동일 reference 의무)
       ⚠️ Bug fix 2026-05-12: 이전 버전이 time.monotonic_ns() (boot ns) 사용
           → reference mismatch → huge negative (~-1.78e12 ms)

진정 의미:
    A = ZED SDK 의 internal queue + ISP processing
    E = sensor latency 의 진정 measurement (sensor capture → frame available)
    D = bridge layer overhead

사용 (Jetson):
    sudo nvpmodel -m 0 && sudo jetson_clocks
    PYTHONPATH=src python3 scripts/jetson_latency_profile.py 2>&1 | tee /tmp/latency_profile.log

paste to Mac:
    /tmp/latency_profile.log 의 마지막 30 lines.
"""
from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np

# Try to import pyzed (Jetson only)
try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed not installed. This script is Jetson-only.", file=sys.stderr)
    sys.exit(1)


def percentile_str(arr, p_list):
    return " ".join(f"p{p}={np.percentile(arr, p):.2f}ms" for p in p_list)


def main():
    print("=" * 60)
    print("  Jetson ZED latency profile — 진정 measurement")
    print("=" * 60)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Init
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.SVGA  # 960×600 — production
    init_params.camera_fps = 120
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.30
    init_params.depth_maximum_distance = 10.0

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: cam.open() failed: {status}", file=sys.stderr)
        sys.exit(1)
    print(f"  ZED opened — resolution={init_params.camera_resolution}, fps={init_params.camera_fps}")
    print(f"  depth_mode={init_params.depth_mode}")
    print()

    # Pre-allocate Mats
    img_l = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()

    # Warm-up 30 frames
    print("  Warming up 30 frames...")
    for _ in range(30):
        cam.grab(runtime)
        cam.retrieve_image(img_l, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

    # Measurement: 30s run
    duration_s = 30.0
    print(f"  Measuring {duration_s} s...")
    t_grab = deque(maxlen=10000)
    t_retr_img = deque(maxlen=10000)
    t_retr_dep = deque(maxlen=10000)
    t_to_np_img = deque(maxlen=10000)
    t_to_np_dep = deque(maxlen=10000)
    sensor_latency = deque(maxlen=10000)
    t_total = deque(maxlen=10000)
    # Frame interval (jitter detection) + image timestamp delta (driver queue?)
    frame_interval = deque(maxlen=10000)
    image_ts_delta = deque(maxlen=10000)
    prev_grab_t: float = -1.0
    prev_image_ts_ns: int = -1

    t_start = time.perf_counter()
    n = 0
    while time.perf_counter() - t_start < duration_s:
        t0 = time.perf_counter()
        # A) grab()
        rc = cam.grab(runtime)
        t1 = time.perf_counter()
        if rc != sl.ERROR_CODE.SUCCESS:
            continue
        t_grab.append((t1 - t0) * 1000)

        # E) sensor latency: ZED IMAGE timestamp vs ZED CURRENT timestamp
        # ★ Bug fix 2026-05-12: 둘 모두 CLOCK_REALTIME ns since epoch — 동일 ref.
        # 이전: time.monotonic_ns() (boot ns) ← reference mismatch → huge negative
        ts_image_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        ts_now_ns = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()
        sensor_lat_ms = (ts_now_ns - ts_image_ns) / 1e6
        sensor_latency.append(sensor_lat_ms)

        # Frame interval (jitter / dropped frame detection)
        if prev_grab_t > 0:
            frame_interval.append((t1 - prev_grab_t) * 1000)
        prev_grab_t = t1
        # Image timestamp delta — driver-reported frame interval
        if prev_image_ts_ns > 0:
            image_ts_delta.append((ts_image_ns - prev_image_ts_ns) / 1e6)
        prev_image_ts_ns = ts_image_ns

        # B) retrieve_image LEFT
        t2 = time.perf_counter()
        cam.retrieve_image(img_l, sl.VIEW.LEFT)
        t3 = time.perf_counter()
        t_retr_img.append((t3 - t2) * 1000)

        # C) retrieve_measure DEPTH
        t4 = time.perf_counter()
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        t5 = time.perf_counter()
        t_retr_dep.append((t5 - t4) * 1000)

        # D) sl.Mat → numpy (zero-copy view vs copy)
        t6 = time.perf_counter()
        arr_img = img_l.get_data()  # numpy array
        # Force materialization (view vs copy)
        _ = arr_img.shape
        t7 = time.perf_counter()
        t_to_np_img.append((t7 - t6) * 1000)

        t8 = time.perf_counter()
        arr_dep = depth.get_data()
        _ = arr_dep.shape
        t9 = time.perf_counter()
        t_to_np_dep.append((t9 - t8) * 1000)

        t_total.append((t9 - t0) * 1000)
        n += 1

    cam.close()
    print(f"  Captured {n} frames in {time.perf_counter() - t_start:.1f} s")
    print()

    # Report
    print("=" * 60)
    print("  RESULTS (per-frame latency ms)")
    print("=" * 60)
    arrs = [
        ("A grab()              ", np.array(t_grab)),
        ("B retrieve_image LEFT ", np.array(t_retr_img)),
        ("C retrieve_measure D  ", np.array(t_retr_dep)),
        ("D sl.Mat→np IMG       ", np.array(t_to_np_img)),
        ("D sl.Mat→np DEPTH     ", np.array(t_to_np_dep)),
        ("E SENSOR LATENCY      ", np.array(sensor_latency)),
        ("F frame_interval (loop)", np.array(frame_interval)),
        ("G image_ts_delta (drv) ", np.array(image_ts_delta)),
        ("TOTAL grab+retrieve+np ", np.array(t_total)),
    ]
    print(f"  {'stage':<23} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print(f"  {'-'*23} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, arr in arrs:
        if len(arr) == 0:
            print(f"  {name:<23}  (no samples)")
            continue
        print(
            f"  {name:<23} "
            f"{arr.mean():>7.2f}  "
            f"{np.percentile(arr, 50):>7.2f}  "
            f"{np.percentile(arr, 95):>7.2f}  "
            f"{np.percentile(arr, 99):>7.2f}  "
            f"{arr.max():>7.2f}"
        )
    print()
    print("=" * 60)
    print("  진정 분석 (interpret):")
    print("=" * 60)
    sensor_p50 = float(np.percentile(sensor_latency, 50))
    sensor_p99 = float(np.percentile(sensor_latency, 99))
    grab_p50 = float(np.percentile(t_grab, 50))
    print(f"  E sensor latency p50 = {sensor_p50:.1f} ms (capture → grab return)")
    print(f"  E sensor latency p99 = {sensor_p99:.1f} ms")
    print(f"  A grab() p50         = {grab_p50:.1f} ms")
    print()
    if sensor_p50 < 25:
        print("  → sensor latency 의 진정 floor ≈ 22 ms (architecture)")
    elif sensor_p50 < 35:
        print("  → sensor latency 좀 높음 (25-35ms) — ZED SDK queue 가능")
    else:
        print(f"  → sensor latency 매우 높음 ({sensor_p50:.0f}ms) — buffering 의심")
    print()
    print("  Easy wins potential:")
    print(f"    bridge (D) 합계 p50 = "
          f"{np.percentile(t_to_np_img, 50) + np.percentile(t_to_np_dep, 50):.2f} ms")
    print(f"    queue overhead (E - A) p50 = "
          f"{sensor_p50 - grab_p50:.1f} ms")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
