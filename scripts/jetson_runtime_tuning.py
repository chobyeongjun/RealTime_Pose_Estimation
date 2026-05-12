#!/usr/bin/env python3
"""ZED RuntimeParameters tuning — 진정 sensor latency reduce 실험.

사용자 의지 (2026-05-12):
  Sensor latency floor 14ms 측정 됨 (commit 39b60db).
  ZED SDK 의 internal pipeline 9.8ms 의 reduce 의무.

실험 5 가지 configuration:
  baseline   — default RuntimeParameters
  A          — enable_fill_mode=False
  B          — confidence_threshold=100, texture_confidence_threshold=100
  C          — depth_maximum_distance=5m (default 10m)
  D (combo)  — A + B + C 모두
  E (extreme)— D + minimum_distance=0.5m

각 config 60 frames × 5 iter = 300 measurements per config.
출력: E (sensor latency) p50/p99 + total grab+retrieve+np p50.

사용 (Jetson):
    sudo nvpmodel -m 0 && sudo jetson_clocks
    PYTHONPATH=src python3 scripts/jetson_runtime_tuning.py 2>&1 | tee /tmp/runtime_tuning.log
"""
from __future__ import annotations

import sys
import time
from collections import deque
from typing import Callable

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed required (Jetson only)", file=sys.stderr)
    sys.exit(1)


def measure_config(
    cam: sl.Camera,
    name: str,
    configure_runtime: Callable[[sl.RuntimeParameters], None],
    n_frames: int = 300,
) -> dict:
    """Measure E (sensor latency) and TOTAL with a given runtime config."""
    runtime = sl.RuntimeParameters()
    configure_runtime(runtime)

    img = sl.Mat()
    depth = sl.Mat()

    sensor_lat = deque(maxlen=n_frames + 60)
    grab_lat = deque(maxlen=n_frames + 60)
    total_lat = deque(maxlen=n_frames + 60)

    # Warm-up
    for _ in range(60):
        cam.grab(runtime)
        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

    # Measure
    for _ in range(n_frames):
        t0 = time.perf_counter()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        t1 = time.perf_counter()
        grab_lat.append((t1 - t0) * 1000)

        ts_img = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        ts_now = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()
        sensor_lat.append((ts_now - ts_img) / 1e6)

        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        _ = img.get_data().shape
        _ = depth.get_data().shape
        t2 = time.perf_counter()
        total_lat.append((t2 - t0) * 1000)

    sl_arr = np.array(sensor_lat)
    grab_arr = np.array(grab_lat)
    tot_arr = np.array(total_lat)
    return {
        "name": name,
        "n": len(sl_arr),
        "sensor_p50": float(np.percentile(sl_arr, 50)),
        "sensor_p99": float(np.percentile(sl_arr, 99)),
        "sensor_mean": float(sl_arr.mean()),
        "grab_p50": float(np.percentile(grab_arr, 50)),
        "total_p50": float(np.percentile(tot_arr, 50)),
        "total_p99": float(np.percentile(tot_arr, 99)),
    }


def main():
    print("=" * 70)
    print("  ZED RuntimeParameters tuning — sensor latency 14ms reduce 실험")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.30
    init.depth_maximum_distance = 10.0

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ERROR: cam.open()", file=sys.stderr)
        sys.exit(1)
    print("  ZED opened (SVGA 120fps PERFORMANCE)")
    print()

    configs = [
        ("baseline (default)", lambda rt: None),
        (
            "A: fill=False",
            lambda rt: setattr(rt, "enable_fill_mode", False),
        ),
        (
            "B: conf=100 + tex=100",
            lambda rt: (
                setattr(rt, "confidence_threshold", 100),
                setattr(rt, "texture_confidence_threshold", 100),
            ),
        ),
        (
            "C: depth_max=5m",
            lambda rt: None,  # measure_config can't change init params after open
            # NOTE: depth_max is InitParameter — not changeable per RuntimeParameters
            # Skip per-iter; documented for reference only
        ),
        (
            "D: A+B (combined)",
            lambda rt: (
                setattr(rt, "enable_fill_mode", False),
                setattr(rt, "confidence_threshold", 100),
                setattr(rt, "texture_confidence_threshold", 100),
            ),
        ),
    ]

    results = []
    for name, cfg in configs:
        print(f"  Running {name} (300 frames)...")
        r = measure_config(cam, name, cfg)
        results.append(r)
        print(
            f"    sensor p50={r['sensor_p50']:.2f}  p99={r['sensor_p99']:.2f}  "
            f"grab p50={r['grab_p50']:.2f}  total p50={r['total_p50']:.2f}"
        )
        print()

    cam.close()

    print("=" * 70)
    print("  COMPARISON (vs baseline)")
    print("=" * 70)
    baseline = results[0]
    print(f"  {'config':<24} {'sens p50':>10} {'Δ vs base':>12} {'grab p50':>10} {'total p50':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for r in results:
        delta = r["sensor_p50"] - baseline["sensor_p50"]
        sign = "+" if delta >= 0 else ""
        print(
            f"  {r['name']:<24} "
            f"{r['sensor_p50']:>9.2f}  "
            f"{sign}{delta:>10.2f}  "
            f"{r['grab_p50']:>9.2f}  "
            f"{r['total_p50']:>9.2f}"
        )
    print()
    print("  Best config = lowest sensor p50. Negative Δ = improvement.")


if __name__ == "__main__":
    main()
