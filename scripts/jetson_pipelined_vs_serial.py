#!/usr/bin/env python3
"""PipelinedCamera vs strict-serial loop A/B test (Codex 2026-05-12 action #1).

진정 — Codex 의 brutal honest critique:
  '5.4ms = Python overhead' 가설은 진정 *predict + 3D + SHM* 의 size 와 일치 — 진정
  Python thread overhead 의 *증거 X*. PipelinedCamera 제거 시 *0-2ms 만 gain*
  가능성, 단 *A/B test 의무*.

This script measures both modes with the SAME workload:
  ① pipelined: grab+rgb release → predict (parallel depth) → depth retrieve → 3D → SHM
  ② serial:    grab → retrieve_rgb → retrieve_depth → predict → 3D → SHM

NO model dependencies — we simulate predict (TRT) + 3D with simple CPU/GPU operations
that match the typical ms budget (5-7ms predict, 1-3ms depth_3d).

사용 (Jetson, sudo nvpmodel + jetson_clocks 적용 후):
    PYTHONPATH=src python3 scripts/jetson_pipelined_vs_serial.py 2>&1 | tee /tmp/ab_test.log

Output:
    per-frame e2e latency p50/p95/p99 for both modes
    + 진정 delta (pipelined gain vs serial)
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed (Jetson only)", file=sys.stderr)
    sys.exit(1)


# Simulated workloads (match real pipeline ms budgets)
def _simulate_predict_ms(target_ms: float = 6.0) -> None:
    """Burn target_ms of CPU — proxy for TRT predict (5-7ms)."""
    t_end = time.perf_counter() + target_ms * 1e-3
    while time.perf_counter() < t_end:
        # Light arithmetic to keep CPU active
        _ = sum(i * i for i in range(50))


def _simulate_depth_3d_ms(target_ms: float = 2.0) -> None:
    """Burn target_ms — proxy for back-project (1-3ms)."""
    t_end = time.perf_counter() + target_ms * 1e-3
    while time.perf_counter() < t_end:
        _ = sum(i * i for i in range(50))


def _simulate_shm_ms(target_ms: float = 0.1) -> None:
    """SHM publish — ~0.1ms typically."""
    time.sleep(target_ms * 1e-3)


def measure_serial(cam: sl.Camera, n_frames: int = 1800) -> dict:
    """Strict serial loop: grab → retrieve_rgb → retrieve_depth → predict → 3D → SHM."""
    runtime = sl.RuntimeParameters()
    img = sl.Mat()
    depth = sl.Mat()
    e2e = deque(maxlen=n_frames + 60)
    grab_lat = deque(maxlen=n_frames + 60)

    # Warm-up
    for _ in range(60):
        cam.grab(runtime)
        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

    for _ in range(n_frames):
        t_start = time.perf_counter()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        t_grabbed = time.perf_counter()
        grab_lat.append((t_grabbed - t_start) * 1000)

        cam.retrieve_image(img, sl.VIEW.LEFT)
        _ = img.get_data()
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        _ = depth.get_data()
        _simulate_predict_ms(6.0)
        _simulate_depth_3d_ms(2.0)
        _simulate_shm_ms(0.1)
        t_done = time.perf_counter()
        e2e.append((t_done - t_start) * 1000)

    return {
        "name": "serial",
        "n": len(e2e),
        "e2e_p50": float(np.percentile(e2e, 50)),
        "e2e_p95": float(np.percentile(e2e, 95)),
        "e2e_p99": float(np.percentile(e2e, 99)),
        "e2e_mean": float(np.array(e2e).mean()),
        "grab_p50": float(np.percentile(grab_lat, 50)),
    }


def measure_pipelined(cam: sl.Camera, n_frames: int = 1800) -> dict:
    """Pipelined: grab+rgb retrieve in main, depth retrieve overlaps with predict.

    Mimics PipelinedCamera 's overlap pattern:
      main thread: grab + retrieve_image LEFT → release for capture thread
                   → simulate_predict (6ms wall time)
      capture thread (would normally exist): retrieve_depth in parallel
    Here we model with manual ordering inside the loop:
      grab → retrieve_image → start_depth_async → predict (overlaps) → wait_depth → 3D → SHM
    """
    runtime = sl.RuntimeParameters()
    img = sl.Mat()
    depth = sl.Mat()
    e2e = deque(maxlen=n_frames + 60)
    grab_lat = deque(maxlen=n_frames + 60)

    # Warm-up
    for _ in range(60):
        cam.grab(runtime)
        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

    # The ZED SDK retrieve calls themselves are synchronous, so true async
    # overlap requires a separate thread. We use a simple thread-per-frame
    # pattern to mirror what PipelinedCamera does.

    for _ in range(n_frames):
        t_start = time.perf_counter()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        t_grabbed = time.perf_counter()
        grab_lat.append((t_grabbed - t_start) * 1000)

        cam.retrieve_image(img, sl.VIEW.LEFT)
        _ = img.get_data()

        # Start depth retrieve in a thread to overlap with predict
        depth_done = threading.Event()

        def _retrieve_depth():
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            _ = depth.get_data()
            depth_done.set()

        depth_thr = threading.Thread(target=_retrieve_depth, daemon=True)
        depth_thr.start()

        # Predict overlaps with depth retrieve
        _simulate_predict_ms(6.0)

        # Wait depth (should be done already in many cases)
        depth_done.wait(timeout=0.020)

        _simulate_depth_3d_ms(2.0)
        _simulate_shm_ms(0.1)
        t_done = time.perf_counter()
        e2e.append((t_done - t_start) * 1000)

    return {
        "name": "pipelined",
        "n": len(e2e),
        "e2e_p50": float(np.percentile(e2e, 50)),
        "e2e_p95": float(np.percentile(e2e, 95)),
        "e2e_p99": float(np.percentile(e2e, 99)),
        "e2e_mean": float(np.array(e2e).mean()),
        "grab_p50": float(np.percentile(grab_lat, 50)),
    }


def main():
    print("=" * 70)
    print("  Pipelined vs Serial A/B test (Codex 2026-05-12 action #1)")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Simulated workload: predict 6ms + depth_3d 2ms + shm 0.1ms")
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

    # Run BOTH 15s each — interleave so thermal/cache state is comparable
    print("  ── Phase 1: Serial (1800 frames @ 120fps ≈ 15s) ──")
    r_serial_1 = measure_serial(cam, n_frames=1800)
    print(f"    e2e p50={r_serial_1['e2e_p50']:.2f}  p95={r_serial_1['e2e_p95']:.2f}  "
          f"p99={r_serial_1['e2e_p99']:.2f}")
    print()
    print("  ── Phase 2: Pipelined (1800 frames) ──")
    r_pipe_1 = measure_pipelined(cam, n_frames=1800)
    print(f"    e2e p50={r_pipe_1['e2e_p50']:.2f}  p95={r_pipe_1['e2e_p95']:.2f}  "
          f"p99={r_pipe_1['e2e_p99']:.2f}")
    print()
    print("  ── Phase 3: Serial again (re-run, 1800 frames) ──")
    r_serial_2 = measure_serial(cam, n_frames=1800)
    print(f"    e2e p50={r_serial_2['e2e_p50']:.2f}  p95={r_serial_2['e2e_p95']:.2f}  "
          f"p99={r_serial_2['e2e_p99']:.2f}")
    print()
    print("  ── Phase 4: Pipelined again ──")
    r_pipe_2 = measure_pipelined(cam, n_frames=1800)
    print(f"    e2e p50={r_pipe_2['e2e_p50']:.2f}  p95={r_pipe_2['e2e_p95']:.2f}  "
          f"p99={r_pipe_2['e2e_p99']:.2f}")
    print()

    cam.close()

    print("=" * 70)
    print("  AVERAGED RESULTS")
    print("=" * 70)
    s_p50 = (r_serial_1["e2e_p50"] + r_serial_2["e2e_p50"]) / 2
    s_p99 = (r_serial_1["e2e_p99"] + r_serial_2["e2e_p99"]) / 2
    p_p50 = (r_pipe_1["e2e_p50"] + r_pipe_2["e2e_p50"]) / 2
    p_p99 = (r_pipe_1["e2e_p99"] + r_pipe_2["e2e_p99"]) / 2
    print(f"  Serial    e2e p50={s_p50:.2f}  p99={s_p99:.2f}")
    print(f"  Pipelined e2e p50={p_p50:.2f}  p99={p_p99:.2f}")
    print()
    delta_p50 = s_p50 - p_p50
    delta_p99 = s_p99 - p_p99
    print(f"  Pipelined gain vs Serial:")
    print(f"    p50: {delta_p50:+.2f} ms")
    print(f"    p99: {delta_p99:+.2f} ms")
    print()
    if delta_p50 > 1.0:
        print(f"  → Pipelined helps p50 by {delta_p50:.1f} ms — KEEP PipelinedCamera")
    elif delta_p50 < -0.5:
        print(f"  → Serial is FASTER by {-delta_p50:.1f} ms — REMOVE PipelinedCamera")
    else:
        print(f"  → Negligible difference ({delta_p50:+.2f} ms) — choose for simplicity")
        print("    (Codex 2026-05-12: 5.4ms gap was predict+3D+SHM, not thread overhead.)")


if __name__ == "__main__":
    main()
