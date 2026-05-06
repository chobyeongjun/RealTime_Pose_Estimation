#!/usr/bin/env python3
"""Bridge-only ZED bench — pipeline 떼고 grab+ts만 측정.

목적
----
Codex R1 protocol 적용. 우리 시스템에서 *ZED 진짜 latency 격리*.
full pipeline에서는 zed_lag p50 ~33ms 측정되지만, 이게:
  (a) ZED SDK 자체 latency  (못 줄임)
  (b) 우리 bridge cycle이 SDK frame을 못 따라가서 적체  (개선 가능)
중 어느 것인지 모름. bridge-only bench로 (a) 격리.

측정 metric
-----------
delta_ts_ms       frame 간격 (ZED 실제 cadence)
zed_lag_ms (★)    host_after_grab_ns - ts_ns (ZED → host 도달 latency)
grab_call_ms      grab() 함수 자체 소요 시간
retrieve_rgb_ms   retrieve_image 시간
retrieve_depth_ms retrieve_measure 시간
getdata_rgb_ms    get_data RGB 시간 (host copy)
getdata_depth_ms  get_data DEPTH 시간

해석 가이드
-----------
delta_ts ≈ 8.3ms 일관:    카메라 정상 frame rate (120fps)
delta_ts 16.6/25ms 자주:   SDK frame skip 또는 우리가 따라잡기 못 함
zed_lag p50 < 5ms:         SDK 내부 buffer 적체 없음 (이상적)
zed_lag p50 17-25ms:       SDK가 1-3 frame 들고 있음 (정상 GMSL2 latency)
zed_lag p50 > 25ms:        SDK buffer 심각하게 적체
grab_call < 2ms:           buffer에 frame 이미 있음 (적체 신호)
grab_call ≈ 8ms:           SDK가 다음 frame 대기 (정상)

실행 (Jetson)
-------------
    sudo python3 scripts/bridge_only_bench.py 30
    sudo python3 scripts/bridge_only_bench.py 30 --no-getdata    # retrieve만
    sudo python3 scripts/bridge_only_bench.py 30 --no-depth      # RGB만
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List

import numpy as np


def load_pyzed():
    try:
        import pyzed.sl as sl  # noqa: WPS433
        return sl
    except ImportError as e:
        print(f"ERROR: pyzed import failed: {e}", file=sys.stderr)
        print("  → Jetson (ZED SDK 설치 호스트) 에서만 실행 가능", file=sys.stderr)
        sys.exit(1)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), q))


def print_stats(name: str, values: List[float]) -> None:
    if not values:
        print(f"  {name:<26}  (no data)")
        return
    arr = np.asarray(values)
    print(
        f"  {name:<26}  "
        f"min={float(arr.min()):6.2f}  "
        f"p50={percentile(values, 50):6.2f}  "
        f"p95={percentile(values, 95):6.2f}  "
        f"p99={percentile(values, 99):6.2f}  "
        f"max={float(arr.max()):6.2f}  "
        f"mean={float(arr.mean()):6.2f} ms"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("duration", type=float, nargs="?", default=30.0,
                    help="측정 시간 (초, default 30)")
    ap.add_argument("--resolution", default="SVGA",
                    choices=["VGA", "SVGA", "HD720", "HD1080", "HD1200"])
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--depth-mode", default="PERFORMANCE",
                    choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"])
    ap.add_argument("--no-depth", action="store_true",
                    help="depth 비활성화 (RGB만 측정)")
    ap.add_argument("--no-getdata", action="store_true",
                    help="get_data() 호출 안 함 (retrieve까지만)")
    ap.add_argument("--warmup", type=int, default=30,
                    help="warmup frame 수 (default 30)")
    args = ap.parse_args()

    sl = load_pyzed()

    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    init.coordinate_units = sl.UNIT.METER
    if args.no_depth:
        init.depth_mode = sl.DEPTH_MODE.NONE
        depth_enabled = False
    else:
        init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
        depth_enabled = True

    zed = sl.Camera()
    status = zed.open(init)
    if str(status) != "SUCCESS":
        print(f"ERROR: ZED open failed: {status}", file=sys.stderr)
        print(f"  → 다른 ZED process 종료: sudo pkill -f run_stream_demo", file=sys.stderr)
        return 1

    print(f"=== Bridge-only ZED bench ===")
    print(f"  resolution: {args.resolution}@{args.fps}fps")
    print(f"  depth     : {args.depth_mode if depth_enabled else 'NONE'}")
    print(f"  get_data  : {'NO' if args.no_getdata else 'YES'}")
    print(f"  duration  : {args.duration}s")
    print(f"  warmup    : {args.warmup} frames")
    print()

    rt = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    # ─── Warmup ──
    print(f"warmup {args.warmup} frames...", end=" ", flush=True)
    for _ in range(args.warmup):
        if zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            if depth_enabled:
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    print("done")

    # ─── Measure ──
    delta_ts_ms: List[float] = []
    zed_lag_ms: List[float] = []
    grab_call_ms: List[float] = []
    retrieve_rgb_ms: List[float] = []
    retrieve_depth_ms: List[float] = []
    getdata_rgb_ms: List[float] = []
    getdata_depth_ms: List[float] = []

    prev_ts_ns = 0
    t_start = time.monotonic()
    n_frames = 0
    n_failed = 0

    while time.monotonic() - t_start < args.duration:
        t0 = time.time_ns()
        ok = zed.grab(rt) == sl.ERROR_CODE.SUCCESS
        t1 = time.time_ns()

        if not ok:
            n_failed += 1
            continue

        ts_ns = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

        grab_call_ms.append((t1 - t0) / 1e6)
        zed_lag_ms.append((t1 - ts_ns) / 1e6)

        if prev_ts_ns:
            delta_ts_ms.append((ts_ns - prev_ts_ns) / 1e6)
        prev_ts_ns = ts_ns

        # retrieve RGB
        t_a = time.time_ns()
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        t_b = time.time_ns()
        retrieve_rgb_ms.append((t_b - t_a) / 1e6)

        if not args.no_getdata:
            t_a = time.time_ns()
            _ = image_mat.get_data(deep_copy=True)
            t_b = time.time_ns()
            getdata_rgb_ms.append((t_b - t_a) / 1e6)

        # retrieve DEPTH
        if depth_enabled:
            t_a = time.time_ns()
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            t_b = time.time_ns()
            retrieve_depth_ms.append((t_b - t_a) / 1e6)

            if not args.no_getdata:
                t_a = time.time_ns()
                _ = depth_mat.get_data(deep_copy=True)
                t_b = time.time_ns()
                getdata_depth_ms.append((t_b - t_a) / 1e6)

        n_frames += 1

    elapsed = time.monotonic() - t_start
    fps = n_frames / elapsed if elapsed > 0 else 0

    # ─── Report ──
    print()
    print(f"=== Results ({n_frames} frames in {elapsed:.1f}s = {fps:.1f} Hz) ===")
    print(f"  failed grabs: {n_failed}")
    print()

    print_stats("grab_call_ms",        grab_call_ms)
    print_stats("zed_lag_ms (★)",      zed_lag_ms)
    print_stats("delta_ts_ms",         delta_ts_ms)
    print_stats("retrieve_rgb_ms",     retrieve_rgb_ms)
    print_stats("getdata_rgb_ms",      getdata_rgb_ms)
    print_stats("retrieve_depth_ms",   retrieve_depth_ms)
    print_stats("getdata_depth_ms",    getdata_depth_ms)

    # delta_ts 분포
    print()
    print("=== delta_ts 분포 (frame 간격) ===")
    if delta_ts_ms:
        bins = [
            (0,    9,  "8.3ms 근처 (120fps 정상)"),
            (9,    12, "9-12ms (약간 느림)"),
            (12,   18, "12-18ms (skip 1 frame)"),
            (18,   25, "18-25ms (skip 2)"),
            (25,   35, "25-35ms (skip 3)"),
            (35,   50, "35-50ms (skip 4-5)"),
            (50,   1e9, ">50ms (큰 skip)"),
        ]
        total = len(delta_ts_ms)
        for lo, hi, label in bins:
            n = sum(1 for d in delta_ts_ms if lo <= d < hi)
            pct = n / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:<28}  {n:>6} ({pct:5.1f}%) {bar}")

    # 해석
    print()
    print("=== 해석 ===")
    z_p50 = percentile(zed_lag_ms, 50)
    z_p99 = percentile(zed_lag_ms, 99)
    g_p50 = percentile(grab_call_ms, 50)
    d_p50 = percentile(delta_ts_ms, 50)

    if z_p50 < 5:
        print(f"  zed_lag p50={z_p50:.1f}ms — SDK 내부 buffer 적체 없음 (이상적)")
    elif z_p50 < 17:
        print(f"  zed_lag p50={z_p50:.1f}ms — SDK가 약간 들고 있음 (적체 시작)")
    elif z_p50 < 25:
        print(f"  zed_lag p50={z_p50:.1f}ms — SDK가 1-3 frame 적체 (typical GMSL2 buffering)")
    else:
        print(f"  zed_lag p50={z_p50:.1f}ms — SDK buffer 심각하게 적체")

    if g_p50 < 2:
        print(f"  grab_call p50={g_p50:.1f}ms — buffer에 frame 이미 있음 (적체 신호)")
    elif g_p50 < 9:
        print(f"  grab_call p50={g_p50:.1f}ms — SDK가 다음 frame 대기 (정상)")
    else:
        print(f"  grab_call p50={g_p50:.1f}ms — frame 도착 지연")

    if d_p50:
        if 7.5 <= d_p50 <= 9.5:
            print(f"  delta_ts p50={d_p50:.1f}ms — 120fps 정상 cadence")
        elif d_p50 < 7.5:
            print(f"  delta_ts p50={d_p50:.1f}ms — frame 더 빠름? (예상 외)")
        else:
            print(f"  delta_ts p50={d_p50:.1f}ms — frame skip 발생 중")

    # 비교
    print()
    print("=== 비교: full pipeline (2026-05-06 baseline) ===")
    print(f"  bridge-only zed_lag p50  = {z_p50:.1f} ms")
    print(f"  full pipeline zed_lag p50 ~ 33 ms")
    if z_p50 < 25:
        diff = 33 - z_p50
        if diff > 5:
            print(f"  → 차이 +{diff:.1f}ms — full pipeline에서 우리 bridge cycle이 못 따라가서 추가 적체")
        else:
            print(f"  → 차이 작음 — zed_lag는 SDK 자체 한계")
    else:
        print(f"  → bridge-only도 큼 — SDK 자체 한계, pipeline 영향 작음")

    zed.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
