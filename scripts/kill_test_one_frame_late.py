"""Kill-test: ZED SDK 의 multi-thread Camera 안전성 + One-frame-late 가능성 검증.

Codex consult 2026-05-10 (token 287634, high reasoning) 권장.

목적:
    Path A (One-frame-late depth thread) 가 진짜 valid 한지 *2일 안 결정*.
    falsification gate 의 4 조건 모두 pass 시만 implement 진행.
    하나라도 실패 → 즉시 Path B (V4L2 + VPI).

사용법 (Jetson):
    cd ~/realtime-vision-control
    git pull origin local_backup
    sudo python3 scripts/kill_test_one_frame_late.py --duration 60 \\
        2>&1 | tee /tmp/kill_test_one_frame_late.log
    tail -50 /tmp/kill_test_one_frame_late.log

Falsification gate (Codex Q6):
    1. Frame association: depth_frame_id 가 rgb_frame_id - 1 (consistent)
    2. bridge_proc_p99 < 4ms
    3. depth_age_p99 < 16.7ms (2 frames at 120fps)
    4. Stale invalidation 작동 (worker block 시 publish invalid)

모든 4 pass → Path A 진행 (TDD red → green → Jetson)
하나라도 fail → 즉시 Path B (V4L2 + VPI) 시작

NOTE — minimal experiment. main thread (grab + retrieve_image) +
worker thread (retrieve_measure). production wire-up 은 별도.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional

LOGGER = logging.getLogger("kill_test")


@dataclass
class DepthPacket:
    frame_id: int
    ts_ns: int             # ZED IMAGE timestamp
    retrieved_ns: int      # 우리 retrieved 시각
    retrieve_ms: float     # retrieve_measure latency

    def __repr__(self):
        return f"<DepthPacket fid={self.frame_id} retrieve={self.retrieve_ms:.2f}ms>"


@dataclass
class FrameStats:
    rgb_frame_id: int
    rgb_ts_ns: int
    bridge_proc_ms: float
    grab_ms: float
    ret_rgb_ms: float
    depth_packet: Optional[DepthPacket]
    depth_age_ms: float          # rgb_ts - depth_ts (in ms)
    rgb_minus_depth_fid: int     # rgb_frame_id - depth_frame_id (expect 1)


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0,
                    help="kill-test 측정 시간 (s)")
    ap.add_argument("--warmup", type=int, default=100,
                    help="warmup frames (skip stats)")
    args = ap.parse_args()

    import sys
    # Force unbuffered logging (sudo + tee 조합 buffer 회피)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        stream=sys.stdout, force=True)
    sys.stdout.reconfigure(line_buffering=True)

    # Signal handler — 어떤 종료 원인 인지 명시
    import signal
    def signal_handler(signum, frame):
        LOGGER.error(f"signal {signum} received → graceful exit")
        sys.exit(130)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    LOGGER.info("=== Kill-test: ZED multi-thread + One-frame-late ===")
    LOGGER.info(f"  duration={args.duration}s, warmup={args.warmup} frames")

    # ZED initialization (single Camera instance shared between main + worker)
    import pyzed.sl as sl
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        LOGGER.error(f"ZED open failed: {status}")
        return 1

    LOGGER.info(f"ZED opened: {sl.Camera.get_sdk_version()}")

    # Shared state
    image_mat = sl.Mat()      # main thread owns
    depth_mat = sl.Mat()      # worker thread owns
    rt = sl.RuntimeParameters()

    # Worker thread state
    depth_request_event = threading.Event()
    depth_response_queue: Queue = Queue(maxsize=2)
    worker_stop = threading.Event()

    # Track latest grabbed frame info (main → worker handoff)
    latest_grab = {"frame_id": -1, "ts_ns": 0}
    latest_grab_lock = threading.Lock()

    def depth_worker():
        """Worker thread: 매 grab 후 main 이 trigger 하면 retrieve_measure."""
        LOGGER.info("[worker] thread started")
        while not worker_stop.is_set():
            triggered = depth_request_event.wait(timeout=0.1)
            if not triggered:
                continue
            depth_request_event.clear()

            with latest_grab_lock:
                target_fid = latest_grab["frame_id"]
                target_ts = latest_grab["ts_ns"]

            if target_fid < 0:
                continue

            t0 = time.perf_counter()
            try:
                ret = zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            except Exception as e:
                LOGGER.error(f"[worker] retrieve_measure exception: {e}")
                continue
            t1 = time.perf_counter()

            if ret != sl.ERROR_CODE.SUCCESS:
                LOGGER.warning(f"[worker] retrieve_measure failed: {ret}")
                continue

            packet = DepthPacket(
                frame_id=target_fid,
                ts_ns=target_ts,
                retrieved_ns=time.time_ns(),
                retrieve_ms=(t1 - t0) * 1e3,
            )

            try:
                # latest-only — old packet drop
                while not depth_response_queue.empty():
                    try:
                        depth_response_queue.get_nowait()
                    except Empty:
                        break
                depth_response_queue.put_nowait(packet)
            except Exception as e:
                LOGGER.error(f"[worker] queue push failed: {e}")

        LOGGER.info("[worker] thread exiting")

    worker = threading.Thread(target=depth_worker, name="depth_worker", daemon=True)
    worker.start()

    # Main loop — grab + retrieve_image only (depth is async)
    stats: list[FrameStats] = []
    rgb_frame_id = 0
    t_start = time.time()
    grab_fail_count = 0
    LOGGER.info(f"[main] entering loop, duration={args.duration}s")

    while time.time() - t_start < args.duration:
        t0 = time.perf_counter()
        try:
            grab_result = zed.grab(rt)
        except Exception as e:
            LOGGER.error(f"[main] grab() exception: {e}")
            grab_fail_count += 1
            time.sleep(0.001)
            continue

        if grab_result != sl.ERROR_CODE.SUCCESS:
            grab_fail_count += 1
            if grab_fail_count <= 3 or grab_fail_count % 100 == 0:
                LOGGER.warning(f"[main] grab fail #{grab_fail_count}: {grab_result}")
            time.sleep(0.001)
            continue
        t_grab = time.perf_counter()

        rgb_ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

        # Trigger worker for THIS frame's depth
        with latest_grab_lock:
            latest_grab["frame_id"] = rgb_frame_id
            latest_grab["ts_ns"] = rgb_ts_ns
        depth_request_event.set()

        # retrieve_image (main thread)
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        t_rgb = time.perf_counter()

        # Try to grab a depth packet (latest available, non-blocking)
        depth_packet: Optional[DepthPacket] = None
        try:
            depth_packet = depth_response_queue.get_nowait()
        except Empty:
            pass

        depth_age_ms = float("nan")
        rgb_minus_depth_fid = -1
        if depth_packet is not None:
            depth_age_ms = (rgb_ts_ns - depth_packet.ts_ns) / 1e6
            rgb_minus_depth_fid = rgb_frame_id - depth_packet.frame_id

        bridge_proc_ms = (t_rgb - t0) * 1e3
        stats.append(FrameStats(
            rgb_frame_id=rgb_frame_id,
            rgb_ts_ns=rgb_ts_ns,
            bridge_proc_ms=bridge_proc_ms,
            grab_ms=(t_grab - t0) * 1e3,
            ret_rgb_ms=(t_rgb - t_grab) * 1e3,
            depth_packet=depth_packet,
            depth_age_ms=depth_age_ms,
            rgb_minus_depth_fid=rgb_minus_depth_fid,
        ))

        rgb_frame_id += 1

        # Progress log every 200 frames (~1.5s at 120fps)
        if rgb_frame_id % 200 == 0:
            LOGGER.info(f"[main] frame {rgb_frame_id} elapsed={time.time()-t_start:.1f}s "
                        f"bridge={bridge_proc_ms:.2f}ms depth_age={depth_age_ms:.2f}ms")

    LOGGER.info(f"[main] loop exit. total_frames={rgb_frame_id}, grab_fail={grab_fail_count}")

    # Stop worker
    worker_stop.set()
    depth_request_event.set()
    worker.join(timeout=2.0)
    zed.close()

    # Analysis
    if len(stats) <= args.warmup:
        LOGGER.error(f"insufficient frames: {len(stats)} <= warmup {args.warmup}")
        return 1
    s = stats[args.warmup:]
    LOGGER.info(f"Total frames: {len(stats)}, post-warmup analysis: {len(s)}")
    LOGGER.info(f"Throughput: {len(stats)/args.duration:.1f} Hz")

    # Gate 1: Frame association consistency
    fid_diffs = [x.rgb_minus_depth_fid for x in s if x.depth_packet is not None]
    if fid_diffs:
        from collections import Counter
        fid_diff_counter = Counter(fid_diffs)
        most_common = fid_diff_counter.most_common(3)
        LOGGER.info(f"Frame ID diff distribution: {most_common}")

    # Gate 2: bridge_proc_p99
    bp = [x.bridge_proc_ms for x in s]
    bp_p50 = percentile(bp, 50)
    bp_p99 = percentile(bp, 99)
    LOGGER.info(f"bridge_proc p50/p99: {bp_p50:.2f}/{bp_p99:.2f} ms")

    # grab + ret_rgb breakdown
    grab = [x.grab_ms for x in s]
    rrgb = [x.ret_rgb_ms for x in s]
    LOGGER.info(f"  grab    p50/p99: {percentile(grab, 50):.2f}/{percentile(grab, 99):.2f} ms")
    LOGGER.info(f"  ret_rgb p50/p99: {percentile(rrgb, 50):.2f}/{percentile(rrgb, 99):.2f} ms")

    # Gate 3: depth_age_p99
    valid_packets = [x for x in s if x.depth_packet is not None]
    LOGGER.info(f"Frames with depth packet: {len(valid_packets)} / {len(s)} "
                f"({100*len(valid_packets)/len(s):.1f}%)")
    if valid_packets:
        ages = [x.depth_age_ms for x in valid_packets]
        retrieves = [x.depth_packet.retrieve_ms for x in valid_packets]
        LOGGER.info(f"depth_age   p50/p99: {percentile(ages, 50):.2f}/{percentile(ages, 99):.2f} ms")
        LOGGER.info(f"retrieve_ms p50/p99: {percentile(retrieves, 50):.2f}/{percentile(retrieves, 99):.2f} ms")

    # ===== Falsification gate =====
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("FALSIFICATION GATE (Codex Q6)")
    LOGGER.info("=" * 60)

    gate_results = []

    # Gate 1: frame association consistency
    if fid_diffs:
        most_common_diff, most_common_count = Counter(fid_diffs).most_common(1)[0]
        consistency = most_common_count / len(fid_diffs)
        LOGGER.info(f"  [G1] Frame association: most-common diff={most_common_diff}, "
                    f"consistency={100*consistency:.1f}%")
        gate1_pass = consistency >= 0.95 and most_common_diff in (0, 1)
        gate_results.append(("G1 frame association", gate1_pass))
    else:
        LOGGER.warning("  [G1] No depth packets — gate FAIL")
        gate_results.append(("G1 frame association", False))

    # Gate 2: bridge_proc_p99 < 4ms
    gate2_pass = bp_p99 < 4.0
    LOGGER.info(f"  [G2] bridge_proc_p99 < 4ms: {bp_p99:.2f}ms → {'PASS' if gate2_pass else 'FAIL'}")
    gate_results.append(("G2 bridge_proc_p99 < 4ms", gate2_pass))

    # Gate 3: depth_age_p99 < 16.7ms
    if valid_packets:
        ages_p99 = percentile(ages, 99)
        gate3_pass = ages_p99 < 16.7
        LOGGER.info(f"  [G3] depth_age_p99 < 16.7ms: {ages_p99:.2f}ms → {'PASS' if gate3_pass else 'FAIL'}")
        gate_results.append(("G3 depth_age_p99 < 16.7ms", gate3_pass))
    else:
        gate_results.append(("G3 depth_age_p99 < 16.7ms", False))

    # Gate 4: stale invalidation (간단 — depth packet 가 항상 있어야)
    stale_count = sum(1 for x in s if x.depth_packet is None or x.rgb_minus_depth_fid > 2)
    gate4_pass = stale_count < len(s) * 0.05  # 5% 이하
    LOGGER.info(f"  [G4] stale rate < 5%: {100*stale_count/len(s):.1f}% → "
                f"{'PASS' if gate4_pass else 'FAIL'}")
    gate_results.append(("G4 stale rate < 5%", gate4_pass))

    LOGGER.info("")
    LOGGER.info("=" * 60)
    all_pass = all(p for _, p in gate_results)
    LOGGER.info(f"OVERALL: {'PASS' if all_pass else 'FAIL'} ({sum(1 for _, p in gate_results if p)}/{len(gate_results)})")
    if all_pass:
        LOGGER.info("→ Path A (One-frame-late depth thread) PROCEED")
    else:
        LOGGER.info("→ Path A FAIL. Switch to Path B (V4L2 + VPI sparse stereo) immediately.")
    LOGGER.info("=" * 60)

    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
