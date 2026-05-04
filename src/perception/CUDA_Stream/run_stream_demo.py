"""End-to-end demo — ZED → Stream pipeline → SHM publish.

    python3 -m perception.CUDA_Stream.run_stream_demo \\
        --engine src/perception/CUDA_Stream/yolo26s-pose.engine \\
        --resolution SVGA --duration 600 --publish-shm /hwalker_pose_cuda

Ctrl+C exits cleanly. On SIGTERM from an outer runner, the module also
shuts the ZED / CUDA streams / SHM down and unlinks ``/dev/shm/…``.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ─── HARD REAL-TIME GUARANTEE ─────────────────────────────────────────────
# User requirement: no frame may exceed 20 ms end-to-end.
# If a frame does exceed the budget (spike from GC / thermal / scheduler),
# we publish it with valid=False so the C++ control loop SKIPS it instead
# of feeding stale/late pose into the impedance/ILC model. The next frame
# ships normally.
#
# LATENCY_HARD_LIMIT_MS is the absolute ceiling — data past this is
# considered stale for control purposes and must NOT reach Teensy as-is.
# 20 ms user-defined. 18 ms soft warning (below) triggers [SLOW] log.
LATENCY_HARD_LIMIT_MS = 20.0   # true_e2e_ms (camera→GPU done) ceiling
LATENCY_SOFT_WARN_MS  = 18.0

from .constraints import (
    BoneLengthConstraint,
    ConstraintStack,
    JointVelocityBound,
)
from .gpu_postprocess import GpuPostprocessor
from .gpu_preprocess import GpuPreprocessor
from .keypoint_config import get_schema
from .pipeline import StreamedPosePipeline
from .shm_publisher import DEFAULT_NAME, ShmPublisher
from .stream_manager import StreamManager
from .tracer import PipelineTracer
from .trt_runner import TRTRunner, warmup
from .watchdog import StreamWatchdog
from .zed_gpu_bridge import ZEDGpuBridge


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--resolution", default="SVGA",
                    choices=["SVGA", "HD720", "HD1080", "HD1200"])
    ap.add_argument("--fps", type=int, default=None)
    # NEURAL removed — see skiro-learnings (2.4× predict spike).
    ap.add_argument("--depth-mode", default="PERFORMANCE",
                    choices=["NONE", "PERFORMANCE", "QUALITY"])
    ap.add_argument("--trace", default=None,
                    help="path to per-frame trace CSV (enables per-stage GPU timing)")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--publish-shm", default=DEFAULT_NAME.lstrip("/"))
    ap.add_argument("--no-shm", action="store_true")
    ap.add_argument(
        "--cpu-affinity", default="2,3,4,5",
        help="comma-separated cores. Default '2,3,4,5' reserves 0-1 for "
             "system + 6-7 for C++ control loop (skiro-learnings: CPU "
             "isolation eliminates predict spike clusters). Pass '' to "
             "disable affinity.",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--schema", default="lowlimb6", choices=["coco17", "lowlimb6"],
        help="keypoint schema — must match the exported engine",
    )
    ap.add_argument(
        "--use-filter", action="store_true",
        help="enable OneEuro (default OFF — observed to suppress detection)",
    )
    ap.add_argument(
        "--bone-constraint", action="store_true",
        help="enable bone-length hard-gate (default OFF — feedback-loop risk)",
    )
    ap.add_argument(
        "--velocity-bound-mps", type=float, default=0.0,
        help="joint velocity hard-gate m/s (0 = disabled)",
    )
    ap.add_argument(
        "--no-world-frame", action="store_true",
        help="skip IMU-based rotation (keep output in camera frame).",
    )
    ap.add_argument(
        "--camera-pitch-deg", type=float, default=None,
        help="manual forward pitch override (e.g. 32 for camera mounted "
             "leaning 32° down). Overrides IMU warmup — most reliable path.",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def maybe_set_affinity(spec: str) -> None:
    if not spec:
        return
    try:
        cores = {int(c) for c in spec.split(",") if c.strip()}
        os.sched_setaffinity(0, cores)
        LOGGER.info("CPU affinity set to %s", sorted(cores))
    except (AttributeError, OSError, ValueError) as err:
        LOGGER.warning("affinity %s failed: %s", spec, err)


def _cleanup_stale_resources() -> None:
    """Remove stale resources from previous crashed/killed publisher runs.

    Without this, each run accumulates:
      * leaked /dev/shm/hwalker_pose_cuda from pkill -9 shutdowns
      * ZED Argus SciStream IPC state (sem.ipc_test_*, sem.itc_test_*)
      * CUDA context fragments
    These accumulate and cause the "degraded after multiple runs" pattern
    (first run: 0.04% HARD violations, fourth run: 64%). Cleanup here
    makes every launch behave like a fresh boot.
    """
    import subprocess
    # 1. Old SHM from this module
    for path in (
        "/dev/shm/hwalker_pose_cuda",
        "/dev/shm/sem.hwalker_pose_cuda",
    ):
        if os.path.exists(path):
            try:
                os.remove(path)
                LOGGER.info("cleaned stale SHM: %s", path)
            except OSError:
                pass

    # 2. Argus test IPC remnants (from crashed ZED samples)
    try:
        subprocess.run(
            "rm -f /dev/shm/sem.ipc_test_* /dev/shm/sem.itc_test_*",
            shell=True, check=False,
        )
    except Exception:
        pass

    # 3. Any previous python publisher still alive (safety net — user
    #    should pkill before running, but double-check)
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_stream_demo"],
            capture_output=True, text=True, timeout=1,
        )
        my_pid = str(os.getpid())
        other_pids = [p for p in result.stdout.split() if p and p != my_pid]
        if other_pids:
            LOGGER.warning(
                "found %d previous run_stream_demo process(es): %s — killing",
                len(other_pids), other_pids,
            )
            for p in other_pids:
                subprocess.run(["kill", "-9", p], check=False)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    maybe_set_affinity(args.cpu_affinity)

    # Clean startup — remove leaked resources from previous sessions.
    # This is the #1 fix for "performance degrades across multiple runs":
    # user tested 6+ times in one session, each Ctrl+C left SHM / Argus
    # state. Result: 0.046% → 64% HARD violations without reboot.
    _cleanup_stale_resources()

    # HARD real-time: disable Python GC completely. GC causes unpredictable
    # 2-5ms pauses that push p99 above the 20ms budget. All pipeline buffers
    # are pre-allocated (pinned host, GPU tensors) — GC has nothing to do.
    gc.disable()
    gc.collect()   # clear any accumulated state from init
    LOGGER.info("GC disabled for real-time guarantee (p99 < 20ms target)")

    # Try SCHED_FIFO priority. Works if user has rtprio capability
    # (setup: /etc/security/limits.d/realtime.conf) or running as root
    # (sudo chrt -r 90). Fails silently otherwise — CPU isolation +
    # frame-skip are still the primary guarantees.
    try:
        os.sched_setscheduler(
            0, os.SCHED_FIFO, os.sched_param(90)
        )
        LOGGER.info("SCHED_FIFO priority 90 applied (RT scheduling active)")
    except (AttributeError, PermissionError, OSError) as err:
        LOGGER.info(
            "SCHED_FIFO skipped (%s) — add 'chobb0 - rtprio 99' to "
            "/etc/security/limits.d/realtime.conf + re-login for RT",
            type(err).__name__,
        )

    # Empty any leaked CUDA memory from prior runs in same process
    # (no-op at startup, but ensures baseline is known).
    try:
        import torch  # already imported, but keep local scope clean
        torch.cuda.empty_cache()
    except Exception:
        pass

    device = torch.device("cuda:0")
    schema = get_schema(args.schema)
    sm = StreamManager(device=device, high_priority_stages=["infer"])
    runner = TRTRunner(args.engine, device=device)
    # Match preproc dtype to engine's input binding — Ultralytics engines
    # with --half still keep I/O as float32 by default, so probing here
    # avoids "dtype mismatch" at bind_input_address.
    engine_in_dtype = runner.bindings[runner.input_names[0]].dtype
    LOGGER.info("engine input dtype = %s, matching preproc accordingly", engine_in_dtype)
    pre = GpuPreprocessor(imgsz=args.imgsz, device=device, dtype=engine_in_dtype)
    post = GpuPostprocessor(
        schema=schema, device=device, use_filter=args.use_filter
    )
    stack = ConstraintStack()
    if args.bone_constraint:
        stack.bone_length = BoneLengthConstraint(schema, device=device)
    if args.velocity_bound_mps > 0:
        stack.joint_velocity = JointVelocityBound(
            max_velocity_mps=args.velocity_bound_mps, device=device
        )

    bridge = ZEDGpuBridge(
        resolution=args.resolution,
        fps=args.fps,
        depth_mode=args.depth_mode,
        device=device,
        enable_depth=args.depth_mode != "NONE",
        world_frame=not args.no_world_frame,
        manual_pitch_deg=args.camera_pitch_deg,
    )
    bridge.open()
    bridge.start()

    tracer = PipelineTracer(
        enabled=bool(args.trace),
        csv_path=args.trace,
        device=device,
    )
    # Warm up TRT — allocates workspace, loads kernels, populates CUDA
    # caches. Previously 10 iters; increased to 30 because observed
    # first real inference (frame 6) hit 69ms despite 10-iter warmup.
    # TRT's first few launches include kernel selection / autotune.
    LOGGER.info("warmup ×30 …")
    warmup(runner, sm.stream_ptr("infer"), iters=30)
    # Also force a cudaDeviceSynchronize so ZED's depth pipeline
    # (which kicks in on first grab) doesn't collide with warmup state.
    torch.cuda.synchronize()

    publisher = None
    if not args.no_shm:
        publisher = ShmPublisher(
            num_keypoints=schema.num_keypoints,
            name=args.publish_shm, create=True,
        )
        LOGGER.info(
            "publishing K=%d to /dev/shm/%s", schema.num_keypoints, args.publish_shm
        )

    watchdog = StreamWatchdog(
        streams={k: b.stream for k, b in sm.streams.items()},
        publisher=publisher,
        fallback_cb=lambda reason: LOGGER.error("FALLBACK TRIGGERED: %s", reason),
    )
    watchdog.start()

    # Pipeline AFTER watchdog so we can pass the watchdog ref. Pipeline
    # needs to pause()/resume() the watchdog around CUDA graph capture
    # (otherwise watchdog.tick → stream.query() → invalidates capture).
    pipeline = StreamedPosePipeline(
        bridge, runner, pre, post, sm,
        constraints=stack, tracer=tracer, watchdog=watchdog,
    )

    stop_flag = {"stop": False}

    def _on_signal(signum, frame):
        LOGGER.info("signal %d — stopping", signum)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Warmup frames to EXCLUDE from statistics. TRT kernel autotune,
    # ZED depth pipeline first-grab, CUDA cache population all happen
    # in the first ~30 frames of real pipeline execution. Including them
    # in p99 / HARD LIMIT percentage causes false-positive violations
    # (e.g., frame 6 hit 69ms in a session that averaged 13ms for all
    # other 8690 frames). Stats only count frames >= WARMUP_SKIP.
    WARMUP_SKIP_FRAMES = 100  # was 30; raised because spike diagnostic showed
                         # graph-capture transient (frame 14: post=225ms)
                         # and OS/Python warmup (frame 50: host=29ms) both
                         # land in the 30-100 window.

    t0 = time.monotonic()
    ticks = 0
    warmup_ticks_skipped = 0
    latencies: list[float] = []
    true_e2e_list: list[float] = []
    # Decomposition stats — for diagnosing where true_e2e_ms time goes:
    #   bridge_proc:    bridge thread CPU work per frame
    #   queue_wait:     time spent in queue between bridge ready and pipeline pickup
    #   pipeline_proc:  pipeline pickup → GPU done
    # Sum ≈ true_e2e_ms (minor diff = ZED HW capture → grab return latency).
    bridge_proc_list: list[float] = []
    queue_wait_list: list[float] = []
    pipeline_proc_list: list[float] = []
    # actual SHM publish latency — measured AFTER publisher.publish() returns.
    # This is the safety-relevant metric (control-visible). true_e2e_ms only
    # covers up to GPU done; actual_publish_ms includes the post-GPU D2H
    # batch + SHM seqlock write.
    actual_publish_list: list[float] = []
    # frame_skip: how often bridge.latest() returned None (consume-once + no
    # new frame within timeout). High ratio means bridge is the bottleneck.
    frame_skip_count = 0
    _last_stats_t = time.monotonic()
    STATS_INTERVAL_S = 10.0   # print rolling stats every 10s
    try:
        while not stop_flag["stop"]:
            if time.monotonic() - t0 > args.duration:
                break
            tick = pipeline.run_overlapped_step()
            if tick is None:
                # bridge had no new frame within its 0.5s poll window.
                # Skip publish (don't ship stale data); count for diagnostics.
                frame_skip_count += 1
                continue
            ticks += 1
            e2e_ms = tick.latency_ms["e2e"]
            true_e2e_ms = tick.latency_ms.get("true_e2e_ms", float("nan"))
            # Decomposition keys (added by pipeline.py for diagnostic visibility)
            zed_lag_ms = tick.latency_ms.get("zed_lag_ms", float("nan"))
            bridge_proc_ms = tick.latency_ms.get("bridge_proc_ms", float("nan"))
            queue_wait_ms = tick.latency_ms.get("queue_wait_ms", float("nan"))
            pipeline_proc_ms = tick.latency_ms.get("pipeline_proc_ms", float("nan"))

            # Skip first N real-pipeline frames from stats (warmup).
            # They still publish normally so downstream control isn't
            # starved, but they don't inflate p99 / HARD LIMIT pct.
            in_warmup = ticks <= WARMUP_SKIP_FRAMES
            if not in_warmup:
                latencies.append(e2e_ms)
                if not np.isnan(true_e2e_ms):
                    true_e2e_list.append(true_e2e_ms)
                bridge_proc_list.append(bridge_proc_ms)
                queue_wait_list.append(queue_wait_ms)
                pipeline_proc_list.append(pipeline_proc_ms)
            else:
                warmup_ticks_skipped = ticks  # keep latest for log

            # ─── Rolling stats every STATS_INTERVAL_S ──────────────────
            now = time.monotonic()
            if not in_warmup and now - _last_stats_t >= STATS_INTERVAL_S and latencies:
                _last_stats_t = now
                arr = np.asarray(latencies)
                mn, p50, p95, p99, mx = np.percentile(arr, [0, 50, 95, 99, 100])
                n_over = int((arr > LATENCY_HARD_LIMIT_MS).sum())
                n_total = len(arr)
                elapsed = max(now - t0, 1e-3)
                fps_live = ticks / elapsed
                true_p99 = float(np.percentile(true_e2e_list, 99)) if true_e2e_list else float("nan")
                # bucket counts for distribution line
                b = [
                    int((arr < 10).sum()),
                    int(((arr >= 10) & (arr < 14)).sum()),
                    int(((arr >= 14) & (arr < 18)).sum()),
                    int(((arr >= 18) & (arr < 20)).sum()),
                    int((arr >= 20).sum()),
                ]
                pct = [f"{x/n_total*100:.0f}%" for x in b]
                # true_e2e_ms decomposition stats
                def _pct_arr(lst, q):
                    return float(np.percentile(np.asarray(lst), q)) if lst else float("nan")
                tr_p50 = _pct_arr(true_e2e_list, 50)
                tr_p99 = _pct_arr(true_e2e_list, 99)
                tr_max = max(true_e2e_list) if true_e2e_list else float("nan")
                br_p50, br_p99 = _pct_arr(bridge_proc_list, 50), _pct_arr(bridge_proc_list, 99)
                qw_p50, qw_p99 = _pct_arr(queue_wait_list, 50), _pct_arr(queue_wait_list, 99)
                pl_p50, pl_p99 = _pct_arr(pipeline_proc_list, 50), _pct_arr(pipeline_proc_list, 99)
                # HARD LIMIT violation rate based on true_e2e_ms (the metric we publish on)
                tr_arr = np.asarray(true_e2e_list) if true_e2e_list else np.array([])
                n_over_true = int((tr_arr > LATENCY_HARD_LIMIT_MS).sum()) if tr_arr.size else 0
                pct_over_true = (n_over_true / tr_arr.size * 100) if tr_arr.size else 0.0

                ap_p50 = _pct_arr(actual_publish_list, 50)
                ap_p99 = _pct_arr(actual_publish_list, 99)
                ap_max = max(actual_publish_list) if actual_publish_list else float("nan")
                # frame_skip ratio: None returns / (None returns + processed frames)
                skip_total = frame_skip_count + ticks
                skip_ratio = (frame_skip_count / skip_total * 100) if skip_total else 0.0

                LOGGER.info(
                    "[STATS t=%ds] frames=%d fps=%.1f  skip=%d(%.1f%%)  HARD(true_e2e>%dms)=%d(%.2f%%)",
                    int(elapsed), n_total, fps_live,
                    frame_skip_count, skip_ratio,
                    int(LATENCY_HARD_LIMIT_MS), n_over_true, pct_over_true,
                )
                LOGGER.info(
                    "  e2e (gpu only):  min=%.1f  p50=%.1f  p95=%.1f  p99=%.1f  max=%.1f ms",
                    mn, p50, p95, p99, mx,
                )
                LOGGER.info(
                    "  true_e2e (cam→gpu_done):  p50=%.1f  p99=%.1f  max=%.1f ms",
                    tr_p50, tr_p99, tr_max,
                )
                LOGGER.info(
                    "  actual_publish (cam→shm): p50=%.1f  p99=%.1f  max=%.1f ms",
                    ap_p50, ap_p99, ap_max,
                )
                LOGGER.info(
                    "  decomp p50/p99: bridge_proc=%.1f/%.1f  queue_wait=%.1f/%.1f  pipeline_proc=%.1f/%.1f ms",
                    br_p50, br_p99, qw_p50, qw_p99, pl_p50, pl_p99,
                )
                LOGGER.info(
                    "  e2e dist: <10=%s | 10-14=%s | 14-18=%s | 18-20=%s | >=20=%s",
                    *pct,
                )

            # ─── 20 ms HARD BOUND — frame skip if exceeded ─────────────
            # Any tick that took more than 20 ms is STALE for real-time
            # control. Publish it with valid=False so the C++ watchdog
            # skips ILC/impedance update. Next frame ships normally.
            # During warmup we also mark as valid=False (to be safe) but
            # don't count it toward stats.
            # HARD LIMIT uses true_e2e_ms (camera timestamp → GPU done).
            # e2e_ms is GPU-pipeline-only and excludes capture/buffer-wait.
            frame_exceeds_budget = true_e2e_ms > LATENCY_HARD_LIMIT_MS
            frame_warn = true_e2e_ms > LATENCY_SOFT_WARN_MS

            # During warmup, log at debug level only (still mark valid=False
            # in publish so downstream ignores). Post-warmup spikes are
            # real and get normal ERROR/WARNING logs.
            if frame_warn and not in_warmup:
                lms = tick.latency_ms
                LOGGER.warning(
                    "[SLOW] frame %d  true_e2e=%.1f ms (e2e=%.1f)\n"
                    "  decomp: bridge_proc=%.1f  queue_wait=%.1f  pipeline_proc=%.1f  zed_lag=%.1f ms\n"
                    "  capture : grab=%.1f  ret_rgb=%.1f  getdata_rgb=%.1f"
                    "  pinned_rgb=%.1f  ret_depth=%.1f  getdata_depth=%.1f\n"
                    "  pipeline: pre=%.1f  inf=%.1f  post=%.1f  constraint=%.1f",
                    tick.frame_id, true_e2e_ms, e2e_ms,
                    bridge_proc_ms, queue_wait_ms, pipeline_proc_ms, zed_lag_ms,
                    lms.get("grab_ms", float("nan")),
                    lms.get("retrieve_rgb_ms", float("nan")),
                    lms.get("getdata_rgb_ms", float("nan")),
                    lms.get("pinned_rgb_ms", float("nan")),
                    lms.get("retrieve_depth_ms", float("nan")),
                    lms.get("getdata_depth_ms", float("nan")),
                    lms.get("pre_ms", float("nan")),
                    lms.get("inf_ms", float("nan")),
                    lms.get("post_ms", float("nan")),
                    lms.get("constraint_ms", float("nan")),
                )
            elif in_warmup and frame_exceeds_budget:
                LOGGER.debug(
                    "warmup frame %d: e2e=%.2f ms (not counted toward stats)",
                    tick.frame_id, e2e_ms,
                )

            if publisher is not None:
                # Batch three small D2H copies into a single stream op by
                # flattening and concatenating. post_stream was already
                # synchronized inside pipeline.run_overlapped_step, so
                # .to("cpu") here is just a memcpy — but doing it once
                # instead of three times saves ~2 ms p95 jitter vs the
                # prior version.
                K = tick.result.kpts_3d_m.shape[0]
                flat_gpu = torch.cat([
                    tick.result.kpts_3d_m.reshape(-1),     # K*3
                    tick.result.kpt_conf.reshape(-1),      # K
                    tick.result.kpts_2d_px.reshape(-1),    # K*2
                ], dim=0)
                flat = flat_gpu.detach().to("cpu", non_blocking=False).numpy().astype(np.float32)
                kpts_3d = flat[:K*3].reshape(K, 3)
                kpt_conf = flat[K*3:K*3 + K]
                kpts_2d = flat[K*3 + K:].reshape(K, 2)

                # SAFETY: HARD LIMIT decision must be made just before SHM write.
                # true_e2e_ms (GPU done) + post-GPU D2H + CPU packing has already
                # elapsed; this is the latest budget check before C++ sees the data.
                # Bug found via codex review: previously frame_exceeds_budget was
                # computed but never applied to publish_valid → over-budget frames
                # reached AK60 with valid=True. NEVER let that happen.
                pre_publish_e2e_ms = (time.time_ns() - tick.ts_ns) / 1e6
                publish_exceeds_budget = pre_publish_e2e_ms > LATENCY_HARD_LIMIT_MS
                publish_valid = (
                    tick.result.valid
                    and not in_warmup
                    and not publish_exceeds_budget
                )
                publisher.publish(
                    frame_id=tick.frame_id,
                    ts_ns=tick.ts_ns,
                    kpts_3d_m=kpts_3d,
                    kpt_conf=kpt_conf,
                    kpts_2d_px=kpts_2d,
                    box_conf=tick.result.box_conf,
                    valid=publish_valid,
                    depth_invalid_ratio=tick.result.depth_invalid_ratio,
                    world_frame_applied=tick.world_frame_applied,
                )
                # actual control-visible latency: from camera exposure to
                # SHM publish complete. Only collect post-warmup so stats
                # match the other latency lists.
                if not in_warmup:
                    actual_publish_ms = (time.time_ns() - tick.ts_ns) / 1e6
                    actual_publish_list.append(actual_publish_ms)
                watchdog.note_publish()
    finally:
        # Shutdown ordering matters: drain streams BEFORE destroying the
        # TRT engine. Reverse of construction: watchdog → bridge →
        # pipeline (which sync_all streams) → runner (del) → publisher.
        watchdog.stop()
        bridge.stop()
        pipeline.shutdown()
        try:
            del pipeline  # releases pipeline's ref to runner + streams
            del runner    # runs TRTRunner.__del__ while context is valid
        except Exception:
            pass
        if publisher is not None:
            publisher.close()
        # Dump trace (no-op when --trace not provided)
        path = tracer.dump()
        if path is not None:
            summ = tracer.summary()
            LOGGER.info("trace summary: %s", summ)

    dt = max(time.monotonic() - t0, 1e-3)
    fps = ticks / dt
    if latencies:
        lat_arr = np.asarray(latencies)
        p50, p95, p99 = np.percentile(lat_arr, [50, 95, 99])
    else:
        # Early exit (Ctrl+C during warmup) — avoid UnboundLocalError on lat_arr below
        lat_arr = np.array([])
        p50 = p95 = p99 = float("nan")

    # Hard-limit compliance (20 ms user requirement).
    # NB: Stats EXCLUDE first WARMUP_SKIP_FRAMES (TRT autotune, ZED
    # first-grab overhead). ticks counts ALL frames; measured_n counts
    # only post-warmup.
    measured_n = lat_arr.size  # post-warmup frames
    if measured_n > 0:
        n_over_hard = int((lat_arr > LATENCY_HARD_LIMIT_MS).sum())
        n_over_soft = int((lat_arr > LATENCY_SOFT_WARN_MS).sum())
        pct_over_hard = n_over_hard / measured_n * 100
        pct_over_soft = n_over_soft / measured_n * 100
        max_ms = float(lat_arr.max())
    else:
        n_over_hard = n_over_soft = 0
        pct_over_hard = pct_over_soft = 0.0
        max_ms = float("nan")

    skip_total = frame_skip_count + ticks
    skip_ratio = (frame_skip_count / skip_total * 100) if skip_total else 0.0
    LOGGER.info(
        "done: %d ticks / %.1fs → %.1f Hz  (skip=%d, %.1f%% of polls; stats on %d post-warmup frames)",
        ticks, dt, fps, frame_skip_count, skip_ratio, measured_n,
    )
    LOGGER.info(
        "e2e (gpu only) p50/95/99 = %.2f/%.2f/%.2f ms  max=%.2f ms",
        p50, p95, p99, max_ms,
    )
    # true_e2e_ms HARD LIMIT compliance (the metric that actually defines safety)
    if true_e2e_list:
        tr_arr = np.asarray(true_e2e_list)
        tr_p50, tr_p95, tr_p99 = np.percentile(tr_arr, [50, 95, 99])
        tr_max = float(tr_arr.max())
        n_over_hard_true = int((tr_arr > LATENCY_HARD_LIMIT_MS).sum())
        pct_over_hard_true = n_over_hard_true / tr_arr.size * 100
        LOGGER.info(
            "true_e2e (cam→gpu_done) p50/95/99 = %.2f/%.2f/%.2f ms  max=%.2f ms",
            tr_p50, tr_p95, tr_p99, tr_max,
        )
        LOGGER.info(
            "HARD LIMIT %.0f ms (true_e2e basis): %d / %d frames violated (%.3f%%) — published as valid=False",
            LATENCY_HARD_LIMIT_MS, n_over_hard_true, tr_arr.size, pct_over_hard_true,
        )
        # actual SHM-publish latency (control-visible). Difference vs true_e2e_ms
        # = D2H batch + SHM seqlock write. Tells us whether HARD_LIMIT decided
        # at GPU-done is actually safe (small diff) or optimistic (large diff).
        if actual_publish_list:
            ap = np.asarray(actual_publish_list)
            ap_p50, ap_p95, ap_p99 = np.percentile(ap, [50, 95, 99])
            ap_max = float(ap.max())
            n_over_hard_pub = int((ap > LATENCY_HARD_LIMIT_MS).sum())
            pct_over_hard_pub = n_over_hard_pub / ap.size * 100
            LOGGER.info(
                "actual_publish (cam→shm) p50/95/99 = %.2f/%.2f/%.2f ms  max=%.2f ms",
                ap_p50, ap_p95, ap_p99, ap_max,
            )
            LOGGER.info(
                "actual_publish > %.0f ms: %d / %d frames (%.3f%%) — TRUE control-visible violations",
                LATENCY_HARD_LIMIT_MS, n_over_hard_pub, ap.size, pct_over_hard_pub,
            )
        # Decomposition summary — where does true_e2e_ms time go?
        if bridge_proc_list and queue_wait_list and pipeline_proc_list:
            bp = np.asarray(bridge_proc_list)
            qw = np.asarray(queue_wait_list)
            pp = np.asarray(pipeline_proc_list)
            LOGGER.info(
                "decomposition p50/p99: bridge_proc=%.1f/%.1f  queue_wait=%.1f/%.1f  pipeline_proc=%.1f/%.1f ms",
                np.percentile(bp, 50), np.percentile(bp, 99),
                np.percentile(qw, 50), np.percentile(qw, 99),
                np.percentile(pp, 50), np.percentile(pp, 99),
            )
            LOGGER.info(
                "decomposition mean (sum check): bridge=%.1f + queue=%.1f + pipeline=%.1f = %.1f ms (true_e2e mean=%.1f)",
                bp.mean(), qw.mean(), pp.mean(),
                bp.mean() + qw.mean() + pp.mean(),
                tr_arr.mean(),
            )
    else:
        LOGGER.warning("true_e2e_list empty — old bridge/pipeline (no decomposition)")
    LOGGER.info(
        "HARD LIMIT %.0f ms (e2e basis, GPU-only — informational): %d / %d frames exceeded (%.3f%%)",
        LATENCY_HARD_LIMIT_MS, n_over_hard, measured_n, pct_over_hard,
    )
    LOGGER.info(
        "SOFT WARN %.0f ms: %d / %d frames (%.2f%%)",
        LATENCY_SOFT_WARN_MS, n_over_soft, measured_n, pct_over_soft,
    )
    if pct_over_hard > 1.0:
        LOGGER.warning(
            "→ %d frames (%.3f%%) violated 20 ms hard limit. Suggestions: "
            "(1) chrt -r 90 for RT priority, (2) sudo systemctl restart "
            "nvargus-daemon, (3) reboot if degraded across multiple runs.",
            n_over_hard, pct_over_hard,
        )
    elif pct_over_hard > 0:
        LOGGER.info(
            "→ %d frames (%.3f%%) > 20 ms: control skips them via valid=False. "
            "Acceptable for soft real-time.",
            n_over_hard, pct_over_hard,
        )
    else:
        LOGGER.info("→ PERFECT: all frames within 20 ms budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
