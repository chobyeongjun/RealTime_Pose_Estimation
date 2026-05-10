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
from .shm_publisher import (
    DEFAULT_NAME,
    INVALID_BUDGET_EXCEED,
    INVALID_UNKNOWN,
    INVALID_WARMUP,
    ShmPublisher,
    VALID_OK,
)
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
    ap.add_argument(
        "--mock-pipeline", action="store_true",
        help="A6 진단 — TRT/preproc/post를 mock으로 대체. bridge cycle 격리용. "
             "publish는 valid=False로 통과. 결과 의미 없음 (zeros). "
             "full pipeline의 bridge_proc 26ms vs bridge-only 8.2ms 차이 격리.",
    )
    ap.add_argument(
        "--idle-pipeline", action="store_true",
        help="A11 진단 — Pipeline thread (main thread) sleep만, "
             "bridge thread만 동작. 우리 코드 경로의 bridge cycle 격리. "
             "Pipeline 영향 0인 상태에서 bridge_proc/grab/retrieve 분포 측정.",
    )
    # Plan v7 (2026-05-07) — zed_lag 21ms 진단/격파 levers
    ap.add_argument(
        "--exposure-us", type=int, default=None,
        help="Plan v7 Round 1 — ZED exposure MANUAL microseconds. "
             "None=AUTO (default). 5000-12000 권장 (밝은 실내~어두움).",
    )
    ap.add_argument(
        "--sensing-mode", default="STANDARD",
        choices=["STANDARD", "FILL"],
        help="Plan v7 Round 3 — ZED depth sensing mode. STANDARD (default) = "
             "valid pixels only, FILL = hole filling (더 무거움).",
    )
    ap.add_argument(
        "--diag-zed-lag", action="store_true",
        help="Plan v7 Round 0 — warmup 첫 5 frames 의 ZED 다층 timestamp "
             "(IMAGE/CURRENT/bridge) 출력. zed_lag 21ms 격리.",
    )
    ap.add_argument(
        "--lpost-ablation", action="store_true",
        help="L_post Phase 0 ablation — post 안 .cpu() 제거 + sticky/EMA/"
             "constraints 모두 OFF. publish 직전 단일 통합 D2H. "
             "Codex Round 6 권장 spec. Codex H_post 가설 (post host block이 "
             "bridge에 영향) 양적 검증용. 측정 외 운영용 아님.",
    )
    # Phase 4 D1 + Phase 5 D1 (Codex R3+R4) — flag-based ablation:
    #   --frame-overlap : token-aware overlap cycle (submit/retire pattern)
    #   --post-async    : D2H async + retire branch (host sync 제거)
    # 8 조합 sweep 으로 어느 lever 진짜 효과 격리 (zedlag_sweep.sh combinations).
    ap.add_argument(
        "--frame-overlap", action="store_true",
        help="[DEPRECATED 2026-05-10] token-aware frame overlap cycle. "
             "12 case ablation sweep 측정 결과: 효과 0 + p99 +10-15ms 회귀 "
             "(packed D2H multi-inflight 충돌). PRODUCTION 사용 금지. "
             "측정 비교용으로만 유지.",
    )
    ap.add_argument(
        "--post-async", action="store_true",
        help="Phase 5 D1 — async D2H + retire branch. "
             "post() 의 .cpu() 제거, scalar D2H async + finalize_async() 가 retire "
             "시점에 .tolist() + sticky/EMA commit. --lpost-ablation 우선.",
    )
    ap.add_argument(
        "--zed-cuda-interop", action="store_true",
        help="γ Phase — ZED CUDA interop (shared_ctx path). "
             "ZED MEM::GPU + DLPack zero-copy. bridge_proc 14.4→8-10ms 추정.",
    )
    # A.3 (Codex consult 2026-05-10) — GPU stream priority ablation.
    # 현재 hardcoded `high_priority_stages=["infer"]` 를 flag 화 — off/infer-only/all-high
    # 비교로 priority lever 의 *실제 효과* 검증. Codex falsification: priority 모드 변경 시
    # inf_ms / pipeline_proc p99 차이 없으면 lever 효과 없음 (이미 active 상태 → 0ms 가능).
    ap.add_argument(
        "--gpu-stream-priority",
        choices=["off", "infer-only", "all-high"],
        default="infer-only",
        help="A.3 — CUDA stream priority. "
             "off=모든 stages low (baseline), "
             "infer-only=infer만 high (현재 default, hardcoded 였음), "
             "all-high=모든 stages high. "
             "stream_manager.py:115 가 high_priority_stages 인자에 따라 매핑.",
    )
    ap.add_argument(
        "--post-fusion", action="store_true",
        help="A.2 — fused post kernel (un-letterbox + 3D lift + EMA candidate single Triton kernel). "
             "post 7-step sequential (~7ms) → fused (~3-4ms 예상). "
             "Codex falsification: post p99 < 1.5ms 개선 안 되면 효과 없음. "
             "(★ NEXT — gpu_postprocess_fused.py 미작성 시 ImportError 로 fail-fast.)",
    )
    ap.add_argument(
        "--graph-extended", action="store_true",
        help="A.4 — CUDA graph 확장 (post 까지 capture). "
             "*--post-fusion 필수* (current post 가 graph-hostile scalar paths). "
             "Codex falsification: extended_graph_eager > 0 또는 actual_publish p99 변화 없음.",
    )
    # Codex R4 Q5 — sweep ablation 측정 시 constraints OFF 강제 (.item() 잔여 host
    # sync 가 post_async 효과 mask). lpost-ablation 와 별개 — flag combination
    # 의 모든 case 에 명시적으로 적용 가능.
    ap.add_argument(
        "--no-constraints", action="store_true",
        help="constraints stack 명시적 OFF (sweep 측정 시 Codex R4 권고). "
             "constraints.py:.item() 의 host sync 잔여 제거.",
    )
    # Codex review 발견 — runtime correctness assert 측정. log parsing 으로
    # 검증 못 하는 frame_id 단조 / ts 일치 / valid over-budget 즉시 catch.
    ap.add_argument(
        "--strict-correctness", action="store_true",
        help="Runtime assert: frame_id 단조 / tick.ts == tick.meta.ts / "
             "valid=True 시 budget 검증. fail 시 즉시 stop.",
    )
    args = ap.parse_args()
    # A.2 / A.4 fail-fast — 미구현 lever 가 silent ignore 되지 않게 (Codex consult Q1, 2026-05-10).
    if args.post_fusion:
        raise SystemExit(
            "--post-fusion: A.2 Triton kernel 미구현 (next PR). "
            "Codex Q1 spec — gpu_postprocess_fused.py 작성 필요. "
            "현재는 flag plumbing 만 (phase_a_sweep.sh case 3-8 의 스펙 보존)."
        )
    if args.graph_extended:
        raise SystemExit(
            "--graph-extended: A.4 미구현 (next PR — A.2 의존). "
            "current post 가 graph-hostile scalar paths — fusion 후에만 capture 가능."
        )
    return args


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
    # A.3 (2026-05-10) — flag 기반 stream priority ablation.
    # off=모두 low, infer-only=infer만 high (기존 hardcoded 동일), all-high=모두 high.
    if args.gpu_stream_priority == "off":
        _high_stages: list[str] | None = []
    elif args.gpu_stream_priority == "all-high":
        _high_stages = None
    else:
        _high_stages = ["infer"]
    LOGGER.info(
        "StreamManager priority — %s (high_stages=%s)",
        args.gpu_stream_priority, _high_stages if _high_stages else "ALL",
    )
    sm = StreamManager(device=device, high_priority_stages=_high_stages)
    runner = TRTRunner(args.engine, device=device)
    # Match preproc dtype to engine's input binding — Ultralytics engines
    # with --half still keep I/O as float32 by default, so probing here
    # avoids "dtype mismatch" at bind_input_address.
    engine_in_dtype = runner.bindings[runner.input_names[0]].dtype
    LOGGER.info("engine input dtype = %s, matching preproc accordingly", engine_in_dtype)
    pre = GpuPreprocessor(imgsz=args.imgsz, device=device, dtype=engine_in_dtype)
    # Phase 5 D1 — --post-async 와 --lpost-ablation 동시 사용 시 ablation 우선
    # (Codex R4): --post-async 는 ignore. log 만 남김.
    if args.lpost_ablation and args.post_async:
        LOGGER.info(
            "--post-async ignored: --lpost-ablation owns the post path "
            "(둘 다 ON 시 ablation path 가 winner)"
        )
    post = GpuPostprocessor(
        schema=schema,
        device=device,
        use_filter=args.use_filter,
        lpost_ablation=args.lpost_ablation,   # L_post Phase 0
        post_async=args.post_async,           # Phase 5 D1 (Codex R4)
    )
    stack = ConstraintStack()
    # constraints OFF 조건 (Codex R4 Q5 권고):
    #   --lpost-ablation : 기존 (post_async 와 별개로 sticky/EMA/constraints 모두 OFF)
    #   --no-constraints : NEW — sweep ablation 시 명시적 OFF. constraints.py 의
    #                      .item() host sync 잔여가 post_async 효과 mask 하지 않게.
    constraints_disabled = args.lpost_ablation or args.no_constraints
    if constraints_disabled:
        if args.bone_constraint or args.velocity_bound_mps > 0:
            reason = "lpost-ablation" if args.lpost_ablation else "no-constraints"
            LOGGER.warning(
                "--%s: ignoring --bone-constraint / --velocity-bound-mps "
                "(constraints stack disabled, .item() host sync 잔여 제거)",
                reason,
            )
    else:
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
        collect_cycle_stats=args.idle_pipeline,   # A11
        exposure_us=args.exposure_us,             # Plan v7 R1
        sensing_mode=args.sensing_mode,           # Plan v7 R3
        diag_zed_lag=args.diag_zed_lag,           # Plan v7 R0
        zed_cuda_interop=args.zed_cuda_interop,   # γ Phase (STUB)
    )
    bridge.open()
    bridge.start()

    # ───────────────────────────────────────────────────────────────────
    # A11 idle-pipeline mode — main thread sleeps, bridge runs alone.
    # Returns early before pipeline/publisher setup (those are unused).
    # ───────────────────────────────────────────────────────────────────
    if args.idle_pipeline:
        LOGGER.info(
            "=== A11 idle-pipeline mode — main thread sleeps for %.0fs, "
            "bridge runs alone ===",
            args.duration,
        )
        try:
            time.sleep(args.duration)
        finally:
            bridge.stop()
        stats = bridge.get_cycle_stats()
        if not stats:
            LOGGER.error("A11: no cycle stats collected")
            return
        n = len(stats)
        first_ts = stats[0]["ts_ns"]
        last_ts = stats[-1]["ts_ns"]
        actual_dur = (last_ts - first_ts) / 1e9 if last_ts > first_ts else 0.0
        hz = (n - 1) / actual_dur if actual_dur > 0 else 0.0
        LOGGER.info(
            "=== A11 results: %d frames in %.1fs (ts-based) = %.1f Hz ===",
            n, actual_dur, hz,
        )

        def _pct(values, q):
            return float(np.percentile(np.asarray(values), q))

        for key in (
            "grab_ms", "retrieve_rgb_ms", "getdata_rgb_ms", "pinned_rgb_ms",
            "retrieve_depth_ms", "getdata_depth_ms", "bridge_proc_ms",
        ):
            vals = [s[key] for s in stats]
            arr = np.asarray(vals)
            LOGGER.info(
                "  %-20s min=%5.2f p50=%5.2f p95=%5.2f p99=%5.2f max=%5.2f mean=%5.2f ms",
                key,
                float(arr.min()), _pct(vals, 50), _pct(vals, 95),
                _pct(vals, 99), float(arr.max()), float(arr.mean()),
            )
        # delta_ts (frame interval, frame N → N+1)
        if n >= 2:
            delta = [(stats[i]["ts_ns"] - stats[i-1]["ts_ns"]) / 1e6 for i in range(1, n)]
            arr = np.asarray(delta)
            LOGGER.info(
                "  %-20s min=%5.2f p50=%5.2f p95=%5.2f p99=%5.2f max=%5.2f mean=%5.2f ms",
                "delta_ts_ms",
                float(arr.min()), _pct(delta, 50), _pct(delta, 95),
                _pct(delta, 99), float(arr.max()), float(arr.mean()),
            )

        LOGGER.info("A11 idle-pipeline done. Exiting.")
        return

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
    # [DEPRECATED] --frame-overlap: 12 case ablation 측정 결과 효과 0 + p99
    # +10-15ms 회귀. PRODUCTION 사용 금지. 측정 비교용으로만 유지.
    if args.frame_overlap:
        LOGGER.warning(
            "[DEPRECATED] --frame-overlap: 12 case ablation 측정 결과 effect 0 + "
            "p99 +10-15ms 회귀 (packed D2H multi-inflight 충돌). "
            "PRODUCTION 사용 금지. 측정 비교용으로만 유지. "
            "권장 path: --zed-cuda-interop --post-async (case 10, true_e2e p99 ~62ms)"
        )
    pipeline = StreamedPosePipeline(
        bridge, runner, pre, post, sm,
        constraints=stack, tracer=tracer, watchdog=watchdog,
        frame_overlap=args.frame_overlap,
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
    # Codex review fix — runtime correctness check (--strict-correctness).
    # log parsing 으로 catch 못 하는 race 즉시 raise.
    _strict_prev_frame_id = -1
    HARD_LIMIT_TRUE_E2E_MS = 200.0   # over-budget assertion threshold (ms)
    try:
        while not stop_flag["stop"]:
            if time.monotonic() - t0 > args.duration:
                break
            tick = (
                pipeline.run_overlapped_step_mock()
                if args.mock_pipeline
                else pipeline.run_overlapped_step()
            )
            if tick is None:
                # bridge had no new frame within its 0.5s poll window.
                # Skip publish (don't ship stale data); count for diagnostics.
                frame_skip_count += 1
                continue
            ticks += 1
            # Codex strict-correctness — runtime assert.
            if args.strict_correctness:
                # 1) frame_id 단조 증가 (P4D1 latest-only pickup 시 N → N+k 가능, k>=1)
                assert tick.frame_id > _strict_prev_frame_id, (
                    f"frame_id not strictly increasing: "
                    f"{_strict_prev_frame_id} → {tick.frame_id}"
                )
                _strict_prev_frame_id = tick.frame_id
                # 2) ts 일치 (P1D1 contract — meta 가 ground truth)
                if tick.meta is not None:
                    assert tick.ts_ns == tick.meta.ts_ns, (
                        f"ts mismatch: tick.ts_ns={tick.ts_ns} "
                        f"meta.ts_ns={tick.meta.ts_ns}"
                    )
                # 3) over-budget 시 valid=False 강제 (publish 직전 다시 체크하므로
                #    여기는 *비현실적 over-budget* 만 catch — system clock 이상 등)
                _now_ms = (time.time_ns() - tick.ts_ns) / 1e6
                if tick.result.valid and _now_ms > HARD_LIMIT_TRUE_E2E_MS:
                    raise AssertionError(
                        f"valid=True but true_e2e {_now_ms:.1f}ms > "
                        f"HARD_LIMIT {HARD_LIMIT_TRUE_E2E_MS}ms (frame {tick.frame_id})"
                    )
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
                # Batch D2H — single sync, packed transfer.
                K = tick.result.kpts_3d_m.shape[0]
                if args.lpost_ablation:
                    # L_post Phase 0 ablation path: post returned GPU scalars.
                    # Codex Round 7 fix: assert tensor fields present (flag/field
                    # mismatch is a programming error, not silent fallback).
                    if tick.result.valid_mask_t is None:
                        raise RuntimeError(
                            "--lpost-ablation set but PoseResult.valid_mask_t is None — "
                            "GpuPostprocessor not in ablation mode?"
                        )
                    # Codex Round 7 fix: explicit FP32 cast on every cat input.
                    # TRT output dtype could be FP16 in some configurations; mixing
                    # dtypes silently relies on PyTorch promotion rules — risky
                    # for real-time deterministic behavior.
                    flat_gpu = torch.cat([
                        tick.result.kpts_3d_m.float().reshape(-1),                   # K*3
                        tick.result.kpt_conf.float().reshape(-1),                    # K
                        tick.result.kpts_2d_px.float().reshape(-1),                  # K*2
                        tick.result.box_conf_t.float().reshape(-1),                  # 1
                        tick.result.valid_mask_t.to(dtype=torch.float32).reshape(-1),# 1
                        tick.result.depth_invalid_ratio_t.float().reshape(-1),       # 1
                    ], dim=0)
                    flat = flat_gpu.detach().cpu().numpy().astype(np.float32, copy=False)
                    kpts_3d = flat[:K*3].reshape(K, 3)
                    kpt_conf = flat[K*3:K*3 + K]
                    kpts_2d = flat[K*3 + K:K*3 + K + K*2].reshape(K, 2)
                    # Resolve scalars on CPU (single sync above already done)
                    gpu_box_conf = float(flat[K*3 + K + K*2])
                    gpu_valid = bool(flat[K*3 + K + K*2 + 1] > 0.5)
                    gpu_depth_invalid = float(flat[K*3 + K + K*2 + 2])
                    # Stamp back onto tick.result so SLOW logging / publish
                    # downstream see real values (fields started as placeholders).
                    tick.result.box_conf = gpu_box_conf
                    tick.result.valid = gpu_valid
                    tick.result.depth_invalid_ratio = gpu_depth_invalid
                else:
                    # Standard path — kpts only, scalars already CPU floats.
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

                # P1 (2026-05-06): valid_reason classification.
                # Order matters — warmup is the most common reason early on, then
                # budget exceed, then post-stage / constraint rejection. Without a
                # PoseResult.reason field we can only classify the gates we own
                # here; pure post-stage rejections (low-conf detection, occluded
                # joints, constraint reject) collapse to INVALID_UNKNOWN until a
                # follow-up patch threads the reason through.
                if publish_valid:
                    valid_reason = VALID_OK
                elif in_warmup:
                    valid_reason = INVALID_WARMUP
                elif publish_exceeds_budget:
                    valid_reason = INVALID_BUDGET_EXCEED
                else:
                    valid_reason = INVALID_UNKNOWN

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
                    valid_reason=valid_reason,
                )
                # actual control-visible latency: from camera exposure to
                # SHM publish complete. Only collect post-warmup so stats
                # match the other latency lists.
                if not in_warmup:
                    actual_publish_ms = (time.time_ns() - tick.ts_ns) / 1e6
                    actual_publish_list.append(actual_publish_ms)
                watchdog.note_publish()
    finally:
        # Diagnostic counter dump (Codex R11, 2026-05-06) BEFORE destruction.
        # If graph_replay_count == frame_count and eager_count == 0 → graph
        # is healthy → 6.6ms overhead is NOT H1.
        # If set_address_count >> n_io after warmup → bind cache broken → H2.
        try:
            inf_graph = getattr(pipeline, "_inf_graph", None)
            if inf_graph is not None:
                LOGGER.info(
                    "[diag] inf_graph captured=%s replay=%d eager=%d "
                    "set_address=%d frames=%d",
                    inf_graph.captured,
                    inf_graph.replay_count,
                    inf_graph.eager_count,
                    runner.set_address_count,
                    pipeline._frame_count,
                )
            else:
                LOGGER.info(
                    "[diag] inf_graph=None set_address=%d frames=%d",
                    runner.set_address_count,
                    pipeline._frame_count,
                )
        except Exception as exc:
            LOGGER.warning("[diag] counter dump failed: %s", exc)

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
