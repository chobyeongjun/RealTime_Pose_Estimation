"""Triple-buffer 3-stage overlapped pipeline.

At any instant:
    capture_stream   : grabbing frame N+1
    preproc_stream   : letterbox/normalize frame N+1 (after capture done)
    infer_stream     : TRT inference on frame N
    post_stream      : 3D/filter/publish frame N-1

Cross-stream dependencies flow via ``torch.cuda.Event`` only. Host thread
steps once per frame: pick latest ZEDFrame → advance events → wait on the
post_stream event that corresponds to the frame we want to return.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch

from .constraints import ConstraintStack
from .cuda_graph import GraphedStep
from .gpu_postprocess import GpuPostprocessor, PoseResult
from .gpu_preprocess import GpuPreprocessor, LetterboxParams
from .stream_manager import StreamManager
from .tracer import FrameTrace, PipelineTracer
from .trt_runner import TRTRunner
from .zed_gpu_bridge import ZEDFrame, ZEDGpuBridge

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameMeta:
    """Read-only frame metadata. Immutable across pipeline stages.

    All ts_ns fields share the epoch-ns clock domain (``time.time_ns()``):
      ts_ns           — sensor exposure time (ZED hardware capture)
      bridge_start_ns — bridge thread began processing (just after grab returns)
      ready_ns        — bridge finished H2D launch + put frame in queue
      pickup_ns       — pipeline received frame from bridge (just before pre)

    Phase 1 contract (Codex R2): metadata MUST live on the FrameMeta token,
    not interleaved with latency_ms dict — so that frame-token lifetime is
    the single source of truth across submit → retire boundaries.
    """

    frame_id: int
    ts_ns: int
    bridge_start_ns: int
    ready_ns: int
    pickup_ns: int
    capture_ms: dict


@dataclass
class PipelineToken:
    """In-flight frame state for Phase 4 frame overlap (currently UNUSED).

    Each token represents one frame's lifecycle from submit (pickup) to
    retire (publish-ready). Phase 1 Day 1 declares the type; Phase 4
    (frame overlap) populates ``self._pending: Deque[PipelineToken]``.

    Per-token Events (Codex R2 #11): each frame gets its own pre/inf/
    snapshot/post events — eliminates the bundle.done_event re-record race
    in multi-inflight mode. Non-timing events (``enable_timing=False``)
    for stream dependencies; tracer keeps timing events separately.
    """

    seq: int                              # monotonic submit counter
    frame: ZEDFrame                       # strong ref → keeps depth_gpu/calibration alive
    meta: FrameMeta
    lb_params: Optional[LetterboxParams]
    output_snapshot_idx: int              # index into output snapshot ring
    raw_snapshot: torch.Tensor            # owned snapshot of TRT output (D2D copied)
    pre_done: torch.cuda.Event
    inf_done: torch.cuda.Event
    snapshot_done: torch.cuda.Event
    post_done: torch.cuda.Event
    trace: FrameTrace
    result: Optional[PoseResult] = None
    submit_ns: int = 0
    retire_ns: int = 0


@dataclass
class PipelineTick:
    """Output of one pipeline step — what the host consumer receives.

    Phase 1 (Codex R2 #10): ``meta`` and ``trace`` fields ADDED with
    ``Optional[...] = None`` defaults so existing callers continue to work
    unchanged. Phase 1 Steps 2-4 will populate ``meta``; Phase 4 will use
    ``trace`` per-token. Until then both may be ``None``.
    """

    frame_id: int
    ts_ns: int
    result: PoseResult
    latency_ms: dict  # {"e2e", "true_e2e_ms", pre/inf/post stage_ms, ...}
    meta: Optional[FrameMeta] = None
    world_frame_applied: bool = False
    trace: Optional[FrameTrace] = None


class StreamedPosePipeline:
    """3-stage GPU pipeline with 4 streams."""

    def __init__(
        self,
        bridge: ZEDGpuBridge,
        runner: TRTRunner,
        preprocessor: GpuPreprocessor,
        postprocessor: GpuPostprocessor,
        streams: StreamManager,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        constraints: Optional[ConstraintStack] = None,
        tracer: Optional[PipelineTracer] = None,
        watchdog: Optional[Any] = None,
    ) -> None:
        self.bridge = bridge
        self.runner = runner
        self.pre = preprocessor
        self.post = postprocessor
        self.sm = streams
        # Default: no constraints (OFF). Callers opt in by passing a
        # ConstraintStack with calibrated bone_length / joint_velocity.
        # See constraints.py for the rationale.
        self.constraints = constraints or ConstraintStack()
        # Tracer is OFF by default — pass one from run_stream_demo/benchmark_stream
        # with --trace to enable per-stage CUDA-event timing + CSV dump.
        self.tracer = tracer or PipelineTracer(enabled=False)

        # Watchdog reference — needed to pause its stream.query() polling
        # during CUDA graph capture (each query is a CUDA API call and
        # would invalidate the capture with cudaErrorStreamCaptureUnsupported).
        self._watchdog = watchdog

        self._input = input_name or runner.input_names[0]
        self._output = output_name or runner.output_names[0]

        # In-flight bookkeeping for Phase 4 frame overlap (Codex R3).
        # NOTE: ``maxlen`` 금지 — auto-drop 시 live PipelineToken (CUDA event +
        # frame tensor strong ref) 가 silently freed. submit 시 수동으로
        # ``len(_pending) >= self._output_ring_size`` check 로 backpressure.
        # Phase 1 D1-D3 까지 미사용. flag --frame-overlap 활성 시 채움.
        self._pending: Deque["PipelineToken"] = deque()

        # Prime every stream's done_event once so any consumer that calls
        # ``wait_for(X)`` on frame 0 has a defined event to wait on.
        for bundle in streams.streams.values():
            bundle.record_done()

        # CUDA Graph capture for TRT inference — replaces hundreds of
        # cudaLaunchKernel calls with a single graph replay. Captured
        # lazily after warmup so TRT JIT/tactic selection is settled.
        # If capture fails (TRT/driver mismatch), eager fallback runs
        # the same code path with no functional difference.
        self._inf_graph: Optional[GraphedStep] = None
        self._frame_count = 0
        self._graph_warmup_frames = 30
        self._graph_attempted = False

        # Phase 1 Day 3 (Codex R2 #4) — output snapshot ring (size 2).
        # Race protection (Phase 4 prerequisite): inf 가 raw_output (single
        # binding tensor) 에 write 한 직후 D2D copy 로 snapshot[i] 에 옮겨
        # 놓고, post 가 snapshot 을 read. 다음 frame 의 inf 가 raw_output
        # 덮어써도 post 의 read 는 안전.
        # Ring size 2 = inf[N+1] || post[N] 가 다른 slot 사용 (alternate i, i+1).
        # Phase 1 D3: 메모리만 할당 (D2D copy 는 P1D4 에서 추가, cycle 재구조는 P4D1).
        # Phase 4: PipelineToken.output_snapshot_idx 가 token 별 slot 결정.
        output_tensor = self.runner.get_output(self._output)
        self._output_ring_size = 2
        self._output_ring: list[torch.Tensor] = [
            torch.empty_like(output_tensor) for _ in range(self._output_ring_size)
        ]
        self._ring_idx = 0

        # Phase 1 Day 3 — last inf_done event 보관 (Phase 4 prerequisite).
        # Phase 4 시 다음 frame 의 pre 가 ``pre.stream.wait_event(self._last_inf_done)``
        # 으로 self.pre.out (single input buffer) 보호.
        # Single-frame mode (현재) 에서는 미사용 — cycle 끝의 po.stream.synchronize()
        # 가 자연 보호.
        self._last_inf_done: Optional[torch.cuda.Event] = None

        # (Reserved for future fallback use — currently constraint rejects
        # emit zeros+valid=False instead of using stale data. Keeping this
        # as a field documented for future work; DO NOT read it elsewhere
        # without first deciding how it interacts with the valid=False
        # safety contract.)

    # ------------------------------------------------------------------
    # Single step (for benchmarks / reference correctness)
    # ------------------------------------------------------------------
    def run_once(self, frame: ZEDFrame) -> PipelineTick:
        """Run a single frame end-to-end in a way the caller can verify.

        This serializes stages but still uses explicit streams / events.
        Used by tests and the `--no-overlap` flag in the benchmark.
        """
        # Phase 1 Day 1 (Codex R2): capture pickup_ns at function entry —
        # epoch-ns counterpart of t_start (perf_counter). FrameMeta below
        # consumes pickup_ns so true_e2e/queue_wait math stays consistent
        # with run_overlapped_step.
        pickup_ns = time.time_ns()
        t_start = time.perf_counter()
        cap = self.sm.bundle("capture")
        pre = self.sm.bundle("preproc")
        inf = self.sm.bundle("infer")
        po = self.sm.bundle("post")

        # Preproc must wait for ZED's private H2D stream to finish the copy.
        if frame.ready_event is not None:
            pre.stream.wait_event(frame.ready_event)
        else:
            pre.wait_for(cap)
        _, lb = self.pre(frame.rgb_gpu, stream=pre.stream)
        # bind preproc output as TRT input (zero-copy)
        self.runner.bind_input_address(self._input, self.pre.out)
        pre.record_done()

        inf.wait_for(pre)
        with torch.cuda.stream(inf.stream):
            self.runner.infer_async(self.sm.stream_ptr("infer"))
        inf.record_done()

        po.wait_for(inf)
        result = self.post(
            raw_output=self.runner.get_output(self._output),
            depth_hw=frame.depth_gpu,
            lb_params=lb,
            calibration=frame.calibration,
            stream=po.stream,
            ts_s=frame.ts_ns * 1e-9,
        )
        po.record_done()
        po.stream.synchronize()

        t_end = time.perf_counter()
        world_frame_applied = "R_world_from_cam" in frame.calibration
        # Phase 1 Day 1 (Codex R2 #10): emit FrameMeta token alongside latency_ms.
        # Same data, two views: latency_ms keeps the consumer-friendly ms scalars,
        # FrameMeta keeps the raw epoch-ns clock for cross-stage correlation.
        meta = FrameMeta(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            bridge_start_ns=frame.bridge_start_ns,
            ready_ns=frame.ready_ns,
            pickup_ns=pickup_ns,
            capture_ms=dict(frame.capture_ms),
        )
        return PipelineTick(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            result=result,
            world_frame_applied=world_frame_applied,
            latency_ms={
                "e2e": (t_end - t_start) * 1e3,
                "true_e2e_ms": (time.time_ns() - frame.ts_ns) / 1e6,
            },
            meta=meta,
        )

    # ------------------------------------------------------------------
    # CUDA Graph capture — one-shot after warmup
    # ------------------------------------------------------------------
    def _try_capture_inf_graph(self, inf_bundle) -> None:
        """Capture the TRT inference call as a CUDA graph for cheap replay.

        On Orin NX TRT 10.x ``execute_async_v3`` queues hundreds of small
        kernels. Each cudaLaunchKernel adds ~10µs of CPU/driver overhead,
        and ``trtexec`` shows ~2.2ms enqueue time. Replaying a captured
        graph reduces this to a single launch (~50µs).

        Capture must happen AFTER:
          - TRT engine warmup (tactic selection)
          - Input binding pointer is stable (we cache it now in trt_runner)

        Failure path: GraphedStep falls back to eager execution. Same
        result, same correctness — just slower.
        """
        if self._graph_attempted:
            return
        self._graph_attempted = True

        inf_stream_ptr = self.sm.stream_ptr("infer")
        # Make sure address is bound BEFORE capture; trt_runner caches
        # so subsequent calls are no-ops, but the first call mutates
        # context state which must happen outside the graph.
        self.runner.bind_input_address(self._input, self.pre.out)

        def _infer_only() -> None:
            self.runner.infer_async(inf_stream_ptr)

        graph = GraphedStep(stream=inf_bundle.stream, fn=_infer_only, warmup=2)

        # CRITICAL: pause watchdog during capture. Watchdog polls
        # stream.query() every 5ms in a separate thread — each call is a
        # CUDA API call that invalidates an in-progress capture
        # (cudaErrorStreamCaptureUnsupported). This was the root cause of
        # 'attempt 1/3 failed... 2/3 failed... 3/3 failed' even with
        # capture_error_mode='thread_local' (same process, same thread
        # restriction wasn't enough — watchdog runs in same process).
        if self._watchdog is not None:
            self._watchdog.pause()
        try:
            if graph.try_capture():
                self._inf_graph = graph
                LOGGER.info(
                    "CUDA graph capture SUCCESS — TRT inference replays in 1 launch"
                )
        except RuntimeError as err:
            # try_capture now raises after exhausting retries (was silent
            # fallback before — caused non-reproducible 80Hz vs 40Hz runs).
            # We catch here to keep correctness, but log loudly so the user
            # knows their run is in slow eager mode (~40Hz vs 80Hz with graph).
            LOGGER.error(
                "CUDA graph capture FAILED after retries: %s\n"
                "  → Pipeline running in EAGER mode (~40Hz vs 80Hz with graph).\n"
                "  → To recover graph: 'sudo pkill -9 python3', then retry.\n"
                "  → Root cause if 'cudaErrorStreamCaptureInvalidated': "
                "another GPU process (often ZED other-thread) interfered.",
                err,
            )
            self._inf_graph = None
        finally:
            # Always resume watchdog, even if capture raised mid-way.
            if self._watchdog is not None:
                self._watchdog.resume()

    # ------------------------------------------------------------------
    # Overlapped run — the real deal
    # ------------------------------------------------------------------
    def run_overlapped_step_mock(self) -> Optional[PipelineTick]:
        """A6 (2026-05-06) — bridge-only 격리용 mock. GPU 작업 모두 skip.

        목적: full pipeline의 bridge cycle 26ms vs bridge_only_bench의 8.2ms
        차이 (+18ms 적체)의 원인을 격리.
          - mock에서 bridge cycle 회복 → TRT/preproc/post가 진짜 원인 (H1-H4)
          - mock에서도 26ms 그대로 → bridge thread 자체 문제 (H5)

        이 mock은 frame을 받아 *즉시 zeros PoseResult* 반환. preproc/TRT/post
        호출 없음. decomp 측정만 정상 작동 (zed_lag, bridge_proc, queue_wait,
        pipeline_proc).
        """
        frame = self.bridge.latest(timeout=0.5)
        if frame is None:
            return None

        pickup_ns = time.time_ns()
        self._frame_count += 1

        # GPU 작업 모두 skip — zeros PoseResult (publish는 valid=False로 통과)
        K = self.post.K
        zeros_3d = torch.zeros((K, 3), device=self.post.device)
        zeros_2d = torch.zeros((K, 2), device=self.post.device)
        zeros_conf = torch.zeros((K,), device=self.post.device)
        result = PoseResult(
            kpts_2d_px=zeros_2d,
            kpts_3d_m=zeros_3d,
            kpt_conf=zeros_conf,
            box_conf=0.0,
            valid=False,
            depth_invalid_ratio=1.0,
        )

        t_gpu_done_ns = time.time_ns()

        # decomp 측정 (full pipeline과 동일 anchor)
        zed_lag_ms = (frame.bridge_start_ns - frame.ts_ns) / 1e6 if frame.bridge_start_ns else 0.0
        bridge_proc_ms = (frame.ready_ns - frame.bridge_start_ns) / 1e6 if frame.bridge_start_ns and frame.ready_ns else 0.0
        queue_wait_ms = (pickup_ns - frame.ready_ns) / 1e6 if frame.ready_ns else 0.0
        pipeline_proc_ms = (t_gpu_done_ns - pickup_ns) / 1e6

        # Phase 1 Day 1 (Codex R2 #10): emit FrameMeta in mock too —
        # decomp diagnostic 동일 anchor 유지 + token-style metadata 일관성.
        meta = FrameMeta(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            bridge_start_ns=frame.bridge_start_ns,
            ready_ns=frame.ready_ns,
            pickup_ns=pickup_ns,
            capture_ms=dict(frame.capture_ms),
        )
        return PipelineTick(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            result=result,
            world_frame_applied=False,
            latency_ms={
                "e2e": 0.0,
                "constraint_ms": 0.0,
                "true_e2e_ms": (t_gpu_done_ns - frame.ts_ns) / 1e6,
                "zed_lag_ms": zed_lag_ms,
                "bridge_proc_ms": bridge_proc_ms,
                "queue_wait_ms": queue_wait_ms,
                "pipeline_proc_ms": pipeline_proc_ms,
                **frame.capture_ms,
            },
            meta=meta,
        )

    def run_overlapped_step(self) -> Optional[PipelineTick]:
        """Consume the latest ZED frame, advance streams, return last finished."""
        frame = self.bridge.latest(timeout=0.5)
        if frame is None:
            return None

        # pickup_ns: epoch-ns when pipeline received the frame from bridge.
        # This is the boundary between "queue wait" (bridge ready → pipeline
        # pickup) and "pipeline processing" (pickup → GPU done).
        pickup_ns = time.time_ns()

        self._frame_count += 1
        cap = self.sm.bundle("capture")
        pre = self.sm.bundle("preproc")
        inf = self.sm.bundle("infer")
        po = self.sm.bundle("post")

        self.tracer.begin(frame_id=frame.frame_id, ts_ns=frame.ts_ns)
        t_start = time.perf_counter()

        # --- stage A: preproc (on preproc_stream; rgb is already GPU)
        # Wait on the ZED bridge's H2D completion event, not on cap.
        if frame.ready_event is not None:
            pre.stream.wait_event(frame.ready_event)
        else:
            pre.wait_for(cap)
        # NOTE: cap_ms is intentionally NOT tracked here — the ZED H2D
        # happens on a stream inside ZEDGpuBridge that we don't own.
        # e2e_ms minus (pre+inf+post) approximates the capture overhead.
        self.tracer.mark_start("pre", pre.stream)
        _, lb = self.pre(frame.rgb_gpu, stream=pre.stream)
        self.tracer.mark_end("pre", pre.stream)
        pre.record_done()

        # --- stage B: infer (binds preproc output, waits on preproc)
        inf.wait_for(pre)
        # bind_input_address now caches per-pointer (TRT 10.x context state
        # mutation is expensive). Effective cost: ~0 after the first frame.
        self.runner.bind_input_address(self._input, self.pre.out)

        # One-shot graph capture after warmup — settles TRT JIT first.
        if (
            self._frame_count == self._graph_warmup_frames
            and self._inf_graph is None
        ):
            self._try_capture_inf_graph(inf)

        self.tracer.mark_start("inf", inf.stream)
        if self._inf_graph is not None and self._inf_graph.captured:
            # PyTorch 2.x CUDAGraph.replay() uses getCurrentCUDAStream(),
            # NOT the internally stored capture stream. Without this context
            # manager the graph launches on stream 0 — both timing events
            # and inf.record_done() fire with no real work between them,
            # giving inf=0ms and corrupting the post-stage dependency.
            with torch.cuda.stream(inf.stream):
                self._inf_graph.replay()
        else:
            with torch.cuda.stream(inf.stream):
                self.runner.infer_async(self.sm.stream_ptr("infer"))
        self.tracer.mark_end("inf", inf.stream)
        inf.record_done()

        # --- stage C: post (waits on infer)
        po.wait_for(inf)
        self.tracer.mark_start("post", po.stream)
        result = self.post(
            raw_output=self.runner.get_output(self._output),
            depth_hw=frame.depth_gpu,
            lb_params=lb,
            calibration=frame.calibration,
            stream=po.stream,
            ts_s=frame.ts_ns * 1e-9,
        )
        self.tracer.mark_end("post", po.stream)
        po.record_done()
        po.stream.synchronize()  # only sync point in the hot path
        t_end = time.perf_counter()        # GPU pipeline only (pre+inf+post)
        t_gpu_done_ns = time.time_ns()     # true_e2e anchor — before constraint CPU

        # --- stage D: optional constraint gate + occlusion fallback
        # Runs AFTER t_end / t_gpu_done_ns so constraint CPU overhead does
        # not inflate either e2e or true_e2e_ms.
        world_frame_applied = "R_world_from_cam" in frame.calibration
        # L_post Phase 0 (2026-05-06) — when ablation is active, skip constraints
        # (they call .item() which would re-introduce host sync) and skip
        # tracer scalar meta (same reason). Final valid is decided in publish
        # path after the single packed D2H.
        # Codex Round 7 fix: detect via post.lpost_ablation (canonical flag),
        # not via field presence. Field-based detection is fragile — future
        # changes might accidentally populate *_t fields and silently skip
        # constraints. Strict assert catches mismatch immediately.
        ablation_active = self.post.lpost_ablation
        if ablation_active and result.valid_mask_t is None:
            raise RuntimeError(
                "GpuPostprocessor.lpost_ablation=True but result.valid_mask_t is None — "
                "ablation early-return path not taken"
            )
        if not ablation_active and result.valid_mask_t is not None:
            raise RuntimeError(
                "GpuPostprocessor.lpost_ablation=False but result has valid_mask_t — "
                "GPU scalar fields populated outside ablation mode"
            )
        if ablation_active:
            final_result = result
            constraint_ms = 0.0
            # Codex Round 7 fix: marker values (-1) instead of zero placeholders.
            # Trace consumers can detect ablation rows by these sentinels and
            # not misread them as legitimate metric values.
            self.tracer.set_result_meta(
                valid=False,
                occluded_count=-1,         # marker: ablation, real value not available
                depth_invalid_ratio=-1.0,  # marker
                box_conf=-1.0,             # marker
            )
        else:
            t_constraint = time.perf_counter()
            final_result = self._apply_constraints_and_fallback(
                result, ts_s=frame.ts_ns * 1e-9
            )
            constraint_ms = (time.perf_counter() - t_constraint) * 1e3

            # Emit trace AFTER synchronize so elapsed_time is safe to read.
            self.tracer.set_result_meta(
                valid=final_result.valid,
                occluded_count=int((final_result.kpt_conf < self.post.kpt_conf_threshold).sum().item())
                if final_result.valid else 0,
                depth_invalid_ratio=final_result.depth_invalid_ratio,
                box_conf=final_result.box_conf,
            )
        trace = self.tracer.end()

        # ─── true_e2e_ms decomposition (diagnostic visibility) ─────────
        # All four are in epoch-ns clock domain (same as frame.ts_ns).
        #   zed_lag        = ZED hardware capture → bridge began processing
        #   bridge_proc    = bridge thread CPU work (grab → H2D launched)
        #   queue_wait     = bridge ready → pipeline pickup
        #   pipeline_proc  = pipeline pickup → GPU work done
        # Sum ≈ true_e2e_ms (any small residual = clock drift / measurement gap).
        zed_lag_ms = (frame.bridge_start_ns - frame.ts_ns) / 1e6 if frame.bridge_start_ns else 0.0
        bridge_proc_ms = (frame.ready_ns - frame.bridge_start_ns) / 1e6 if frame.bridge_start_ns and frame.ready_ns else 0.0
        queue_wait_ms = (pickup_ns - frame.ready_ns) / 1e6 if frame.ready_ns else 0.0
        pipeline_proc_ms = (t_gpu_done_ns - pickup_ns) / 1e6

        # Phase 1 Day 1 (Codex R2 #10): emit FrameMeta token alongside latency_ms.
        # FrameMeta is the *single source of truth* for frame metadata across
        # submit → retire boundaries (Phase 4). latency_ms keeps consumer-friendly
        # ms scalars; FrameMeta keeps the raw epoch-ns clock for correlation.
        meta = FrameMeta(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            bridge_start_ns=frame.bridge_start_ns,
            ready_ns=frame.ready_ns,
            pickup_ns=pickup_ns,
            capture_ms=dict(frame.capture_ms),
        )
        tick = PipelineTick(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            result=final_result,
            world_frame_applied=world_frame_applied,
            latency_ms={
                "e2e": (t_end - t_start) * 1e3,
                "constraint_ms": constraint_ms,
                "true_e2e_ms": (t_gpu_done_ns - frame.ts_ns) / 1e6,
                # Decomposition — each stage of true_e2e_ms
                "zed_lag_ms": zed_lag_ms,
                "bridge_proc_ms": bridge_proc_ms,
                "queue_wait_ms": queue_wait_ms,
                "pipeline_proc_ms": pipeline_proc_ms,
                **{f"{k}_ms": v for k, v in trace.stage_ms.items()},
                **frame.capture_ms,  # grab_ms, retrieve_rgb_ms, getdata_rgb_ms, etc.
            },
            meta=meta,
        )
        return tick

    def _apply_constraints_and_fallback(
        self, result: PoseResult, ts_s: float
    ) -> PoseResult:
        """Run opt-in constraints; fallback to last accepted if invalid.

        This keeps bad frames (occluded / teleported joints) from
        reaching the SHM with ``valid=True``. Constraint failures are
        converted into ``valid=False`` publishes so the control loop
        retreats to 0 N on the AK60.
        """
        if not result.valid:
            return result

        # calibration observation
        self.constraints.observe(result.kpts_3d_m)
        new_kpts, decision = self.constraints.apply(result.kpts_3d_m, ts_s=ts_s)

        if not decision.accept:
            # Hard reject — mark invalid AND zero the keypoint arrays so a
            # downstream consumer that erroneously ignores `valid` can't
            # drive AK60 using the last bad sample. Never overwrite the
            # constraint's internal prev-state.
            zeros_3d = torch.zeros_like(result.kpts_3d_m)
            zeros_2d = torch.zeros_like(result.kpts_2d_px)
            zeros_c = torch.zeros_like(result.kpt_conf)
            return PoseResult(
                kpts_2d_px=zeros_2d,
                kpts_3d_m=zeros_3d,
                kpt_conf=zeros_c,
                box_conf=result.box_conf,
                valid=False,
                depth_invalid_ratio=result.depth_invalid_ratio,
            )

        return PoseResult(
            kpts_2d_px=result.kpts_2d_px,
            kpts_3d_m=new_kpts,
            kpt_conf=result.kpt_conf,
            box_conf=result.box_conf,
            valid=True,
            depth_invalid_ratio=result.depth_invalid_ratio,
        )

    # ------------------------------------------------------------------
    # Phase 4 D1 — frame overlap helpers (Codex R3, currently UNUSED)
    # ------------------------------------------------------------------
    # 호출 path: ``_run_with_overlap()`` (Step 2.C 에서 추가) — flag
    # ``--frame-overlap`` ON 시. flag OFF 시 ``_run_sequential()`` (현재
    # ``run_overlapped_step``) 가 그대로 호출. 따라서 helper 4개 단독은
    # 영향 0 (호출자 없음).
    #
    # Cycle (Codex R3 Q7):
    #   tick = self._retire_ready(block=False)
    #   frame = self.bridge.latest(timeout=0.0 if tick else 0.5)
    #   if frame and len(_pending) < ring_size: self._submit_token(frame)
    #   return tick or self._retire_ready(block=False)
    #
    # 핵심 spec (Codex R3 정정):
    #   - _pending 은 maxlen 없는 deque
    #   - retire block path: token.post_done.synchronize() (host wait — NOT
    #     torch.cuda.current_stream().wait_event())
    #   - per-token Event: 매 frame 새로 생성 (재기록 없음, L2a 와 차이)
    #   - last_inf_done guard: pre.wait_event(self._last_inf_done) 다음
    #     pre 시작 전 — single self.pre.out 보호
    # ------------------------------------------------------------------

    def _make_token(
        self, frame: ZEDFrame, ring_idx: int
    ) -> "PipelineToken":
        """Token-owned state for Phase 4 frame overlap (Codex R3).

        Per-token Events 는 dependency only (enable_timing=False).
        Tracer 는 별도 timing event pair 보유 (FrameTrace.events).
        """
        pickup_ns = time.time_ns()
        self._frame_count += 1

        meta = FrameMeta(
            frame_id=frame.frame_id,
            ts_ns=frame.ts_ns,
            bridge_start_ns=frame.bridge_start_ns,
            ready_ns=frame.ready_ns,
            pickup_ns=pickup_ns,
            capture_ms=dict(frame.capture_ms),
        )
        # P1D2 helper — token-owned event factory.
        make_event = self.sm.bundle("preproc").make_event
        return PipelineToken(
            seq=self._frame_count,
            frame=frame,
            meta=meta,
            lb_params=None,
            output_snapshot_idx=ring_idx,
            raw_snapshot=self._output_ring[ring_idx],
            pre_done=make_event(),
            inf_done=make_event(),
            snapshot_done=make_event(),
            post_done=make_event(),
            trace=self.tracer.begin_token(
                frame_id=frame.frame_id, ts_ns=frame.ts_ns
            ),
            submit_ns=pickup_ns,
        )

    def _submit_token(self, frame: ZEDFrame) -> None:
        """Submit one frame: pre + inf + D2D snapshot + post (background).

        Stream chain:
          pre.stream:  wait(frame.ready_event) + wait(last_inf_done)
                       → preproc → record(token.pre_done)
          inf.stream:  wait(token.pre_done) → bind_input + graph.replay()
                       → record(token.inf_done)
                       → snapshot.copy_(raw_output) → record(token.snapshot_done)
          post.stream: wait(token.snapshot_done) → post(snapshot)
                       → record(token.post_done)

        ``self._last_inf_done`` 갱신 — 다음 frame 의 pre 가 wait.
        ``_pending.append(token)`` — retire 가 후속 처리.
        """
        cap = self.sm.bundle("capture")
        pre = self.sm.bundle("preproc")
        inf = self.sm.bundle("infer")
        po = self.sm.bundle("post")

        ring_idx = self._ring_idx
        token = self._make_token(frame, ring_idx)
        self._ring_idx = (self._ring_idx + 1) % self._output_ring_size

        # pre stage: input ring 사용 안 함 (single self.pre.out + last_inf_done guard)
        if frame.ready_event is not None:
            pre.wait_event(frame.ready_event)
        else:
            pre.wait_for(cap)

        if self._last_inf_done is not None:
            pre.wait_event(self._last_inf_done)

        self.tracer.mark_start_token(token.trace, "pre", pre.stream)
        _, token.lb_params = self.pre(frame.rgb_gpu, stream=pre.stream)
        self.tracer.mark_end_token(token.trace, "pre", pre.stream)
        pre.record_event(token.pre_done)

        # inf stage: graph replay → D2D snapshot copy → done
        inf.wait_event(token.pre_done)
        self.runner.bind_input_address(self._input, self.pre.out)

        if (
            self._frame_count == self._graph_warmup_frames
            and self._inf_graph is None
        ):
            self._try_capture_inf_graph(inf)

        self.tracer.mark_start_token(token.trace, "inf", inf.stream)
        with torch.cuda.stream(inf.stream):
            if self._inf_graph is not None and self._inf_graph.captured:
                self._inf_graph.replay()
            else:
                self.runner.infer_async(self.sm.stream_ptr("infer"))
        self.tracer.mark_end_token(token.trace, "inf", inf.stream)
        inf.record_event(token.inf_done)
        self._last_inf_done = token.inf_done

        # D2D copy on inf.stream — chained after graph replay
        with torch.cuda.stream(inf.stream):
            token.raw_snapshot.copy_(
                self.runner.get_output(self._output), non_blocking=True
            )
        inf.record_event(token.snapshot_done)

        # post stage: snapshot read (raw_output 직접 사용 X)
        po.wait_event(token.snapshot_done)
        self.tracer.mark_start_token(token.trace, "post", po.stream)
        token.result = self.post(
            raw_output=token.raw_snapshot,
            depth_hw=frame.depth_gpu,
            lb_params=token.lb_params,
            calibration=frame.calibration,
            stream=po.stream,
            ts_s=frame.ts_ns * 1e-9,
        )
        self.tracer.mark_end_token(token.trace, "post", po.stream)
        po.record_event(token.post_done)

        self._pending.append(token)

    def _retire_ready(self, block: bool = False) -> Optional[PipelineTick]:
        """If oldest pending token's post is ready, finalize + return tick.

        block=False (default): non-blocking ``post_done.query()`` — None if
        not ready. block=True: ``post_done.synchronize()`` (host wait —
        NOT torch.cuda.current_stream().wait_event() per Codex R3 정정).
        """
        if not self._pending:
            return None

        token = self._pending[0]
        if not token.post_done.query():
            if not block:
                return None
            token.post_done.synchronize()

        gpu_done_ns = time.time_ns()
        self._pending.popleft()
        return self._finalize_token(token, gpu_done_ns)

    def _finalize_token(
        self, token: "PipelineToken", gpu_done_ns: int
    ) -> PipelineTick:
        """Build PipelineTick from retired token.

        Constraint apply, tracer end, latency_ms compute. ablation_active 시
        constraint skip + tracer scalar marker -1 (P_post Phase 0 그대로).
        """
        if token.result is None:
            raise RuntimeError("PipelineToken retired without result")

        result = token.result
        ablation_active = self.post.lpost_ablation

        # ablation safety asserts (Codex R7 → R3 maintained)
        if ablation_active and result.valid_mask_t is None:
            raise RuntimeError(
                "lpost_ablation=True but result.valid_mask_t is None"
            )
        if not ablation_active and result.valid_mask_t is not None:
            raise RuntimeError(
                "lpost_ablation=False but result has valid_mask_t"
            )

        world_frame_applied = "R_world_from_cam" in token.frame.calibration

        if ablation_active:
            final_result = result
            constraint_ms = 0.0
            self.tracer.set_result_meta_token(
                token.trace,
                valid=False,
                occluded_count=-1,
                depth_invalid_ratio=-1.0,
                box_conf=-1.0,
            )
        else:
            t_constraint = time.perf_counter()
            final_result = self._apply_constraints_and_fallback(
                result, ts_s=token.meta.ts_ns * 1e-9
            )
            constraint_ms = (time.perf_counter() - t_constraint) * 1e3
            self.tracer.set_result_meta_token(
                token.trace,
                valid=final_result.valid,
                occluded_count=int(
                    (final_result.kpt_conf < self.post.kpt_conf_threshold).sum().item()
                )
                if final_result.valid
                else 0,
                depth_invalid_ratio=final_result.depth_invalid_ratio,
                box_conf=final_result.box_conf,
            )

        trace = self.tracer.end_token(token.trace)

        # decomposition (mock + run_overlapped_step 동일 anchor)
        zed_lag_ms = (
            (token.meta.bridge_start_ns - token.meta.ts_ns) / 1e6
            if token.meta.bridge_start_ns
            else 0.0
        )
        bridge_proc_ms = (
            (token.meta.ready_ns - token.meta.bridge_start_ns) / 1e6
            if token.meta.bridge_start_ns and token.meta.ready_ns
            else 0.0
        )
        queue_wait_ms = (
            (token.meta.pickup_ns - token.meta.ready_ns) / 1e6
            if token.meta.ready_ns
            else 0.0
        )
        pipeline_proc_ms = (gpu_done_ns - token.meta.pickup_ns) / 1e6

        token.retire_ns = gpu_done_ns
        return PipelineTick(
            frame_id=token.meta.frame_id,
            ts_ns=token.meta.ts_ns,
            result=final_result,
            world_frame_applied=world_frame_applied,
            latency_ms={
                "e2e": (gpu_done_ns - token.submit_ns) / 1e6,
                "constraint_ms": constraint_ms,
                "true_e2e_ms": (gpu_done_ns - token.meta.ts_ns) / 1e6,
                "zed_lag_ms": zed_lag_ms,
                "bridge_proc_ms": bridge_proc_ms,
                "queue_wait_ms": queue_wait_ms,
                "pipeline_proc_ms": pipeline_proc_ms,
                **{f"{k}_ms": v for k, v in trace.stage_ms.items()},
                **token.meta.capture_ms,
            },
            meta=token.meta,
            trace=trace,
        )

    def shutdown(self) -> None:
        self.sm.synchronize_all()
