"""ZED → GPU bridge.

Two modes:
  * ``mode="shared_ctx"``: attempt ``InitParameters.sdk_cuda_ctx`` sharing so
    ZED output stays in the PyTorch CUDA context. Falls back if unsupported.
  * ``mode="copy_async"``: always-safe path — retrieve on ZED's context,
    then ``cudaMemcpyAsync`` into a torch tensor on the capture stream.

Background capture runs in a thread (mirrors ``benchmarks/zed_camera.py``
``AsyncCamera`` pattern without modifying mainline). Latest frame is kept
in a ``deque(maxlen=2)`` with a lock — stale frames are dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required") from exc

try:
    import pyzed.sl as sl  # type: ignore
except ImportError:  # pragma: no cover — dev/CI path
    sl = None


RES_MAP = {
    "SVGA": "SVGA",
    "VGA": "VGA",
    "HD720": "HD720",
    "HD1080": "HD1080",
    "HD1200": "HD1200",
    "HD2K": "HD2K",
}

DEFAULT_FPS = {
    "SVGA": 120,
    "VGA": 100,
    "HD720": 60,
    "HD1080": 30,
    "HD1200": 30,
}


def _rotation_from_forward_pitch(pitch_deg: float) -> np.ndarray:
    """Rotation about camera X axis — positive pitch == camera nose down.

    When the walker camera is mounted leaning ~32° forward to see the
    subject's legs, gravity in the camera frame is
        g_cam = (0, cos(p), -sin(p))
    and R_world_from_cam must rotate that into (0, 1, 0). That rotation
    is Rx(+p):
        [ 1     0       0    ]
        [ 0   cos(p)  -sin(p)]
        [ 0   sin(p)   cos(p)]
    """
    p = float(np.deg2rad(pitch_deg))
    c, s = float(np.cos(p)), float(np.sin(p))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32
    )


def _rotation_aligning_gravity(gravity_cam: np.ndarray) -> np.ndarray:
    """Compute R such that R @ gravity_cam_hat == (0, 1, 0) (ZED world +Y down).

    Uses Rodrigues' rotation formula. When the measured gravity is
    already aligned (camera upright) returns identity. When opposite
    (camera upside-down) returns a 180° flip about X.
    Input  gravity_cam — (3,) float32, camera-frame acceleration of gravity
    Output R — (3, 3) float32
    """
    g_norm = np.linalg.norm(gravity_cam)
    if g_norm < 1e-3:
        return np.eye(3, dtype=np.float32)
    g_hat = (gravity_cam / g_norm).astype(np.float32)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.cross(g_hat, target)
    s = float(np.linalg.norm(v))
    c = float(np.dot(g_hat, target))
    if s < 1e-6:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180° — pick X axis
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    K = np.array([
        [    0.0, -v[2],  v[1]],
        [  v[2],    0.0, -v[0]],
        [ -v[1],  v[0],    0.0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + K + (K @ K) * ((1.0 - c) / (s * s))
    return R.astype(np.float32)


@dataclass
class ZEDFrame:
    """A single timestamped capture.

    ``ready_event`` is recorded on the private capture stream right after
    H2D of rgb (and depth, if enabled). Downstream consumers MUST call
    ``consumer_stream.wait_event(frame.ready_event)`` before reading
    ``rgb_gpu`` / ``depth_gpu``.

    ``calibration`` carries the intrinsics (fx/fy/cx/cy) plus, when the
    bridge was opened with IMU warmup, an ``R_world_from_cam`` torch
    tensor (3×3, float32) that rotates camera-frame 3D points into a
    gravity-aligned world frame. This mirrors mainline Method B
    (``ZEDIMUWorldFrame._R``) but keeps the IMU retrieve to the warmup
    phase only (skip_imu=True), saving ~1 ms per frame.

    Timing fields (all in same domain as ``ts_ns`` — epoch nanoseconds):
      * ``ts_ns``           — ZED hardware capture time (sensor exposure)
      * ``bridge_start_ns`` — bridge thread began processing (just after grab)
      * ``ready_ns``        — bridge finished H2D launch + put in queue
    These let pipeline decompose true_e2e_ms into bridge / queue_wait /
    pipeline portions for diagnostic visibility.
    """

    rgb_gpu: torch.Tensor  # (H, W, 3) uint8 on CUDA
    depth_gpu: Optional[torch.Tensor]  # (H, W) float32 on CUDA, meters
    ts_ns: int
    frame_id: int
    calibration: Dict[str, Any] = field(default_factory=dict)
    ready_event: Optional["torch.cuda.Event"] = None
    capture_ms: Dict[str, float] = field(default_factory=dict)  # per-step CPU timings
    bridge_start_ns: int = 0   # set by bridge in _grab_one — diagnostics
    ready_ns: int = 0          # set by bridge in _grab_one — diagnostics


class ZEDGpuBridge:
    """Background ZED capture with GPU output."""

    def __init__(
        self,
        resolution: str = "SVGA",
        fps: Optional[int] = None,
        depth_mode: str = "PERFORMANCE",
        device: Optional[torch.device] = None,
        queue_size: int = 2,
        enable_depth: bool = True,
        mode: str = "copy_async",  # "shared_ctx" | "copy_async"
        world_frame: bool = True,       # compute IMU-based R at warmup
        imu_warmup_frames: int = 20,    # gravity vector average window
        manual_pitch_deg: Optional[float] = None,  # override IMU with pitch angle
        collect_cycle_stats: bool = False,  # A11 (2026-05-06) — opt-in cycle history
    ) -> None:
        self.device = device or torch.device("cuda:0")
        self.resolution = resolution
        self.fps = fps or DEFAULT_FPS.get(resolution, 30)
        self.depth_mode = depth_mode
        self.enable_depth = enable_depth
        self.mode = mode
        self.world_frame = world_frame
        self.imu_warmup_frames = imu_warmup_frames
        self.manual_pitch_deg = manual_pitch_deg

        self._frames: Deque[ZEDFrame] = deque(maxlen=queue_size)
        self._frames_lock = threading.Lock()
        self._stop_event = threading.Event()

        # A11 (2026-05-06) — opt-in cycle stats collection. When enabled, every
        # _grab_one() appends a record. Allows post-hoc analysis of bridge cycle
        # *in our code path* without consuming frames (Pipeline-thread-free).
        self._collect_cycle_stats = collect_cycle_stats
        self._cycle_stats: list = []

        # L2a (2026-05-06) — event pool. Per-frame torch.cuda.Event() allocation
        # was a confirmed contention source (Codex R5/R7-Q5: enable_timing=True +
        # alloc churn affects p99 jitter). Pool size = queue_size + 1 = 3
        # matches pinned/GPU buffer ownership. Codex Q3 verified case-1 reuse
        # safety: pipeline always calls wait_event() *before* bridge re-records
        # the same slot, so the host-enqueued wait captures the old record per
        # cudaStreamWaitEvent semantics. enable_timing=False — events are
        # dependency markers, not timing sources (no elapsed_time() callers
        # found via grep). Allocated lazily on first _upload() call.
        self._event_pool: list = []
        self._event_pool_idx = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_id = 0
        # consume-once tracking: latest() returns each frame_id exactly once.
        # Without this, a stalled bridge causes pipeline to reprocess the same
        # frame, inflating true_e2e_ms with stale-reuse latency (codex finding,
        # 2026-05-04). 0 sentinel matches frame_id == 1 first-frame condition.
        self._last_returned_id = 0
        self._zed: Optional[Any] = None
        self._calibration: Dict[str, float] = {}
        self._using_webcam = False
        # Private CUDA stream for H2D copies. Kept isolated from the
        # pipeline's streams to avoid polluting the default stream.
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        # Pre-allocated pinned host buffer ring (one per slot in the
        # frame deque, +1 for the in-flight buffer the capture thread
        # is writing). Re-using these buffers eliminates the per-frame
        # pin_memory() cost (0.5-2ms variance — biggest spike source).
        self._pool_size = queue_size + 1
        self._rgb_pool: list[torch.Tensor] = []
        self._depth_pool: list[torch.Tensor] = []
        self._pool_idx = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        if sl is None:
            LOGGER.warning("pyzed.sl not available — falling back to webcam stub")
            self._open_webcam_fallback()
            return
        # skiro-learnings: NEURAL depth doubles predict latency under
        # concurrent YOLO — never used in production for this pipeline.
        if self.depth_mode.upper() == "NEURAL":
            raise ValueError(
                "NEURAL depth mode is disabled — causes 2.4× predict spike "
                "under YOLO contention (skiro-learnings). Use PERFORMANCE."
            )
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, RES_MAP[self.resolution])
        init.camera_fps = self.fps
        init.coordinate_units = sl.UNIT.METER
        if self.enable_depth:
            init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode)
            init.depth_minimum_distance = 0.1
        else:
            init.depth_mode = sl.DEPTH_MODE.NONE

        if self.mode not in ("copy_async",):
            # sdk_cuda_ctx sharing would need a CUcontext pointer that
            # PyTorch doesn't expose. We keep the option open for future
            # ZED SDK + cuda-python integration but error loudly so
            # callers don't silently get the wrong path.
            raise ValueError(
                f"unsupported ZED mode={self.mode!r}; only 'copy_async' is "
                "implemented (see zed-python-api issue #35)"
            )

        self._zed = sl.Camera()
        status = self._zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        cam_info = self._zed.get_camera_information().camera_configuration
        fx = cam_info.calibration_parameters.left_cam.fx
        fy = cam_info.calibration_parameters.left_cam.fy
        cx = cam_info.calibration_parameters.left_cam.cx
        cy = cam_info.calibration_parameters.left_cam.cy
        self._calibration = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        # reusable host buffers
        self._image_mat = sl.Mat()
        self._depth_mat = sl.Mat()

        # Build the static rotation R that maps camera-frame 3D points
        # into a gravity-aligned world frame (world +Y == down). Three
        # sources, tried in order:
        #   1. ``manual_pitch_deg`` override (most reliable — just trust the user)
        #   2. IMU warmup (mainline Method B parity)
        #   3. disabled (camera frame kept as-is)
        R: Optional[np.ndarray] = None
        if self.world_frame:
            if self.manual_pitch_deg is not None:
                R = _rotation_from_forward_pitch(self.manual_pitch_deg)
                LOGGER.info(
                    "manual pitch override: %.1f° → R_world_from_cam =\n%s",
                    self.manual_pitch_deg,
                    np.array2string(R, precision=3, suppress_small=True),
                )
            else:
                R = self._compute_world_rotation_from_imu()
                if R is not None:
                    LOGGER.info(
                        "IMU warmup: R_world_from_cam =\n%s",
                        np.array2string(R, precision=3, suppress_small=True),
                    )
        if R is not None:
            R_gpu = torch.from_numpy(R).to(self.device).contiguous()
            self._calibration["R_world_from_cam"] = R_gpu
        elif self.world_frame:
            LOGGER.warning(
                "No world rotation available — sagittal view will be in camera frame. "
                "Pass --camera-pitch-deg <angle> or investigate IMU warmup.",
            )

        LOGGER.info(
            "ZED opened %s@%dHz mode=%s world_frame=%s",
            self.resolution, self.fps, self.mode,
            "R_world_from_cam" in self._calibration,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_R(q: np.ndarray) -> np.ndarray:
        """quaternion [x, y, z, w] → 3×3 rotation matrix (ZED SDK convention).

        Identical to mainline ``calibration.ZEDIMUWorldFrame._quat_to_R``.
        """
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    def _compute_world_rotation_from_imu(self) -> Optional[np.ndarray]:
        """Average IMU orientation QUATERNION over N frames → R_world_from_cam.

        **Why quaternion, not get_linear_acceleration():**
        ZED SDK 5.x returns ``get_linear_acceleration()`` as **gravity-
        compensated** (norm ≈ 0 at rest). The accelerometer-based approach
        therefore fails the ``> 5.0 m/s²`` gravity-sanity filter and
        collects 0 samples → warmup returns None.

        ZED's internal sensor fusion already gives us the camera's
        absolute orientation as a quaternion (``get_pose().get_orientation()``).
        Averaging quaternions over N frames and converting to a rotation
        matrix is the mainline approach (``calibration.ZEDIMUWorldFrame``)
        and works regardless of SDK version / accelerometer mode.

        Returns ``R_world_from_cam`` such that
        ``R @ p_cam`` gives the point in a gravity-aligned world frame
        (world +Y == down, matching mainline Method B convention).
        """
        if sl is None or self._zed is None:
            return None
        sensors = sl.SensorsData()
        quats: list[np.ndarray] = []
        rt = sl.RuntimeParameters()
        tmp_mat = sl.Mat()
        n_tried = 0
        n_target = max(int(self.imu_warmup_frames), 5)
        while len(quats) < n_target and n_tried < n_target * 4:
            n_tried += 1
            if self._zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue
            # drain image so subsequent grab doesn't stall on the queue
            self._zed.retrieve_image(tmp_mat, sl.VIEW.LEFT)
            if self._zed.get_sensors_data(
                sensors, sl.TIME_REFERENCE.IMAGE
            ) != sl.ERROR_CODE.SUCCESS:
                continue
            imu = sensors.get_imu_data()
            # ZED fused orientation quaternion — [ox, oy, oz, ow]
            o = imu.get_pose().get_orientation().get()
            q = np.array([o[0], o[1], o[2], o[3]], dtype=np.float32)
            if not np.all(np.isfinite(q)):
                continue
            norm = float(np.linalg.norm(q))
            if norm < 0.5:   # unit quaternion should have norm ≈ 1
                continue
            q = q / norm     # normalize per-sample
            quats.append(q)

        if len(quats) < 5:
            LOGGER.warning(
                "IMU warmup: only %d/%d quaternions collected — world frame disabled. "
                "Check IMU availability or pass --camera-pitch-deg for fallback.",
                len(quats), n_target,
            )
            return None

        # Simple mean (valid for small angular differences during warmup
        # where camera is static). Final quaternion re-normalized.
        q_mean = np.mean(np.stack(quats, axis=0), axis=0)
        q_mean = q_mean / np.linalg.norm(q_mean)
        R = self._quat_to_R(q_mean)

        LOGGER.info(
            "IMU warmup (quaternion, N=%d): q_mean=[%.3f, %.3f, %.3f, %.3f]",
            len(quats),
            q_mean[0], q_mean[1], q_mean[2], q_mean[3],
        )
        return R

    def _open_webcam_fallback(self) -> None:
        import cv2

        self._webcam = cv2.VideoCapture(0)
        if not self._webcam.isOpened():
            raise RuntimeError("No ZED and webcam fallback also failed")
        self._using_webcam = True
        self._calibration = {"fx": 600, "fy": 600, "cx": 320, "cy": 240}
        LOGGER.warning("Using webcam fallback (no depth, calibration is stub)")

    def start(self) -> None:
        if self._capture_thread is not None:
            return
        # Allocate private H2D stream on the consumer thread first — the
        # capture thread uses torch.cuda.stream(...) by reference.
        if torch.cuda.is_available() and self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=self.device)
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="ZEDCaptureLoop", daemon=True
        )
        self._capture_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if self._zed is not None:
            self._zed.close()
            self._zed = None
        if self._using_webcam and hasattr(self, "_webcam"):
            self._webcam.release()

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._grab_one()
                if frame is not None:
                    with self._frames_lock:
                        self._frames.append(frame)
            except Exception as err:  # pragma: no cover — keep thread alive
                LOGGER.error("capture loop error: %s", err)
                time.sleep(0.01)

    def _grab_one(self) -> Optional[ZEDFrame]:
        if self._using_webcam:
            return self._grab_webcam()
        assert self._zed is not None and sl is not None
        rt = sl.RuntimeParameters()
        cap = {}

        t0 = time.perf_counter()
        if self._zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
            return None
        cap["grab_ms"] = (time.perf_counter() - t0) * 1e3

        # ts_ns and bridge_start_ns are both epoch-ns (same domain as time.time_ns()).
        # ts_ns = sensor exposure time (ZED hardware), bridge_start_ns = right after
        # grab() returned. Their difference is ZED SDK's internal latency from
        # exposure to grab completion (typically 1-3 ms on Orin NX).
        ts_ns = int(
            self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        )
        bridge_start_ns = time.time_ns()

        t0 = time.perf_counter()
        self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        cap["retrieve_rgb_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        # IMPORTANT: skiro-learnings — always copy=True to avoid race with
        # next grab(); the copy cost at SVGA is ~0.5ms.
        # L1 (2026-05-06): keep BGRA 4-channel as-is. The previous
        # `np.ascontiguousarray(bgra[:,:,:3][:,:,::-1])` was costing ~4ms
        # (A11 measurement 2026-05-06: getdata_rgb 0.36ms in raw bench vs
        # 4.76ms here). GPU preproc now handles BGR→RGB channel select +
        # alpha drop on the GPU side at sub-microsecond cost.
        bgra_host = self._image_mat.get_data(deep_copy=True)
        cap["getdata_rgb_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        rgb_pinned = self._get_pinned_rgb(bgra_host)   # BGRA 4ch (was RGB 3ch pre-L1)
        cap["pinned_rgb_ms"] = (time.perf_counter() - t0) * 1e3

        depth_pinned = None
        if self.enable_depth:
            t0 = time.perf_counter()
            self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
            cap["retrieve_depth_ms"] = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            depth_host = self._depth_mat.get_data(deep_copy=True)
            depth_pinned = self._get_pinned_depth(depth_host)
            cap["getdata_depth_ms"] = (time.perf_counter() - t0) * 1e3

        rgb_gpu, depth_gpu, ready_event = self._upload(rgb_pinned, depth_pinned)

        # ready_ns = right after H2D was LAUNCHED (cudaMemcpyAsync queued + event
        # recorded). The actual GPU completion is signalled by ready_event; CPU
        # timestamp here is what's available without sync-blocking the bridge.
        ready_ns = time.time_ns()

        self._frame_id += 1

        # A11 — opt-in cycle stats (post-hoc analysis without consuming frame).
        if self._collect_cycle_stats:
            self._cycle_stats.append({
                "frame_id": self._frame_id,
                "ts_ns": ts_ns,
                "grab_ms": cap.get("grab_ms", 0.0),
                "retrieve_rgb_ms": cap.get("retrieve_rgb_ms", 0.0),
                "getdata_rgb_ms": cap.get("getdata_rgb_ms", 0.0),
                "pinned_rgb_ms": cap.get("pinned_rgb_ms", 0.0),
                "retrieve_depth_ms": cap.get("retrieve_depth_ms", 0.0),
                "getdata_depth_ms": cap.get("getdata_depth_ms", 0.0),
                "bridge_proc_ms": (ready_ns - bridge_start_ns) / 1e6,
            })

        return ZEDFrame(
            rgb_gpu=rgb_gpu,
            depth_gpu=depth_gpu,
            ts_ns=ts_ns,
            frame_id=self._frame_id,
            calibration=dict(self._calibration),  # snapshot copy — not a reference
            ready_event=ready_event,
            capture_ms=cap,
            bridge_start_ns=bridge_start_ns,
            ready_ns=ready_ns,
        )

    # ------------------------------------------------------------------
    # Pinned buffer pool — avoids per-frame pin_memory() spike
    # ------------------------------------------------------------------
    def _get_pinned_rgb(self, host: np.ndarray) -> torch.Tensor:
        """Return a pinned tensor with ``host`` copied into it.

        Lazily allocates the pool on first call (we need to know the
        actual shape first). After that we rotate through the pool and
        memcpy into the existing pinned buffer — no allocation in the
        hot path.
        """
        if not torch.cuda.is_available():
            return torch.from_numpy(host)
        if not self._rgb_pool or self._rgb_pool[0].shape != host.shape:
            self._rgb_pool = [
                torch.empty(host.shape, dtype=torch.uint8, pin_memory=True)
                for _ in range(self._pool_size)
            ]
            self._pool_idx = 0
        slot = self._pool_idx % self._pool_size
        buf = self._rgb_pool[slot]
        # source.copy_() is an in-place memcpy; cheap and deterministic.
        buf.copy_(torch.from_numpy(host))
        return buf

    def _get_pinned_depth(self, host: np.ndarray) -> torch.Tensor:
        if not torch.cuda.is_available():
            return torch.from_numpy(host)
        if not self._depth_pool or self._depth_pool[0].shape != host.shape:
            self._depth_pool = [
                torch.empty(host.shape, dtype=torch.float32, pin_memory=True)
                for _ in range(self._pool_size)
            ]
        slot = self._pool_idx % self._pool_size
        self._pool_idx += 1  # advance once per frame (rgb was slot N, depth reuses N)
        buf = self._depth_pool[slot]
        buf.copy_(torch.from_numpy(host))
        return buf

    def _upload(
        self,
        rgb_pinned: torch.Tensor,
        depth_pinned: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional["torch.cuda.Event"]]:
        """Run H2D on the private capture stream and record a ready event."""
        if self._h2d_stream is None:
            # e.g. CUDA unavailable → eager path (dev machines)
            rgb_gpu = rgb_pinned.to(self.device, non_blocking=True)
            depth_gpu = (
                depth_pinned.to(self.device, non_blocking=True)
                if depth_pinned is not None
                else None
            )
            return rgb_gpu, depth_gpu, None

        # L2a (2026-05-06) — event pool reuse instead of per-frame allocation.
        # Lazily allocated on first call (saves init cost when CUDA is the
        # fallback path). enable_timing=False per Codex Q3 (no elapsed_time
        # callers; timing variant is heavier).
        if not self._event_pool:
            self._event_pool = [
                torch.cuda.Event(enable_timing=False, blocking=False)
                for _ in range(self._pool_size)
            ]
            self._event_pool_idx = 0
        slot = self._event_pool_idx % self._pool_size
        self._event_pool_idx += 1
        event = self._event_pool[slot]

        with torch.cuda.stream(self._h2d_stream):
            rgb_gpu = rgb_pinned.to(self.device, non_blocking=True)
            depth_gpu = (
                depth_pinned.to(self.device, non_blocking=True)
                if depth_pinned is not None
                else None
            )
            event.record(self._h2d_stream)
        return rgb_gpu, depth_gpu, event

    def _grab_webcam(self) -> Optional[ZEDFrame]:
        ok, bgr = self._webcam.read()
        if not ok:
            return None
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        rgb_pinned = self._get_pinned_rgb(rgb)
        rgb_gpu, _, ready_event = self._upload(rgb_pinned, None)
        self._frame_id += 1
        return ZEDFrame(
            rgb_gpu=rgb_gpu,
            depth_gpu=None,
            ts_ns=time.time_ns(),
            frame_id=self._frame_id,
            calibration=self._calibration,
            ready_event=ready_event,
        )

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------
    def latest(self, timeout: float = 1.0) -> Optional[ZEDFrame]:
        """Return the latest UNCONSUMED frame, or None if no new frame arrives in timeout.

        Each frame is returned at most once. Consecutive calls return None
        until the bridge produces a new frame. This prevents the pipeline
        from reprocessing the same frame, which had been silently inflating
        true_e2e_ms with stale-reuse latency.

        With deque(maxlen=2), the bridge may evict frames between pickups
        (intentional — we always serve the freshest available).
        """
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._frames_lock:
                if self._frames:
                    frame = self._frames[-1]
                    if frame.frame_id != self._last_returned_id:
                        self._last_returned_id = frame.frame_id
                        return frame
            time.sleep(0.001)
        return None

    @property
    def calibration(self) -> Dict[str, float]:
        return dict(self._calibration)

    def get_cycle_stats(self) -> list:
        """Return collected cycle stats (A11). Empty list if collect_cycle_stats=False."""
        return list(self._cycle_stats)
