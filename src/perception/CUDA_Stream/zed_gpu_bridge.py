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
        # Plan v7 (2026-05-07) — zed_lag 21ms 진단 levers
        exposure_us: Optional[int] = None,  # None=AUTO, 양수=MANUAL microseconds
        sensing_mode: str = "STANDARD",     # STANDARD | FILL
        diag_zed_lag: bool = False,         # warmup 5 frames timestamp 다층 print
        # γ Phase (2026-05-08) — ZED CUDA interop ablation flag.
        # OFF (default): copy_async path (현재 H2D + pinned pool)
        # ON: shared_ctx path (ZED MEM::GPU + DLPack zero-copy)
        # Codex R10 verified: ZED ctx == PyTorch ctx → DLPack 가능.
        zed_cuda_interop: bool = False,
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
        self.exposure_us = exposure_us
        self.sensing_mode = sensing_mode.upper()
        self.diag_zed_lag = diag_zed_lag
        # γ Phase — ZED CUDA interop flag. ON 시 self.mode 강제 'shared_ctx'.
        # 단 Codex R5 응답 후 _grab_one + _upload 의 shared_ctx path 구현 완료.
        # 그 전에 flag ON 호출 시 NotImplementedError raise (silent broken 방지).
        self.zed_cuda_interop = zed_cuda_interop
        if self.zed_cuda_interop:
            self.mode = "shared_ctx"

        self._frames: Deque[ZEDFrame] = deque(maxlen=queue_size)
        self._frames_lock = threading.Lock()
        self._stop_event = threading.Event()

        # A11 (2026-05-06) — opt-in cycle stats collection. When enabled, every
        # _grab_one() appends a record. Allows post-hoc analysis of bridge cycle
        # *in our code path* without consuming frames (Pipeline-thread-free).
        self._collect_cycle_stats = collect_cycle_stats
        self._cycle_stats: list = []

        # L2a REVERTED (2026-05-06) — event pool was net-negative in Full pipeline.
        # Measurement: 3 runs mean true_e2e p99 +5.9ms vs L1-only, bridge_proc
        # p99 +2.3ms, queue_wait p99 +6.4ms. Hypothesis: cudaStreamWaitEvent
        # case-2 over-wait (Codex Q2) — when bridge re-records the slot before
        # pipeline enqueues wait, the host-enqueued wait latches onto a *future*
        # record. Codex deemed this "safe" but did not predict the latency cost.
        # Empirical result: net negative. Reverting to per-frame Event allocation.
        # Lesson: do NOT pool CUDA events when consumer wait timing is not
        # deterministic relative to producer record timing.
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

        if self.mode == "shared_ctx":
            # γ Phase (2026-05-08, Codex R10 verified ZED ctx == PyTorch ctx).
            # shared_ctx path 는 _grab_one + _upload 의 ZED MEM::GPU + DLPack
            # zero-copy 구현 필요. Codex R5 응답 후 γ.C-E 통합 commit 시 활성.
            # 현재 STUB — flag --zed-cuda-interop ON 호출 시 NotImplementedError.
            raise NotImplementedError(
                "γ ZED CUDA interop (shared_ctx path) STUB. "
                "Codex R5 응답 후 _grab_one + _upload 의 MEM::GPU + DLPack "
                "implementation 완료 시 활성. 현재는 --zed-cuda-interop OFF "
                "(default) 만 사용 가능."
            )
        if self.mode not in ("copy_async",):
            raise ValueError(
                f"unsupported ZED mode={self.mode!r}; only 'copy_async' / "
                "'shared_ctx' (γ STUB) supported."
            )

        self._zed = sl.Camera()
        status = self._zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        # Plan v7 — exposure mode override
        # 검증된 사실 (Stereolabs Camera Controls doc, 2026-05-06):
        #   ZED 2/2i/Mini (구형): VIDEO_SETTINGS.EXPOSURE = 0..100 percentage of framerate (-1 = AUTO)
        #   ZED X / X Mini (GMSL): VIDEO_SETTINGS.EXPOSURE_TIME = microseconds (1024..66000 default range)
        # 우리는 ZED X Mini → EXPOSURE_TIME path 가 정확. 단 SDK 버전에 따라
        # enum 미존재 가능성 있어 hasattr 으로 안전하게 시도.
        if self.exposure_us is not None:
            exposure_time_enum = getattr(sl.VIDEO_SETTINGS, "EXPOSURE_TIME", None)
            applied_path = None
            if exposure_time_enum is not None:
                try:
                    self._zed.set_camera_settings(exposure_time_enum, self.exposure_us)
                    applied_path = f"EXPOSURE_TIME={self.exposure_us}us (ZED X family)"
                except Exception as err:
                    LOGGER.warning("EXPOSURE_TIME set failed: %s — fall back to percentage", err)
            if applied_path is None:
                # Fallback for ZED 2/Mini: EXPOSURE (0-100 percentage of frame interval)
                frame_us = int(1_000_000 / self.fps)
                pct = max(0, min(100, int(100 * self.exposure_us / frame_us)))
                try:
                    self._zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, pct)
                    applied_path = f"EXPOSURE={pct}% (ZED 2/Mini, ≈{self.exposure_us}us / frame={frame_us}us)"
                except Exception as err:
                    LOGGER.error("ZED exposure set failed BOTH APIs: %s", err)
                    applied_path = f"FAILED ({err})"
            LOGGER.info("ZED exposure → MANUAL via %s", applied_path)
        else:
            LOGGER.info("ZED exposure → AUTO (default)")

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
        # Plan v7 — sensing_mode 적용.
        # 검증된 사실 (4.0 Migration Guide + RuntimeParameters API doc):
        #   SDK 4.0+: RuntimeParameters.enable_fill_mode (bool, default False)
        #   SDK 3.x : RuntimeParameters.sensing_mode = SENSING_MODE.{STANDARD,FILL}
        #   SDK 4.0 에서 SENSING_MODE enum 자체 제거됨.
        # → enable_fill_mode 먼저 시도 (4.0+ 정확), 실패 시 SDK 3.x fallback.
        if not getattr(self, "_sensing_path_logged", False):
            self._sensing_path_logged = True
            self._sensing_path = "unset"
        try:
            rt.enable_fill_mode = (self.sensing_mode == "FILL")
            if self._sensing_path == "unset":
                self._sensing_path = f"enable_fill_mode={rt.enable_fill_mode} (SDK 4.0+)"
                LOGGER.info("ZED sensing → %s", self._sensing_path)
        except AttributeError:
            try:
                sensing_enum = getattr(sl, "SENSING_MODE", None)
                if sensing_enum is not None:
                    rt.sensing_mode = getattr(sensing_enum, self.sensing_mode)
                    if self._sensing_path == "unset":
                        self._sensing_path = f"SENSING_MODE.{self.sensing_mode} (SDK 3.x)"
                        LOGGER.info("ZED sensing → %s", self._sensing_path)
            except Exception as err:
                if self._sensing_path == "unset":
                    self._sensing_path = f"FAILED ({err})"
                    LOGGER.warning("ZED sensing path failed: %s", err)
        cap = {}

        t0 = time.perf_counter()
        if self._zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
            return None
        cap["grab_ms"] = (time.perf_counter() - t0) * 1e3

        # ts_ns and bridge_start_ns are both epoch-ns (same domain as time.time_ns()).
        # ts_ns = sensor exposure time (ZED hardware), bridge_start_ns = right after
        # grab() returned. Their difference is ZED SDK's internal latency from
        # exposure to grab completion (typically 1-3 ms on Orin NX per SDK doc;
        # ⚠️  우리 측정 21ms — Plan v7 진단 대상).
        ts_ns = int(
            self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        )
        bridge_start_ns = time.time_ns()

        # Plan v7 — Round 0 진단 (warmup 5 frames). 다층 timestamp 비교로
        # zed_lag 21ms 의 의미 격리.
        # 검증된 정의 (Stereolabs forum, ZED X timestamps thread):
        #   TIME_REFERENCE.IMAGE   = "time the image is received at the host
        #                            (~us = exposure end)"
        #   TIME_REFERENCE.CURRENT = "current SDK call time"
        # 즉 (current - image) = SDK 가 이 grab() 처리에 *지금까지* 사용한 시간.
        #     (bridge - image) = exposure end → time.time_ns() 까지 호스트 시간.
        # 두 값이 같은 epoch (UNIX ns) 라면 일관. 다르면 sanity warning.
        if self.diag_zed_lag and self._frame_id < 5:
            try:
                ts_current_ns = int(
                    self._zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()
                )
                image_to_current_ms = (ts_current_ns - ts_ns) / 1e6
                image_to_bridge_ms = (bridge_start_ns - ts_ns) / 1e6
                current_to_bridge_ms = (bridge_start_ns - ts_current_ns) / 1e6

                # Epoch sanity — 두 timestamp 가 같은 UNIX epoch 인가
                epoch_warn = ""
                if abs(image_to_bridge_ms) > 10_000 or image_to_bridge_ms < 0:
                    epoch_warn = " ⚠️ EPOCH MISMATCH (image_to_bridge 비현실적)"

                LOGGER.info(
                    "[zed_ts] frame=%d image=%d current=%d bridge=%d  "
                    "image_to_current=%.2fms image_to_bridge=%.2fms "
                    "current_to_bridge=%.2fms grab_ms=%.2f%s",
                    self._frame_id,
                    ts_ns, ts_current_ns, bridge_start_ns,
                    image_to_current_ms, image_to_bridge_ms,
                    current_to_bridge_ms, cap["grab_ms"], epoch_warn,
                )
            except Exception as err:
                LOGGER.warning("[zed_ts] diag failed: %s", err)

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
            # CUDA stream not available — eager path. Note: self.device defaults
            # to cuda:0; if the host has no CUDA at all the .to(self.device)
            # call below raises. CPU-only dev environments must pass
            # device=torch.device("cpu") explicitly to ZEDGpuBridge.
            rgb_gpu = rgb_pinned.to(self.device, non_blocking=True)
            depth_gpu = (
                depth_pinned.to(self.device, non_blocking=True)
                if depth_pinned is not None
                else None
            )
            return rgb_gpu, depth_gpu, None

        # L2a REVERTED — back to per-frame allocation (see __init__ comment).
        with torch.cuda.stream(self._h2d_stream):
            rgb_gpu = rgb_pinned.to(self.device, non_blocking=True)
            depth_gpu = (
                depth_pinned.to(self.device, non_blocking=True)
                if depth_pinned is not None
                else None
            )
            event = torch.cuda.Event(enable_timing=True, blocking=False)
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
