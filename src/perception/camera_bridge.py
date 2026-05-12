"""Camera-agnostic stereo capture interface — 논문 의 generalization prerequisite.

사용자 의지 (논문 + 일반화):
"이 카메라에서만 되는 기술이 아니라 다른 카메라에서도 될 수 있는 new tech".

이 file = abstract CameraBridge interface. concrete implementations:
    - ZEDBridge   (ZED SDK 5.x, GMSL2)
    - V4L2Bridge  (V4L2 raw bayer + VPI pipeline + sparse stereo)
    - (future) RealSenseBridge, BaslerBridge, OakDBridge, ...

진정 contribution (논문):
    - Camera-agnostic 인터페이스가 동일 Plan D EKF 와 통합
    - Stereo + global shutter + 60Hz+ 의무, 단 sensor 선택 자유
    - V4L2 bypass path 시 sparse stereo (custom CUDA) + raw distortion
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraCalibration:
    """Generic stereo camera calibration (any vendor).

    Fields are *rectified-or-raw* 의 자유 — caller 가 select.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_mm: float
    distortion: list           # 5 (pinhole) or 12 (Brown-Conrady extended) coeffs
    width: int
    height: int
    is_rectified: bool         # True = rectified (disto=zeros), False = raw
    vendor: str                # 'ZED', 'V4L2', 'RealSense', 'Basler', etc.
    sensor_id: str             # serial number 또는 identifier
    sdk_version: str = ""

    def fxfy(self) -> Tuple[float, float]:
        return self.fx, self.fy

    def principal(self) -> Tuple[float, float]:
        return self.cx, self.cy

    def baseline_m(self) -> float:
        return self.baseline_mm / 1000.0


@dataclass
class StereoCaptureFrame:
    """Per-frame stereo capture output (camera-agnostic).

    All fields are *raw or rectified* — caller 가 calib.is_rectified 의 확인 의무.
    """
    rgb_bgra: np.ndarray       # (H, W, 4) uint8 — left RGB (or rectified)
    depth_m: Optional[np.ndarray]      # (H, W) float32 (m) — None if not computed
    rgb_right_bgra: Optional[np.ndarray] = None    # (H, W, 4) uint8 — right (V4L2 path)
    rgb_ts_ns: int = 0          # left capture timestamp (CLOCK_REALTIME)
    depth_ts_ns: int = 0        # depth retrieve timestamp (= rgb_ts if same-frame)
    frame_id: int = 0
    depth_invalid_ratio: float = 0.0


class CameraBridge(ABC):
    """Abstract stereo camera + depth source.

    Implementations 의무:
        - start() : open + warmup
        - capture() : single frame (left + depth, optional right)
        - get_calibration() : intrinsics + extrinsics
        - stop() : graceful close

    Optional:
        - capture_raw() : bayer/NV12 raw (V4L2 path 의무, ZED SDK 는 NotImplementedError)
    """

    @abstractmethod
    def start(self) -> None:
        """Open device + initial configuration + warmup."""
        ...

    @abstractmethod
    def capture(self) -> Optional[StereoCaptureFrame]:
        """Single frame. None on transient failure (caller retry)."""
        ...

    @abstractmethod
    def get_calibration(self) -> CameraCalibration:
        """Calibration (intrinsics + baseline + distortion)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Graceful close."""
        ...

    def capture_raw(self) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Optional — V4L2 raw bayer (or NV12) capture.

        Returns:
            (left_raw, right_raw, ts_ns) — raw sensor output.
            None if not supported.
        """
        return None

    def __enter__(self) -> "CameraBridge":
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ─── Stub implementations (for typing / interface tests) ─────────────────

class StubCameraBridge(CameraBridge):
    """Test-only stub — generates synthetic frames (Mac executable)."""

    def __init__(self, width: int = 960, height: int = 600, K: int = 6):
        self.width = width
        self.height = height
        self.K = K
        self._frame_id = 0
        self._rng = np.random.default_rng(42)

    def start(self) -> None:
        self._frame_id = 0

    def capture(self) -> Optional[StereoCaptureFrame]:
        import time
        H, W = self.height, self.width
        # Natural-like RGB (gradient + noise)
        rgb_bgra = np.zeros((H, W, 4), dtype=np.uint8)
        for y in range(H):
            rgb_bgra[y, :, 0] = (y * 200) // H
            rgb_bgra[y, :, 1] = ((H - y) * 200) // H
        rgb_bgra[:, :, 3] = 255
        # Smooth depth
        depth_m = np.full((H, W), 1.5, dtype=np.float32)
        depth_m += self._rng.random((H, W)).astype(np.float32) * 0.5

        ts_ns = time.time_ns()
        frame = StereoCaptureFrame(
            rgb_bgra=rgb_bgra,
            depth_m=depth_m,
            rgb_ts_ns=ts_ns,
            depth_ts_ns=ts_ns,
            frame_id=self._frame_id,
            depth_invalid_ratio=0.0,
        )
        self._frame_id += 1
        return frame

    def get_calibration(self) -> CameraCalibration:
        return CameraCalibration(
            fx=480.0, fy=480.0,
            cx=self.width / 2, cy=self.height / 2,
            baseline_mm=63.0,
            distortion=[0.0] * 12,
            width=self.width, height=self.height,
            is_rectified=True,
            vendor="Stub",
            sensor_id="stub-0001",
            sdk_version="0.0",
        )

    def stop(self) -> None:
        pass


# ─── Smoke test ──────────────────────────────────────────────────────────

def _smoke_test() -> int:
    """Mac executable — abstract interface 의 contract 검증."""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    LOG = logging.getLogger(__name__)

    bridge = StubCameraBridge()
    bridge.start()
    try:
        for i in range(3):
            frame = bridge.capture()
            assert frame is not None
            assert frame.rgb_bgra.shape == (600, 960, 4)
            assert frame.depth_m.shape == (600, 960)
            assert frame.depth_m.dtype == np.float32
            assert frame.rgb_ts_ns > 0
            LOG.info(f"frame {i}: rgb {frame.rgb_bgra.shape}, depth {frame.depth_m.shape}")
        calib = bridge.get_calibration()
        assert calib.fx == 480.0
        assert calib.baseline_m() == 0.063
        LOG.info(f"calib: vendor={calib.vendor}, baseline={calib.baseline_mm}mm, "
                 f"rectified={calib.is_rectified}")
    finally:
        bridge.stop()

    LOG.info("✓ CameraBridge abstract interface smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
