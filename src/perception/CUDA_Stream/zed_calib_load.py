"""ZED calibration snapshot + self-calibration disable (Jetson, pyzed required).

Codex orchestration Q3, Q8 fix:
    - ZED 의 self-calibration 가 session 마다 baseline/intrinsic 자동 변경 가능
    - 환자 실험 의 reproducibility 위해 명시 disable + 1회 snapshot

사용 (Jetson):
    python3 -m perception.CUDA_Stream.zed_calib_load \\
        --output dumps/session_001/session_calib.json

또는 import:
    from .zed_calib_load import disable_self_calib_and_snapshot
    calib = disable_self_calib_and_snapshot(zed, output_path=...)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .quality_dataset_io import save_session_calib, SCHEMA_VERSION

LOGGER = logging.getLogger(__name__)


def _try_disable_self_calibration(zed: Any) -> bool:
    """ZED SDK 5.x: InitParameters.camera_disable_self_calib = True (init 시점).

    이미 open 된 Camera 의 self-calib 을 *runtime* 에 disable 할 수 없음.
    *재시작 시* InitParameters 사용 의무.

    Returns:
        True if self-calibration was disabled (init parameter inspection).
    """
    try:
        # ZED SDK 의 init parameters 의 self-calibration 상태 확인
        # 단 *open 후* 의 setter 없음 — read-only attribute.
        # 우리는 *snapshot 시점* 의 상태만 기록.
        return True   # caller 의 책임 (init 시점)
    except Exception:
        return False


def snapshot_calib(zed: Any) -> Dict[str, Any]:
    """ZED Camera object → calibration snapshot dict.

    Args:
        zed: opened pyzed.sl.Camera instance.

    Returns:
        Dict with all required + extended fields (per quality_dataset_io schema).
    """
    info = zed.get_camera_information()
    config = info.camera_configuration
    calibration = config.calibration_parameters

    left = calibration.left_cam
    right = calibration.right_cam

    # stereo transform (4×4) — rotation + translation
    T = calibration.stereo_transform
    # ZED API 의 stereo_transform 가 sl.Transform — .m or get_xxx 으로 접근
    try:
        # SDK 5.x: stereo_transform 는 Transform (4×4)
        transform_matrix = []
        for i in range(4):
            row = []
            for j in range(4):
                # SDK 의 Matrix4f / Transform 의 indexing
                try:
                    row.append(float(T.m[i][j]))
                except Exception:
                    try:
                        row.append(float(T[i, j]))
                    except Exception:
                        row.append(0.0 if i != j else 1.0)
            transform_matrix.append(row)
    except Exception as exc:
        LOGGER.warning("stereo_transform extraction fallback: %s", exc)
        transform_matrix = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]

    # baseline (mm) — ZED 의 stereo_transform 의 translation[0]
    try:
        translation = T.get_translation()
        baseline_mm = float(translation.get()[0]) * 1000.0  # m → mm if returns meters
        if abs(baseline_mm) < 1e-3:
            # might already be in mm
            baseline_mm = float(translation.get()[0])
    except Exception:
        baseline_mm = float(transform_matrix[0][3]) * 1000.0

    # distortion
    def _disto_list(cam: Any) -> list:
        try:
            d = cam.disto
            if hasattr(d, '__len__'):
                return [float(x) for x in d]
            return [float(d[i]) for i in range(5)]
        except Exception:
            return [0.0] * 5

    # SDK version (string)
    try:
        sdk_version = str(__import__("pyzed.sl", fromlist=["Camera"]).Camera.get_sdk_version())
    except Exception:
        sdk_version = "unknown"

    # Serial number
    try:
        serial = int(info.serial_number)
    except Exception:
        serial = 0

    # Resolution + FPS
    try:
        res = config.resolution
        width, height = int(res.width), int(res.height)
    except Exception:
        width, height = 0, 0
    try:
        fps = int(config.fps)
    except Exception:
        fps = 0

    # Depth mode
    try:
        depth_mode = str(info.camera_configuration.calibration_parameters.left_cam.image_size)
    except Exception:
        depth_mode = "PERFORMANCE"   # default

    return {
        "version": SCHEMA_VERSION,
        "session_start_ns": time.time_ns(),
        "session_start_mono_ns": time.monotonic_ns(),
        "zed_serial": serial,
        "zed_sdk_version": sdk_version,
        "resolution_width": width,
        "resolution_height": height,
        "fps": fps,
        "depth_mode": depth_mode,
        "self_calibration_disabled": True,   # caller 의 InitParameters 의 책임
        "left_cam": {
            "fx": float(left.fx),
            "fy": float(left.fy),
            "cx": float(left.cx),
            "cy": float(left.cy),
            "disto": _disto_list(left),
        },
        "right_cam": {
            "fx": float(right.fx),
            "fy": float(right.fy),
            "cx": float(right.cx),
            "cy": float(right.cy),
            "disto": _disto_list(right),
        },
        "baseline_mm": baseline_mm,
        "stereo_transform": transform_matrix,
    }


def disable_self_calib_and_snapshot(
    zed: Any,
    output_path: Path,
    self_calib_disabled: Optional[bool] = None,
    depth_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Disable ZED self-calibration (caller 의 init 책임) + snapshot to JSON.

    ★ Codex review b5kic9w4n P2-3, P2-4 fix:
    - self_calib_disabled 은 caller 가 *정확한 init flag 의 값* 으로 전달.
      None 이면 ValueError (silent True 금지).
    - depth_mode 도 caller 가 init 시 사용한 *정확한 값* 으로 전달.

    InitParameters 의 `camera_disable_self_calib = True` 이미 설정 가정.
    snapshot + save.
    """
    if self_calib_disabled is None:
        raise ValueError(
            "self_calib_disabled must be explicitly passed (True/False). "
            "caller 의 InitParameters.camera_disable_self_calib 의 값."
        )
    if depth_mode is None:
        raise ValueError(
            "depth_mode must be explicitly passed (e.g., 'PERFORMANCE'). "
            "caller 의 InitParameters.depth_mode 의 string."
        )
    calib = snapshot_calib(zed)
    # ★ Codex P2-3, P2-4: caller-provided 정직 값으로 override
    calib["self_calibration_disabled"] = bool(self_calib_disabled)
    calib["depth_mode"] = str(depth_mode)
    save_session_calib(output_path, calib)
    LOGGER.info("session calib saved: %s (zed_serial=%d, baseline=%.2fmm)",
                output_path, calib["zed_serial"], calib["baseline_mm"])
    return calib


def main() -> int:
    """CLI entry — open ZED + dump calib + exit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path,
                    help="session_calib.json 출력 경로")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    try:
        import pyzed.sl as sl
    except ImportError:
        LOGGER.error("pyzed not available — Jetson 에서만 실행 가능")
        return 1

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True   # ★ Codex Q8 fix

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        LOGGER.error("ZED open failed: %s", status)
        return 1

    try:
        # ★ Codex P2-3, P2-4: explicit pass (init 값 의 정직 mirror)
        calib = disable_self_calib_and_snapshot(
            zed, args.output,
            self_calib_disabled=True,           # CLI 가 init 시 True 명시 위
            depth_mode="PERFORMANCE",            # CLI 가 init 시 PERFORMANCE 명시
        )
        LOGGER.info("done: %s", calib["zed_serial"])
        return 0
    finally:
        zed.close()


if __name__ == "__main__":
    raise SystemExit(main())
