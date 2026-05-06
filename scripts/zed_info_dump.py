"""ZED SDK / camera info dump.

Plan v7 (2026-05-07) — zed_lag 21ms 진단을 위한 system info 수집.
SDK 버전, 펌웨어, GMSL link, 설정 가능한 옵션 모두 출력.

사용법:
    python3 scripts/zed_info_dump.py [--resolution SVGA] [--fps 120]
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", default="SVGA")
    ap.add_argument("--fps", type=int, default=120)
    args = ap.parse_args()

    try:
        import pyzed.sl as sl
    except ImportError:
        print("ERROR: pyzed.sl not available. Run on Jetson with ZED SDK installed.")
        return 1

    print("=" * 64)
    print(f"  ZED SDK version: {sl.Camera.get_sdk_version()}")
    print("=" * 64)

    # VIDEO_SETTINGS / DEPTH_MODE / SENSING_MODE 등 enum 보유 여부
    print("\n[ enum availability ]")
    for name in [
        "VIDEO_SETTINGS", "DEPTH_MODE", "SENSING_MODE",
        "TIME_REFERENCE", "RESOLUTION", "MEM",
    ]:
        attr = getattr(sl, name, None)
        if attr is None:
            print(f"  {name:20s} → NOT AVAILABLE")
            continue
        members = [m for m in dir(attr) if not m.startswith("_") and m.isupper()]
        print(f"  {name:20s} → {members}")

    # VIDEO_SETTINGS.EXPOSURE_TIME (microseconds API) 존재 여부
    vs = getattr(sl, "VIDEO_SETTINGS", None)
    if vs is not None:
        has_exposure_time = hasattr(vs, "EXPOSURE_TIME")
        print(f"\n  EXPOSURE_TIME (microseconds) API: "
              f"{'AVAILABLE' if has_exposure_time else 'NOT AVAILABLE (use EXPOSURE 0-100)'}")

    # 카메라 open + info
    print("\n[ camera info ]")
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: ZED open failed: {status}")
        return 1

    info = zed.get_camera_information()
    cfg = info.camera_configuration

    print(f"  Serial number         : {info.serial_number}")
    print(f"  Camera model          : {info.camera_model}")
    print(f"  Firmware version      : {cfg.firmware_version}")
    print(f"  Resolution            : {cfg.resolution.width}x{cfg.resolution.height}")
    print(f"  FPS                   : {cfg.fps}")

    # exposure 현재 값 (AUTO 일 때 -1, MANUAL 일 때 percentage 또는 us)
    cur_exp = zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
    print(f"  Current EXPOSURE      : {cur_exp} (-1=AUTO)")
    if hasattr(sl.VIDEO_SETTINGS, "EXPOSURE_TIME"):
        cur_exp_us = zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_TIME)
        print(f"  Current EXPOSURE_TIME : {cur_exp_us} us")

    # gain
    cur_gain = zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)
    print(f"  Current GAIN          : {cur_gain}")

    # RuntimeParameters defaults
    print("\n[ RuntimeParameters defaults ]")
    rt = sl.RuntimeParameters()
    for attr in [
        "sensing_mode", "enable_fill_mode", "confidence_threshold",
        "texture_confidence_threshold", "measure3D_reference_frame",
    ]:
        val = getattr(rt, attr, "<n/a>")
        print(f"  {attr:32s} = {val}")

    zed.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
