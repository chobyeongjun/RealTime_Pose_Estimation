#!/usr/bin/env bash
# V4L2 capability check — ZED X Mini 의 raw format 검증 (V4L2 우회 prototype 의 prereq).
#
# 사용 (Jetson):
#   bash scripts/check_v4l2_capability.sh 2>&1 | tee /tmp/v4l2_check.log
#
# 출력: format (NV12 / Bayer / YUYV) + raw calib (rectified 와 다른) + libargus 가용 여부.

echo "============================================================"
echo "  V4L2 capability check — ZED X Mini bypass prereq"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "============================================================"

echo ""
echo "--- (1) v4l2-ctl installed? ---"
if ! command -v v4l2-ctl >/dev/null; then
    echo "  ⚠️ v4l2-ctl 미설치 — sudo apt install v4l-utils"
    exit 1
fi
v4l2-ctl --version | head -1

echo ""
echo "--- (2) Devices ---"
ls -la /dev/video* 2>/dev/null

echo ""
echo "--- (3) /dev/video0 의 list-formats-ext (left camera 추정) ---"
v4l2-ctl --device /dev/video0 --list-formats-ext 2>&1 | head -40

echo ""
echo "--- (4) /dev/video0 current format ---"
v4l2-ctl --device /dev/video0 --get-fmt-video 2>&1

echo ""
echo "--- (5) /dev/video1 의 list-formats-ext (right camera 추정) ---"
v4l2-ctl --device /dev/video1 --list-formats-ext 2>&1 | head -40

echo ""
echo "--- (6) ZED calibration_parameters_raw 추출 (Python) ---"
PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PWD/src" \
    python3 -c "
import sys
try:
    import pyzed.sl as sl
except ImportError as e:
    print(f'pyzed not available: {e}')
    sys.exit(1)

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.SVGA
init.camera_fps = 120
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init.coordinate_units = sl.UNIT.METER
init.camera_disable_self_calib = True

status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(f'ZED open failed: {status}')
    sys.exit(1)

try:
    info = zed.get_camera_information()
    config = info.camera_configuration
    rect = config.calibration_parameters
    print(f'Rectified left:  fx={rect.left_cam.fx:.2f}, cx={rect.left_cam.cx:.2f}')
    print(f'  disto[{len(rect.left_cam.disto)}] = {list(rect.left_cam.disto)[:6]}...')

    raw = getattr(config, 'calibration_parameters_raw', None)
    if raw is None:
        print('  ⚠️ calibration_parameters_raw not available — V4L2 path 시 issue')
    else:
        print(f'Raw left:        fx={raw.left_cam.fx:.2f}, cx={raw.left_cam.cx:.2f}')
        print(f'  disto[{len(raw.left_cam.disto)}] = {list(raw.left_cam.disto)[:6]}...')
        rect_zero = all(x == 0.0 for x in rect.left_cam.disto)
        raw_nonzero = any(x != 0.0 for x in raw.left_cam.disto)
        if rect_zero and raw_nonzero:
            print('  ✓ raw vs rectified 의 difference 확인 — V4L2 raw 사용 시 raw disto 의무')
        elif rect_zero and not raw_nonzero:
            print('  ⚠️ raw also all zero — perhaps already rectified by hardware')
finally:
    zed.close()
" 2>&1

echo ""
echo "--- (7) libargus / Argus camera 가용 ---"
ls /usr/lib/aarch64-linux-gnu/tegra/libnvargus* 2>/dev/null | head -5 || echo "  no libnvargus*"
which gst-launch-1.0 >/dev/null 2>&1 && echo "  gstreamer 가용 (gst-launch-1.0)" || echo "  no gstreamer"

echo ""
echo "--- (8) NVIDIA VPI version ---"
dpkg -l 2>/dev/null | grep -i "vpi" | head -3

echo ""
echo "============================================================"
echo "  Done. Paste output → V4L2 path 결정 prereq."
echo "============================================================"
