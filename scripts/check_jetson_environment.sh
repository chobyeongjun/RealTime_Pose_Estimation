#!/usr/bin/env bash
# Jetson 환경 정직 검증 — ZED SDK + GMSL + jetson_clocks + CPU 사용 패턴.
#
# ZED RawBuffer prototype 시작 전 prerequisite. 결과 paste 하면
# 다음 turn 의 prototype implementation 가능.
#
# 사용:
#   cd ~/realtime-vision-control
#   bash scripts/check_jetson_environment.sh

echo "============================================================"
echo "  Jetson environment check — $(date +%Y-%m-%d_%H:%M:%S)"
echo "============================================================"

echo ""
echo "--- (1) L4T / JetPack version ---"
cat /etc/nv_tegra_release 2>/dev/null | head -3 || echo "  L4T file not found"

echo ""
echo "--- (2) ZED SDK 설치 + 버전 ---"
ls /usr/local/zed/version.txt 2>/dev/null && cat /usr/local/zed/version.txt 2>/dev/null
ls /usr/local/zed/lib/libsl*.so 2>/dev/null | head -3 || echo "  ZED SDK not in /usr/local/zed/"
dpkg -l 2>/dev/null | grep -i "zed-sdk\|stereolabs" | head -5 || echo "  no dpkg zed-sdk entry"

echo ""
echo "--- (3) pyzed Python API ---"
PYTHONPATH=src python3 -c "
import sys
try:
    import pyzed.sl as sl
    print('  pyzed import OK')
    cam = sl.Camera()
    ver = sl.Camera.get_sdk_version()
    print(f'  SDK version: {ver}')
except Exception as e:
    print(f'  pyzed import FAIL: {e}')
" 2>&1

echo ""
echo "--- (4) ZED RawBuffer / NvBufSurface API 가용 여부 ---"
PYTHONPATH=src python3 -c "
import pyzed.sl as sl
cam = sl.Camera()
methods = sorted([m for m in dir(cam) if 'raw' in m.lower() or 'buffer' in m.lower()])
print(f'  Camera methods (raw/buffer): {methods}')

mat_methods = sorted([m for m in dir(sl.Mat) if 'raw' in m.lower() or 'buffer' in m.lower() or 'native' in m.lower() or 'gpu' in m.lower()])
print(f'  Mat methods (raw/buffer/native/gpu): {mat_methods}')

if hasattr(sl, 'MEM'):
    mem_types = [m for m in dir(sl.MEM) if not m.startswith('_')]
    print(f'  MEM types: {mem_types}')

# RUNTIME_PARAMETERS 검사 — image_only 같은 옵션
rt_params = sl.RuntimeParameters()
attrs = sorted([a for a in dir(rt_params) if not a.startswith('_')])
print(f'  RuntimeParameters attrs: {attrs}')
" 2>&1

echo ""
echo "--- (5) GMSL / V4L2 video devices ---"
ls /dev/video* 2>/dev/null || echo "  no /dev/video*"
v4l2-ctl --list-devices 2>/dev/null | head -15 || echo "  v4l2-ctl not installed (apt install v4l-utils)"

echo ""
echo "--- (6) jetson_clocks status ---"
sudo /usr/bin/jetson_clocks --show 2>/dev/null | head -20 || echo "  jetson_clocks (try sudo)"

echo ""
echo "--- (7) nvpmodel current ---"
sudo /usr/sbin/nvpmodel -q 2>/dev/null | head -5 || echo "  nvpmodel (try sudo)"

echo ""
echo "--- (8) CPU info + count ---"
echo "  nproc: $(nproc)"
grep "model name" /proc/cpuinfo | head -1
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "  scaling_governor not available"
echo "  current cpufreq:"
for c in 0 4 7; do
    F=$(cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_cur_freq 2>/dev/null)
    [ -n "$F" ] && echo "    cpu${c}: ${F} kHz"
done

echo ""
echo "--- (9) 현재 system service CPU 사용 (5 sec sample) ---"
echo "  (top 명령 5s sampling — Xorg, nvargus-daemon 등 확인)"
top -b -n 2 -d 2 2>/dev/null | grep -iE "Xorg|nvargus|systemd|gdm|gnome|jetson" | head -15

echo ""
echo "--- (10) 현재 process CPU affinity ---"
ps -eo pid,comm,psr,pcpu --sort=-pcpu 2>/dev/null | head -15

echo ""
echo "--- (11) NVIDIA VPI library ---"
dpkg -l 2>/dev/null | grep -i "vpi" | head -5 || echo "  no VPI dpkg"
ls /opt/nvidia/vpi*/ 2>/dev/null | head -5 || echo "  /opt/nvidia/vpi* not found"

echo ""
echo "--- (12) PyTorch + CuPy + CUDA 버전 ---"
PYTHONPATH=src python3 -c "
import torch
print(f'  torch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}')
try:
    import cupy
    print(f'  cupy: {cupy.__version__}, CUDA: {cupy.cuda.runtime.runtimeGetVersion()}')
except ImportError as e:
    print(f'  cupy: {e}')
try:
    import triton
    print(f'  triton: {triton.__version__}')
except ImportError as e:
    print(f'  triton: {e}')
" 2>&1

echo ""
echo "============================================================"
echo "  Done. Paste this output → ZED RawBuffer prototype 가능 여부 결정."
echo "============================================================"
