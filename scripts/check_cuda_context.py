#!/usr/bin/env python3
"""ZED + PyTorch CUDA context 검증 (Jetson 전용).

목적
----
ZED SDK와 PyTorch가 같은 CUDA context를 쓰는지 확인.
이 결과가 ZED CUDA interop 구현 경로를 결정한다:

    모든 ctx 동일 → DLPack / 직접 PyTorch wrap 가능 (interop 길 1, ~1주)
    ctx 다름      → C++ extension + sdk_cuda_ctx 필요 (interop 길 2, ~2주)

근거: Codex R5 답변 (xhigh reasoning, 2026-05-05).

사용법 (Jetson)
--------------
    # 다른 ZED process 종료 후
    sudo pkill -f run_stream_demo
    python3 scripts/check_cuda_context.py

출력 예시
--------
    cuInit returned: 0
    after cuInit                    err= 0  ctx= 0x0
    after torch init                err= 0  ctx= 0x55a1234567ab
    after zed open                  err= 0  ctx= 0x55a1234567ab   ← 같으면 OK
    after zed grab                  err= 0  ctx= 0x55a1234567ab
    after retrieve_image            err= 0  ctx= 0x55a1234567ab
    after retrieve_measure          err= 0  ctx= 0x55a1234567ab
    ============================================================
    판정:
      ✓ 모든 ctx 동일 → DLPack/PyTorch wrap 경로 가능 (interop 길 1)
    ============================================================
"""
from __future__ import annotations

import ctypes
import sys


LIBCUDA_CANDIDATES = [
    "libcuda.so",
    "libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/libcuda.so",
    "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so",
    "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/local/cuda/lib64/libcuda.so.1",
]


def load_libcuda():
    last_err = None
    for path in LIBCUDA_CANDIDATES:
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            last_err = e
            continue
    print(f"ERROR: libcuda.so not found in {LIBCUDA_CANDIDATES}", file=sys.stderr)
    print(f"  last error: {last_err}", file=sys.stderr)
    print(f"  → ldconfig -p | grep libcuda 로 실제 경로 확인", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    libcuda = load_libcuda()

    cuInit = libcuda.cuInit
    cuInit.restype = ctypes.c_int
    cuInit.argtypes = [ctypes.c_uint]

    cuCtxGetCurrent = libcuda.cuCtxGetCurrent
    cuCtxGetCurrent.restype = ctypes.c_int
    cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    seen: list[str] = []

    def ctx(label: str) -> str:
        p = ctypes.c_void_p(0)
        err = cuCtxGetCurrent(ctypes.byref(p))
        s = hex(p.value or 0)
        print(f"  {label:<30}  err= {err}  ctx= {s}")
        return s

    err = cuInit(0)
    print(f"cuInit returned: {err}")
    seen.append(ctx("after cuInit"))

    # Step 1: PyTorch CUDA init
    try:
        import torch  # noqa: WPS433
    except ImportError as e:
        print(f"ERROR: torch import failed: {e}", file=sys.stderr)
        print(f"  → setup_jetson.sh 경유로 torch 설치 필요", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(f"ERROR: torch.cuda.is_available() = False", file=sys.stderr)
        print(f"  → Jetson CUDA 환경 점검 필요", file=sys.stderr)
        return 1
    torch.cuda.init()
    # primary context lazy init — force one alloc
    _ = torch.empty((1,), device="cuda")
    seen.append(ctx("after torch init"))

    # Step 2: ZED SDK init
    try:
        import pyzed.sl as sl  # noqa: WPS433
    except ImportError as e:
        print(f"ERROR: pyzed import failed: {e}", file=sys.stderr)
        print(f"  → 이 script는 Jetson (ZED SDK 설치된 호스트) 에서만 의미 있음", file=sys.stderr)
        return 1

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    open_status = zed.open(init)
    print(f"zed open status: {open_status}")
    if str(open_status) != "SUCCESS":
        print(f"ERROR: ZED open failed: {open_status}", file=sys.stderr)
        print(f"  → 다른 ZED process가 잡고 있을 가능성", file=sys.stderr)
        print(f"    sudo pkill -f run_stream_demo; sudo systemctl restart nvargus-daemon", file=sys.stderr)
        return 1
    seen.append(ctx("after zed open"))

    # Step 3: ZED grab + retrieve
    rt = sl.RuntimeParameters()
    grab_status = zed.grab(rt)
    print(f"zed grab status: {grab_status}")
    if str(grab_status) == "SUCCESS":
        seen.append(ctx("after zed grab"))

        mat = sl.Mat()
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        seen.append(ctx("after retrieve_image"))

        depth_mat = sl.Mat()
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        seen.append(ctx("after retrieve_measure"))

    zed.close()

    # 판정
    print()
    print("=" * 60)
    print("판정:")
    unique = set(c for c in seen if c != "0x0")
    if len(unique) <= 1:
        print(f"  ✓ 모든 ctx 동일 ({list(unique)[0] if unique else '없음'})")
        print(f"    → DLPack/PyTorch wrap 경로 가능 (interop 길 1, ~1주)")
        result = 0
    else:
        print(f"  ⚠ ctx 다름:")
        for c in sorted(unique):
            print(f"      {c}")
        print(f"    → C++ extension + sdk_cuda_ctx 필요 (interop 길 2, ~2주)")
        result = 0  # 정보 출력은 정상 종료

    print("=" * 60)
    return result


if __name__ == "__main__":
    sys.exit(main())
