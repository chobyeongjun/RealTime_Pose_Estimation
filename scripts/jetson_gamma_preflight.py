"""γ ZED CUDA interop pre-flight — 1 frame probe.

사용법:
    cd ~/realtime-vision-control
    PYTHONPATH=src python3 scripts/jetson_gamma_preflight.py

기대 출력:
    === gamma PRE-FLIGHT PASSED ===

검증:
    1. CuPy import OK
    2. ZED retrieve_image(MEM.GPU) OK
    3. CPU/GPU pixel parity (bit-perfect)
    4. ZED retrieve_measure(MEM.GPU) OK
    5. depth finite ratio > 0.3

Fail 시: γ case (08-11) skip, α case (00-07) 만 측정.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import cupy
    except ImportError:
        print("FAIL: CuPy not installed. Install: pip3 install --user cupy-cuda12x")
        return 1

    print(f"CuPy: {cupy.__version__}")

    import torch
    import numpy as np
    import pyzed.sl as sl
    from torch.utils.dlpack import from_dlpack

    # 1. ZED open
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"FAIL: zed open: {status}")
        return 1

    rt = sl.RuntimeParameters()
    status = zed.grab(rt)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print(f"FAIL: zed grab: {status}")
        return 1

    # 2. CPU retrieve (parity baseline)
    mat_cpu = sl.Mat()
    zed.retrieve_image(mat_cpu, sl.VIEW.LEFT)
    rgb_cpu = mat_cpu.get_data(deep_copy=True)
    print(f"CPU RGB: shape={rgb_cpu.shape} dtype={rgb_cpu.dtype}")

    # 3. GPU retrieve
    mat_gpu = sl.Mat()
    status = zed.retrieve_image(mat_gpu, sl.VIEW.LEFT, sl.MEM.GPU)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print(f"FAIL: retrieve_image MEM.GPU: {status}")
        return 1

    rgb_cu = mat_gpu.get_data(sl.MEM.GPU, deep_copy=False)
    print(f"CuPy RGB: type={type(rgb_cu).__name__} shape={rgb_cu.shape} dtype={rgb_cu.dtype}")

    rgb_view = from_dlpack(rgb_cu)
    rgb_gpu = torch.empty(tuple(rgb_view.shape), dtype=torch.uint8, device="cuda:0")
    rgb_gpu.copy_(rgb_view, non_blocking=True)
    torch.cuda.synchronize()

    # 4. Pixel parity (center)
    h, w = rgb_cpu.shape[:2]
    cy, cx = h // 2, w // 2
    rgb_back = rgb_gpu.cpu().numpy()
    diff = int(np.abs(rgb_cpu[cy, cx].astype(int) - rgb_back[cy, cx].astype(int)).max())
    print(f"Pixel diff (center): {diff} (expect 0)")
    if diff != 0:
        zed.close()
        print(f"FAIL: pixel parity diff={diff}")
        return 1

    # 5. Depth GPU
    mat_d_gpu = sl.Mat()
    status = zed.retrieve_measure(mat_d_gpu, sl.MEASURE.DEPTH, sl.MEM.GPU)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print(f"FAIL: retrieve_measure DEPTH MEM.GPU: {status}")
        return 1

    depth_cu = mat_d_gpu.get_data(sl.MEM.GPU, deep_copy=False)
    depth_view = from_dlpack(depth_cu)
    if depth_view.ndim == 3 and depth_view.shape[-1] == 1:
        depth_view = depth_view[..., 0]
    print(f"Depth GPU: shape={tuple(depth_view.shape)} dtype={depth_view.dtype}")

    finite = torch.isfinite(depth_view).float().mean().item()
    print(f"Depth finite ratio: {finite:.3f} (expect > 0.3)")
    if finite <= 0.3:
        zed.close()
        print(f"FAIL: depth all NaN — {finite:.3f}")
        return 1

    zed.close()
    print()
    print("=== gamma PRE-FLIGHT PASSED ===")
    print("    --zed-cuda-interop 안전. case 08-11 측정 가능.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
