#!/usr/bin/env python3
"""L1 BGRA path self-test — GPU preproc이 BGRA 4ch와 RGB 3ch 동등 처리.

Mac (CPU only)에서는 GpuPreprocessor가 CUDA tensor 요구하므로 syntax check만.
Jetson에서 실행 시 실제 GPU 동등성 검증.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perception.CUDA_Stream.gpu_preprocess import GpuPreprocessor


def test_bgra_equivalent_to_rgb():
    """BGRA 4ch와 동일 색상의 RGB 3ch를 preproc 거치면 같은 출력."""
    if not torch.cuda.is_available():
        print("  (skipped — no CUDA, syntax check only)")
        return

    H, W = 600, 960
    # PyTorch device("cuda") and device("cuda:0") are not equal — must specify index
    device = torch.device("cuda:0")

    # 합성 BGRA (B=10, G=100, R=200, A=255)
    bgra = np.zeros((H, W, 4), dtype=np.uint8)
    bgra[..., 0] = 10
    bgra[..., 1] = 100
    bgra[..., 2] = 200
    bgra[..., 3] = 255

    # 동등 RGB (R=200, G=100, B=10)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    rgb[..., 1] = 100
    rgb[..., 2] = 10

    bgra_gpu = torch.from_numpy(bgra).to(device)
    rgb_gpu = torch.from_numpy(rgb).to(device)

    pre = GpuPreprocessor(imgsz=640, dtype=torch.float32, device=device)
    stream = torch.cuda.Stream()

    out_bgra, lb_bgra = pre(bgra_gpu, stream=stream)
    out_bgra_copy = out_bgra.clone()  # save before next call overwrites self.out

    out_rgb, lb_rgb = pre(rgb_gpu, stream=stream)
    stream.synchronize()

    assert lb_bgra.src_h == lb_rgb.src_h == H
    assert lb_bgra.src_w == lb_rgb.src_w == W
    assert lb_bgra.scale == lb_rgb.scale

    diff = (out_bgra_copy - out_rgb).abs().max().item()
    assert diff < 1e-4, f"BGRA → RGB equivalence: max diff {diff} (expected <1e-4)"
    print(f"  T1 BGRA == RGB output (max diff {diff:.2e}) — PASS")


def test_shape_validation():
    """잘못된 channel 수는 에러."""
    if not torch.cuda.is_available():
        print("  (skipped — no CUDA)")
        return
    device = torch.device("cuda:0")
    pre = GpuPreprocessor(imgsz=640, dtype=torch.float32, device=device)
    stream = torch.cuda.Stream()

    # 5-channel — 에러
    bad = torch.zeros((100, 100, 5), dtype=torch.uint8, device=device)
    try:
        pre(bad, stream=stream)
        assert False, "5-channel should raise"
    except ValueError as e:
        assert "expected" in str(e)
        print(f"  T2 shape validation — PASS")


def main() -> int:
    print("=== L1 BGRA self-test ===")
    try:
        test_bgra_equivalent_to_rgb()
        test_shape_validation()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    print("\nALL L1 self-tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
