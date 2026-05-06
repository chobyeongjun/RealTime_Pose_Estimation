#!/usr/bin/env python3
"""L_post Phase 0 packing self-test (Codex R5 권장).

publish-stage 통합 D2H의 packing layout이 정확한지 검증.
GPU 없이도 동작 (CPU tensors).

실행:
    python3 scripts/test_lpost_packing.py
"""
from __future__ import annotations

import sys

import torch


def test_packing_layout():
    """39 floats packing (K=6 case) layout 검증."""
    K = 6
    kpts_3d_m = torch.full((K, 3), 1.5)            # K*3 = 18 → all 1.5
    kpt_conf = torch.full((K,), 0.7)               # K = 6 → all 0.7
    kpts_2d_px = torch.full((K, 2), 100.0)         # K*2 = 12 → all 100.0
    box_conf_t = torch.tensor(0.9)                  # () → 0.9
    valid_mask_t = torch.tensor(True)               # () bool → True
    depth_invalid_ratio_t = torch.tensor(0.25)      # () → 0.25

    flat_gpu = torch.cat([
        kpts_3d_m.float().reshape(-1),
        kpt_conf.float().reshape(-1),
        kpts_2d_px.float().reshape(-1),
        box_conf_t.float().reshape(-1),
        valid_mask_t.to(dtype=torch.float32).reshape(-1),
        depth_invalid_ratio_t.float().reshape(-1),
    ], dim=0)
    flat = flat_gpu.detach().cpu().numpy().astype("float32", copy=False)

    expected_total = K * 3 + K + K * 2 + 1 + 1 + 1
    assert flat.shape == (expected_total,), f"flat shape {flat.shape} != ({expected_total},)"
    print(f"  T1 packing layout: {flat.shape} == ({expected_total},) — PASS")

    # Verify ranges:
    assert (flat[:K*3] == 1.5).all(), "kpts_3d_m section corrupted"
    assert (flat[K*3:K*3+K] == 0.7).all(), "kpt_conf section corrupted"
    assert (flat[K*3+K:K*3+K+K*2] == 100.0).all(), "kpts_2d_px section corrupted"
    assert abs(flat[K*3+K+K*2] - 0.9) < 1e-6, f"box_conf wrong: {flat[K*3+K+K*2]}"
    assert flat[K*3+K+K*2+1] == 1.0, f"valid_mask wrong: {flat[K*3+K+K*2+1]}"
    assert abs(flat[K*3+K+K*2+2] - 0.25) < 1e-6, f"depth_invalid wrong"
    print(f"  T2 section values — PASS")

    # Reconstruct CPU side (mirrors run_stream_demo.py path)
    kpts_3d_out = flat[:K*3].reshape(K, 3)
    kpt_conf_out = flat[K*3:K*3 + K]
    kpts_2d_out = flat[K*3 + K:K*3 + K + K*2].reshape(K, 2)
    gpu_box_conf = float(flat[K*3 + K + K*2])
    gpu_valid = bool(flat[K*3 + K + K*2 + 1] > 0.5)
    gpu_depth_invalid = float(flat[K*3 + K + K*2 + 2])

    assert kpts_3d_out.shape == (K, 3)
    assert kpt_conf_out.shape == (K,)
    assert kpts_2d_out.shape == (K, 2)
    assert gpu_box_conf == 0.9
    assert gpu_valid is True
    assert abs(gpu_depth_invalid - 0.25) < 1e-6
    print(f"  T3 reconstruction (kpts/conf/scalars) — PASS")


def test_invalid_mask_path():
    """valid_mask=False 시 publish gate가 invalid로 결정되는지."""
    K = 6
    kpts_3d_m = torch.zeros(K, 3)
    kpt_conf = torch.zeros(K)
    kpts_2d_px = torch.zeros(K, 2)
    box_conf_t = torch.tensor(0.0)              # below threshold
    valid_mask_t = torch.tensor(False)           # invalid
    depth_invalid_ratio_t = torch.tensor(1.0)    # all bad

    flat_gpu = torch.cat([
        kpts_3d_m.float().reshape(-1),
        kpt_conf.float().reshape(-1),
        kpts_2d_px.float().reshape(-1),
        box_conf_t.float().reshape(-1),
        valid_mask_t.to(dtype=torch.float32).reshape(-1),
        depth_invalid_ratio_t.float().reshape(-1),
    ], dim=0)
    flat = flat_gpu.detach().cpu().numpy().astype("float32", copy=False)

    gpu_valid = bool(flat[-2] > 0.5)
    assert gpu_valid is False, "invalid mask should produce False"
    print(f"  T4 invalid mask path — PASS")


def test_dtype_promotion_fp16_input():
    """TRT FP16 output 시뮬: kpts/conf가 FP16이어도 cat이 FP32로 정상 처리."""
    K = 6
    # FP16 input
    kpts_3d_m = torch.full((K, 3), 1.5, dtype=torch.float16)
    kpt_conf = torch.full((K,), 0.7, dtype=torch.float16)
    kpts_2d_px = torch.full((K, 2), 100.0, dtype=torch.float16)
    box_conf_t = torch.tensor(0.9, dtype=torch.float16)
    valid_mask_t = torch.tensor(True)
    depth_invalid_ratio_t = torch.tensor(0.25)

    flat_gpu = torch.cat([
        kpts_3d_m.float().reshape(-1),
        kpt_conf.float().reshape(-1),
        kpts_2d_px.float().reshape(-1),
        box_conf_t.float().reshape(-1),
        valid_mask_t.to(dtype=torch.float32).reshape(-1),
        depth_invalid_ratio_t.float().reshape(-1),
    ], dim=0)
    flat = flat_gpu.detach().cpu().numpy().astype("float32", copy=False)

    assert flat.dtype.name == "float32"
    # FP16 → FP32 cast is exact for these values; box_conf=0.9 not exact in fp16
    # so allow small tolerance.
    assert abs(flat[K*3+K+K*2] - 0.9) < 1e-3, f"box_conf wrong (fp16 path): {flat[K*3+K+K*2]}"
    print(f"  T5 FP16 → FP32 cast — PASS")


def main() -> int:
    print("=== L_post packing self-test ===")
    try:
        test_packing_layout()
        test_invalid_mask_path()
        test_dtype_promotion_fp16_input()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    print("\nALL L_post packing self-tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
