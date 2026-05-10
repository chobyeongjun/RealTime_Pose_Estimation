"""TDD red phase — A.2 post fusion vs unfused output 검증.

Codex consult Q6 (2026-05-10) spec.

목적:
    `gpu_postprocess_fused.py` 가 작성되기 *전* 이 test 작성 (TDD red).
    fused output 이 unfused (current GpuPostprocessor.post()) 의 reference 와
    수치적으로 동일한지 검증.

실행 (Jetson, CUDA 필요):
    pytest tests/test_post_fusion.py -v

Pass criteria (Codex Q6):
    - 2D max diff ≤ 0.25 px (synthetic), ≤ 0.5 px (recorded)
    - 3D max diff ≤ 1e-4 m (simple), ≤ 2e-3 m (patch nanmedian),
                  ≤ 5e-3 m (FP16)
    - valid_mask: bit-exact
    - perf gate: fused saves ≥ 1.0 ms p50 또는 ≥ 0.8 ms p99 over 500 frames

NOTE — gpu_postprocess_fused.py 미존재 시 ImportError 로 fail (TDD red).
       다음 commit 에서 implement → green phase.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np


# ----------------------------------------------------------------------
# Test fixtures — synthetic detection + depth + calibration
# ----------------------------------------------------------------------

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — Jetson 에서만 실행")
    return torch.device("cuda:0")


@pytest.fixture
def calib():
    """ZED X Mini SVGA 의 일반적 intrinsic (synthetic)."""
    return {
        "fx": 480.0,
        "fy": 480.0,
        "cx": 480.0,
        "cy": 300.0,
        "R_world_from_cam": None,
    }


@pytest.fixture
def calib_with_R():
    """IMU rotation 포함."""
    R = torch.eye(3, device="cuda:0")
    return {
        "fx": 480.0, "fy": 480.0, "cx": 480.0, "cy": 300.0,
        "R_world_from_cam": R,
    }


@pytest.fixture
def synthetic_keypoints(device):
    """K=6 lower-limb keypoints in letterbox space (640x640)."""
    return torch.tensor([
        [320.0, 380.0],   # L_hip
        [340.0, 380.0],   # R_hip
        [310.0, 460.0],   # L_knee
        [350.0, 460.0],   # R_knee
        [305.0, 540.0],   # L_ankle
        [355.0, 540.0],   # R_ankle
    ], device=device)


@pytest.fixture
def synthetic_depth(device):
    """1.5m flat depth (ZED PERFORMANCE 모드 가정)."""
    H, W = 600, 960
    depth = torch.full((H, W), 1.5, device=device)
    return depth


@pytest.fixture
def synthetic_kp_conf(device):
    return torch.tensor([0.9, 0.9, 0.85, 0.85, 0.8, 0.8], device=device)


@pytest.fixture
def letterbox_params():
    from perception.CUDA_Stream.gpu_preprocess import LetterboxParams
    return LetterboxParams(
        pad_x=0.0, pad_y=20.0, scale=0.667, src_w=960, src_h=600,
    )


# ----------------------------------------------------------------------
# Reference implementation — current GpuPostprocessor.post() path
# ----------------------------------------------------------------------

@pytest.fixture
def reference_post(device):
    """Current PyTorch implementation (gpu_postprocess.py post path) 의 fixture.

    fused vs reference 비교의 oracle.
    """
    from perception.CUDA_Stream.gpu_postprocess import GpuPostprocessor
    from perception.CUDA_Stream.keypoint_config import get_schema
    schema = get_schema("lowlimb6")
    return GpuPostprocessor(schema=schema, device=device, use_filter=False)


# ----------------------------------------------------------------------
# RED phase — gpu_postprocess_fused.py 미존재 → ImportError fail
# ----------------------------------------------------------------------

class TestFusedExists:
    """A.2 fused module 의 import 가능 여부."""

    def test_module_importable(self):
        """gpu_postprocess_fused.py 가 작성되어야 함."""
        from perception.CUDA_Stream import gpu_postprocess_fused  # noqa: F401

    def test_has_fused_post_function(self):
        from perception.CUDA_Stream import gpu_postprocess_fused
        assert hasattr(gpu_postprocess_fused, "fused_post")

    def test_has_pytorch_fallback(self):
        """Triton 미설치 시 PyTorch path 가용."""
        from perception.CUDA_Stream import gpu_postprocess_fused
        assert hasattr(gpu_postprocess_fused, "fused_post_pytorch")

    def test_triton_optional(self):
        """Triton import 가 optional (try/except) — Jetson 에서 미설치 가능."""
        from perception.CUDA_Stream import gpu_postprocess_fused
        # HAS_TRITON flag 가 module level 에 있어야
        assert hasattr(gpu_postprocess_fused, "HAS_TRITON")


# ----------------------------------------------------------------------
# Numerical correctness — fused vs reference
# ----------------------------------------------------------------------

class TestNumericalEquivalence:
    """Codex Q6 — 수치 tolerance."""

    TOL_2D_PX_SYNTH = 0.25
    TOL_3D_M_SIMPLE = 1e-4
    TOL_3D_M_PATCH = 2e-3

    def test_simple_synthetic_match(
        self, device, synthetic_keypoints, synthetic_depth,
        synthetic_kp_conf, letterbox_params, calib, reference_post,
    ):
        """동일 input 으로 fused vs reference output RMSE 검증."""
        from perception.CUDA_Stream import gpu_postprocess_fused

        # reference (current path)
        # ref_xy_src, ref_xyz, ref_valid = reference_post._compute_for_test(
        #     synthetic_keypoints, synthetic_depth, synthetic_kp_conf,
        #     letterbox_params, calib,
        # )
        # fused
        # fused_xy_src, fused_xyz, fused_valid = gpu_postprocess_fused.fused_post(
        #     synthetic_keypoints, synthetic_depth, synthetic_kp_conf,
        #     letterbox_params, calib,
        # )

        # 2D tolerance
        # diff_2d = (ref_xy_src - fused_xy_src).abs().max().item()
        # assert diff_2d <= self.TOL_2D_PX_SYNTH

        # 3D tolerance (patch nanmedian 영향)
        # diff_3d = (ref_xyz - fused_xyz).abs().max().item()
        # assert diff_3d <= self.TOL_3D_M_PATCH

        # valid_mask bit-exact
        # assert torch.equal(ref_valid, fused_valid)
        pytest.skip("RED phase — gpu_postprocess_fused.fused_post 미구현")

    def test_depth_nan_handling(self, device, synthetic_keypoints,
                                  synthetic_kp_conf, letterbox_params, calib):
        """depth=NaN keypoint → valid=False, 3D=0."""
        H, W = 600, 960
        depth = torch.full((H, W), float('nan'), device=device)
        # ... fused 호출 + valid mask 검증
        pytest.skip("RED phase")

    def test_zero_depth(self, device, synthetic_keypoints,
                        synthetic_kp_conf, letterbox_params, calib):
        """depth=0 → invalid (CLAUDE.md: np.isfinite(z) and z > 0 가드)."""
        H, W = 600, 960
        depth = torch.zeros((H, W), device=device)
        pytest.skip("RED phase")

    def test_negative_depth(self, device, synthetic_keypoints,
                             synthetic_kp_conf, letterbox_params, calib):
        """depth<0 → invalid."""
        pytest.skip("RED phase")

    def test_kp_conf_below_threshold(self, device, synthetic_keypoints,
                                       synthetic_depth, letterbox_params, calib):
        """kp_conf < threshold → 3D=0 (gating)."""
        kp_conf = torch.full((6,), 0.1, device=device)  # below default 0.5
        pytest.skip("RED phase")

    def test_R_identity_vs_None(self, device, synthetic_keypoints,
                                  synthetic_depth, synthetic_kp_conf,
                                  letterbox_params, calib_with_R):
        """R=I 와 R=None 동일 결과."""
        pytest.skip("RED phase")

    def test_R_nontrivial_rotation(self, device, synthetic_keypoints,
                                     synthetic_depth, synthetic_kp_conf,
                                     letterbox_params):
        """R = 32deg tilt (Walker mount) 검증."""
        import math
        c, s = math.cos(math.radians(32)), math.sin(math.radians(32))
        R = torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ], device=device, dtype=torch.float32)
        pytest.skip("RED phase")

    def test_clamp_at_image_borders(self, device, synthetic_depth,
                                      synthetic_kp_conf, letterbox_params, calib):
        """letterbox 외 keypoint → src_w-1, src_h-1 clamp."""
        kp = torch.tensor([
            [-100.0, -100.0],   # 좌상단 outside
            [10000.0, 10000.0],  # 우하단 outside
            [320.0, 380.0],
            [340.0, 380.0],
            [350.0, 460.0],
            [355.0, 540.0],
        ], device=device)
        pytest.skip("RED phase")

    def test_lowlimb6_schema(self):
        pytest.skip("RED phase")

    def test_coco17_schema(self):
        """K=17 case 도 동작."""
        pytest.skip("RED phase")


# ----------------------------------------------------------------------
# PyTorch fallback (Triton 미설치 시)
# ----------------------------------------------------------------------

class TestPyTorchFallback:
    def test_fallback_same_output(self, device, synthetic_keypoints,
                                    synthetic_depth, synthetic_kp_conf,
                                    letterbox_params, calib):
        """fused_post_pytorch 가 fused_post 와 동일 output."""
        pytest.skip("RED phase")

    def test_fallback_no_triton(self, monkeypatch):
        """HAS_TRITON=False forced → PyTorch path 자동 선택."""
        pytest.skip("RED phase")


# ----------------------------------------------------------------------
# Performance gate (Jetson 에서 측정)
# ----------------------------------------------------------------------

class TestPerformanceGate:
    """Codex Q6 — fused 가 unfused 대비 ≥ 1.0ms p50 또는 ≥ 0.8ms p99 절감."""

    PERF_GATE_P50_MS = 1.0
    PERF_GATE_P99_MS = 0.8
    N_FRAMES = 500

    def test_perf_gate(self, device):
        """500 warm frames 에서 fused vs unfused 측정."""
        pytest.skip("RED phase — Jetson 에서 measurement 가능")
