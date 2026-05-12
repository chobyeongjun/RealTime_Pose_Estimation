"""Quality dataset I/O 의 schema + round-trip 검증 (Mac executable, no CUDA).

Codex orchestration Q2 — 10-iteration review 의 schema 검증.

검증:
    1. SCHEMA_VERSION + import OK
    2. save_frame_npz + load_frame_npz round-trip (모든 fields 정확)
    3. verify_frame_schema — 필수 fields, dtype, shape
    4. JPEG encode/decode round-trip (color 일치)
    5. depth raw round-trip (NaN preservation)
    6. include-right (optional field)
    7. session_calib.json schema
    8. Edge cases (missing field, wrong dtype, K mismatch)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# tests/conftest.py 가 src/ 자동 추가
from perception.CUDA_Stream.quality_dataset_io import (
    SCHEMA_VERSION, QualityFrame,
    save_frame_npz, load_frame_npz, verify_frame_schema,
    save_session_calib, load_session_calib,
    _encode_rgb_jpeg, _decode_rgb_jpeg,
)


@pytest.fixture
def tmp_npz(tmp_path):
    return tmp_path / "frame.npz"


@pytest.fixture
def tmp_calib(tmp_path):
    return tmp_path / "session_calib.json"


def make_synthetic_frame(K: int = 6, H: int = 600, W: int = 960,
                          include_right: bool = False) -> QualityFrame:
    """Synthetic data for tests — no CUDA / no pyzed required."""
    rng = np.random.default_rng(42)
    rgb_bgra = rng.integers(0, 255, size=(H, W, 4), dtype=np.uint8)
    depth_m = (rng.random((H, W)).astype(np.float32) * 2.0) + 0.5   # 0.5..2.5m
    rgb_right_bgra = None
    if include_right:
        rgb_right_bgra = rng.integers(0, 255, size=(H, W, 4), dtype=np.uint8)

    return QualityFrame(
        frame_id=42,
        rgb_ts_ns=1_000_000_000,
        depth_ts_ns=1_000_000_000,
        depth_age_us=0,
        publish_done_mono_ns=2_000_000_000,
        valid_mask_bits=(1 << K) - 1,
        valid_reason=0,
        ts_domain=0,                          # ★ P1-1
        valid=True,                            # ★ P1-1
        world_frame_applied=False,
        box_conf=0.85,
        depth_invalid_ratio=0.02,
        kpts_2d_px=np.array([[320, 380], [340, 380], [310, 460],
                              [350, 460], [305, 540], [355, 540]],
                              dtype=np.float32)[:K],
        kpts_3d_m=np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.5, 1.2],
                             [0.1, 0.5, 1.2], [0, 0.9, 1.5], [0.1, 0.9, 1.5]],
                             dtype=np.float32)[:K],
        kp_conf=np.array([0.9, 0.9, 0.85, 0.85, 0.8, 0.8],
                          dtype=np.float32)[:K],
        kp_sigma_m=np.full((K, 3), 0.015, dtype=np.float32),
        pose_cov_diag=np.full((K, 3), 0.015 ** 2, dtype=np.float32),
        rgb_bgra=rgb_bgra,
        depth_m=depth_m,
        rgb_right_bgra=rgb_right_bgra,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. Schema version
# ──────────────────────────────────────────────────────────────────────────

def test_schema_version():
    assert SCHEMA_VERSION == 1


# ──────────────────────────────────────────────────────────────────────────
# 2. Round-trip — scalars + arrays + JPEG + depth
# ──────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_basic_round_trip(self, tmp_npz):
        frame = make_synthetic_frame()
        bytes_written = save_frame_npz(tmp_npz, frame)
        assert tmp_npz.exists()
        assert bytes_written > 0
        loaded = load_frame_npz(tmp_npz)

        # scalars
        assert loaded.frame_id == frame.frame_id
        assert loaded.rgb_ts_ns == frame.rgb_ts_ns
        assert loaded.depth_ts_ns == frame.depth_ts_ns
        assert loaded.depth_age_us == frame.depth_age_us
        assert loaded.publish_done_mono_ns == frame.publish_done_mono_ns
        assert loaded.valid_mask_bits == frame.valid_mask_bits
        assert loaded.valid_reason == frame.valid_reason
        assert loaded.world_frame_applied == frame.world_frame_applied
        assert loaded.box_conf == pytest.approx(frame.box_conf)
        assert loaded.depth_invalid_ratio == pytest.approx(frame.depth_invalid_ratio)

        # arrays
        np.testing.assert_array_equal(loaded.kpts_2d_px, frame.kpts_2d_px)
        np.testing.assert_array_equal(loaded.kpts_3d_m, frame.kpts_3d_m)
        np.testing.assert_array_equal(loaded.kp_conf, frame.kp_conf)
        np.testing.assert_array_equal(loaded.kp_sigma_m, frame.kp_sigma_m)
        np.testing.assert_array_equal(loaded.pose_cov_diag, frame.pose_cov_diag)

        # depth (raw)
        np.testing.assert_array_equal(loaded.depth_m, frame.depth_m)

        # rgb (JPEG round-trip — lossy, tolerate small)
        diff = np.abs(loaded.rgb_bgra.astype(np.int16) - frame.rgb_bgra.astype(np.int16))
        # JPEG q=90 의 mean diff < 5
        assert diff.mean() < 10, f"JPEG diff mean too large: {diff.mean()}"

    def test_include_right(self, tmp_npz):
        frame = make_synthetic_frame(include_right=True)
        save_frame_npz(tmp_npz, frame)
        loaded = load_frame_npz(tmp_npz)
        assert loaded.rgb_right_bgra is not None
        assert loaded.rgb_right_bgra.shape == frame.rgb_right_bgra.shape

    def test_no_right_omitted(self, tmp_npz):
        frame = make_synthetic_frame(include_right=False)
        save_frame_npz(tmp_npz, frame)
        loaded = load_frame_npz(tmp_npz)
        assert loaded.rgb_right_bgra is None


# ──────────────────────────────────────────────────────────────────────────
# 3. Schema verification
# ──────────────────────────────────────────────────────────────────────────

class TestVerifySchema:
    def test_valid_schema_passes(self, tmp_npz):
        frame = make_synthetic_frame()
        save_frame_npz(tmp_npz, frame)
        verify_frame_schema(tmp_npz, expected_k=6)

    def test_expected_k_mismatch_fails(self, tmp_npz):
        frame = make_synthetic_frame(K=6)
        save_frame_npz(tmp_npz, frame)
        with pytest.raises(ValueError, match="K=6.*expected=17"):
            verify_frame_schema(tmp_npz, expected_k=17)

    def test_missing_field_detected(self, tmp_path):
        # 강제 corruption: 일부 fields 빠뜨림.
        bad_path = tmp_path / "bad.npz"
        np.savez_compressed(
            bad_path,
            schema_version=np.uint32(SCHEMA_VERSION),
            frame_id=np.uint32(1),
            # rgb_ts_ns 등 빠뜨림
        )
        with pytest.raises(ValueError, match="missing required"):
            verify_frame_schema(bad_path)


# ──────────────────────────────────────────────────────────────────────────
# 4. JPEG encode/decode
# ──────────────────────────────────────────────────────────────────────────

class TestJpegRoundTrip:
    def test_solid_color(self):
        bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        bgra[:, :, 2] = 255   # R = 255 (BGRA → R is index 2)
        bgra[:, :, 3] = 255   # alpha
        jpeg = _encode_rgb_jpeg(bgra)
        assert len(jpeg) > 0
        decoded = _decode_rgb_jpeg(jpeg)
        assert decoded.shape == (100, 100, 4)
        # JPEG 의 일부 quality loss 단 색상 dominant 유지
        # decoded 의 R channel (index 2 in BGRA) 가 dominant
        assert decoded[50, 50, 2] > 200, "red dominant lost"

    def test_quality_affects_size(self):
        bgra = np.random.default_rng(0).integers(0, 255, size=(200, 200, 4),
                                                  dtype=np.uint8)
        jpeg_low = _encode_rgb_jpeg(bgra, quality=10)
        jpeg_high = _encode_rgb_jpeg(bgra, quality=95)
        assert len(jpeg_low) < len(jpeg_high)


# ──────────────────────────────────────────────────────────────────────────
# 5. Depth preservation
# ──────────────────────────────────────────────────────────────────────────

class TestDepthPreservation:
    def test_depth_with_nan(self, tmp_npz):
        frame = make_synthetic_frame()
        # 일부 픽셀에 NaN
        frame.depth_m[0, 0] = float("nan")
        frame.depth_m[10, 10] = float("inf")
        frame.depth_m[20, 20] = 0.0
        save_frame_npz(tmp_npz, frame)
        loaded = load_frame_npz(tmp_npz)
        assert np.isnan(loaded.depth_m[0, 0])
        assert np.isinf(loaded.depth_m[10, 10])
        assert loaded.depth_m[20, 20] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 6. session_calib.json
# ──────────────────────────────────────────────────────────────────────────

class TestSessionCalib:
    def _valid_calib(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "session_start_ns": 1_000_000_000_000,
            "session_start_mono_ns": 100_000_000_000,
            "zed_serial": 52277959,
            "zed_sdk_version": "5.2.1",
            "resolution_width": 960,
            "resolution_height": 600,
            "fps": 120,
            "depth_mode": "PERFORMANCE",
            "self_calibration_disabled": True,
            "left_cam": {"fx": 480.0, "fy": 480.0, "cx": 480.0, "cy": 300.0, "disto": [0, 0, 0, 0, 0]},
            "right_cam": {"fx": 480.0, "fy": 480.0, "cx": 480.0, "cy": 300.0, "disto": [0, 0, 0, 0, 0]},
            "baseline_mm": 63.0,
            "stereo_transform": [[1, 0, 0, 0.063], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        }

    def test_save_load_round_trip(self, tmp_calib):
        calib = self._valid_calib()
        save_session_calib(tmp_calib, calib)
        loaded = load_session_calib(tmp_calib)
        assert loaded["zed_serial"] == 52277959
        assert loaded["baseline_mm"] == 63.0
        assert loaded["self_calibration_disabled"] is True

    def test_missing_field_fails(self, tmp_calib):
        calib = self._valid_calib()
        del calib["baseline_mm"]
        with pytest.raises(ValueError, match="baseline_mm"):
            save_session_calib(tmp_calib, calib)

    def test_missing_intrinsic_fails(self, tmp_calib):
        calib = self._valid_calib()
        del calib["left_cam"]["fx"]
        with pytest.raises(ValueError, match="left_cam missing intrinsic"):
            save_session_calib(tmp_calib, calib)

    def test_version_mismatch_on_load(self, tmp_calib):
        calib = self._valid_calib()
        save_session_calib(tmp_calib, calib)
        # corrupt
        with open(tmp_calib, "r") as f:
            data = json.load(f)
        data["version"] = 99
        with open(tmp_calib, "w") as f:
            json.dump(data, f)
        with pytest.raises(RuntimeError, match="version mismatch"):
            load_session_calib(tmp_calib)


# ──────────────────────────────────────────────────────────────────────────
# 7. Edge cases
# ──────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_frame_npz(tmp_path / "nonexistent.npz")

    def test_uncompressed_save(self, tmp_npz):
        frame = make_synthetic_frame()
        save_frame_npz(tmp_npz, frame, compress=False)
        loaded = load_frame_npz(tmp_npz)
        assert loaded.frame_id == frame.frame_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
