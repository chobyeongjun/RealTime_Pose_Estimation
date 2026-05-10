"""SHM v2 packet schema 의 binary layout 검증 (TDD).

Codex orchestration bvfvkxo1m (2026-05-11) Q2.

검증:
    1. Total size = 64 + K * 48 (aligned to 64B)
    2. Header offsets 정확
    3. version == 2
    4. publish + read round-trip (모든 fields 정확)
    5. seq even/odd protocol (seqlock)
    6. Two-timestamp 의 depth_age_us 자동 계산
    7. valid_mask_bits 의 default derivation
    8. kp_sigma_m / pose_cov_diag default
    9. v1 reader vs v2 packet → version mismatch RuntimeError

실행:
    pytest tests/test_plan_d_packet_schema.py -v
또는:
    python3 -m pytest tests/test_plan_d_packet_schema.py -v
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest


# Test 자체가 ZED 의존 없음 — Mac 에서도 실행 가능 (단 CUDA / pyzed import X).
# shm_publisher 만 import.
from perception.CUDA_Stream.shm_publisher import (
    ShmPublisher, ShmReader,
    VERSION, HEADER_SIZE, compute_size,
    SEQ_OFF, VERSION_OFF, K_OFF, FRAME_ID_OFF,
    RGB_TS_OFF, DEPTH_TS_OFF, DEPTH_AGE_OFF,
    BOX_CONF_OFF, DEPTH_INVALID_OFF,
    VALID_FLAG_OFF, WORLD_FRAME_OFF, VALID_REASON_OFF, TS_DOMAIN_OFF,
    PUBLISH_DONE_OFF, VALID_MASK_BITS_OFF,
    VALID_OK, INVALID_NO_DETECTION, INVALID_STALE_DEPTH,
    DEFAULT_SIGMA_M, DEFAULT_NAME,
)


@pytest.fixture
def shm_name(request):
    """unique SHM name per test (cleanup safe).

    Per-test 격리: PID + test name 으로 충돌 회피.
    """
    safe = request.node.name.replace("[", "_").replace("]", "_")[:48]
    name = f"tv2_{os.getpid()}_{safe}"
    yield name
    # cleanup — best effort (ShmPublisher.close() 가 이미 unlink)
    try:
        os.remove(f"/dev/shm/{name}")
    except FileNotFoundError:
        pass


@pytest.fixture
def kpts_data():
    """K=6 의 sample data."""
    K = 6
    return {
        "K": K,
        "kpts_3d": np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.5, 1.2],
            [0.1, 0.5, 1.2],
            [0.0, 0.9, 1.5],
            [0.1, 0.9, 1.5],
        ], dtype=np.float32),
        "kpts_2d": np.array([
            [320, 380], [340, 380],
            [310, 460], [350, 460],
            [305, 540], [355, 540],
        ], dtype=np.float32),
        "kp_conf": np.array([0.9, 0.9, 0.85, 0.85, 0.8, 0.8], dtype=np.float32),
    }


# ----------------------------------------------------------------------
# 1. Size + offsets
# ----------------------------------------------------------------------

class TestSize:

    def test_compute_size_K6(self):
        """K=6 의 size = 64 + 6*48 = 352, aligned 384."""
        assert compute_size(6) == 384

    def test_compute_size_K17(self):
        """K=17 (COCO) → 64 + 17*48 = 880, aligned 896."""
        assert compute_size(17) == 896

    def test_compute_size_K1(self):
        assert compute_size(1) == 128   # 64 + 48 = 112 → 128

    def test_compute_size_K_invalid(self, shm_name):
        with pytest.raises(ValueError):
            ShmPublisher(0, name=shm_name)
        with pytest.raises(ValueError):
            ShmPublisher(65, name=shm_name)


class TestOffsets:

    def test_header_layout(self):
        """v2 header 의 정확 offsets."""
        assert SEQ_OFF == 0
        assert VERSION_OFF == 4
        assert K_OFF == 8
        assert FRAME_ID_OFF == 12
        assert RGB_TS_OFF == 16
        assert DEPTH_TS_OFF == 24
        assert DEPTH_AGE_OFF == 32
        assert BOX_CONF_OFF == 36
        assert DEPTH_INVALID_OFF == 40
        assert VALID_FLAG_OFF == 44
        assert WORLD_FRAME_OFF == 45
        assert VALID_REASON_OFF == 46
        assert TS_DOMAIN_OFF == 47
        assert PUBLISH_DONE_OFF == 48
        assert VALID_MASK_BITS_OFF == 56
        assert HEADER_SIZE == 64

    def test_version_value(self):
        assert VERSION == 2


# ----------------------------------------------------------------------
# 2. Round-trip (publisher + reader)
# ----------------------------------------------------------------------

class TestRoundTrip:

    def test_basic_publish_read(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            pub.publish(
                frame_id=42,
                rgb_ts_ns=1_000_000_000,        # 1.0s epoch
                kpts_3d_m=kpts_data["kpts_3d"],
                kpt_conf=kpts_data["kp_conf"],
                kpts_2d_px=kpts_data["kpts_2d"],
                box_conf=0.85,
                valid=True,
            )

            reader = ShmReader(name=shm_name, expected_k=K)
            try:
                result = reader.read()
                assert result is not None
                (frame_id, rgb_ts, depth_ts, depth_age_us,
                 kpts_3d, kp_conf, kpts_2d, kp_sigma, pose_cov,
                 box_conf, valid, depth_inv, world_frame,
                 publish_done, valid_reason, ts_domain,
                 valid_mask_bits) = result

                assert frame_id == 42
                assert rgb_ts == 1_000_000_000
                assert depth_ts == 1_000_000_000      # default same-frame
                assert depth_age_us == 0
                assert valid is True
                assert valid_reason == VALID_OK
                np.testing.assert_array_equal(kpts_3d, kpts_data["kpts_3d"])
                np.testing.assert_array_equal(kpts_2d, kpts_data["kpts_2d"])
                np.testing.assert_array_equal(kp_conf, kpts_data["kp_conf"])
                # default kp_sigma_m = uniform 15mm
                np.testing.assert_array_almost_equal(
                    kp_sigma, np.full((K, 3), DEFAULT_SIGMA_M, dtype=np.float32)
                )
                # default pose_cov_diag = kp_sigma_m²
                np.testing.assert_array_almost_equal(
                    pose_cov, np.full((K, 3), DEFAULT_SIGMA_M ** 2, dtype=np.float32)
                )
                # valid → all K bits set
                assert valid_mask_bits == (1 << K) - 1
            finally:
                reader.close()
        finally:
            pub.close()

    def test_one_frame_late_depth(self, shm_name, kpts_data):
        """one-frame-late path: depth_ts = rgb_ts - 8.3ms."""
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            rgb_ts = 1_000_000_000
            depth_ts = rgb_ts - 8_333_333    # ~8.33ms earlier (1 frame at 120fps)
            pub.publish(
                frame_id=42, rgb_ts_ns=rgb_ts,
                kpts_3d_m=kpts_data["kpts_3d"],
                kpt_conf=kpts_data["kp_conf"],
                kpts_2d_px=kpts_data["kpts_2d"],
                box_conf=0.9, valid=True,
                depth_ts_ns=depth_ts,
            )
            reader = ShmReader(name=shm_name, expected_k=K)
            try:
                result = reader.read()
                assert result is not None
                _, rgb_ts_r, depth_ts_r, depth_age_us, *_ = result
                assert rgb_ts_r == rgb_ts
                assert depth_ts_r == depth_ts
                # depth_age_us ≈ 8333 us
                assert 8000 <= depth_age_us <= 8500, \
                    f"depth_age_us {depth_age_us} not in [8000, 8500]"
            finally:
                reader.close()
        finally:
            pub.close()

    def test_invalid_publish(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            pub.publish(
                frame_id=42, rgb_ts_ns=1_000_000_000,
                kpts_3d_m=kpts_data["kpts_3d"],
                kpt_conf=kpts_data["kp_conf"],
                kpts_2d_px=kpts_data["kpts_2d"],
                box_conf=0.1, valid=False,
                valid_reason=INVALID_NO_DETECTION,
            )
            reader = ShmReader(name=shm_name, expected_k=K)
            try:
                result = reader.read()
                _, _, _, _, _, _, _, _, _, _, valid, _, _, _, valid_reason, _, valid_mask_bits = result
                assert valid is False
                assert valid_reason == INVALID_NO_DETECTION
                assert valid_mask_bits == 0
            finally:
                reader.close()
        finally:
            pub.close()

    def test_per_keypoint_covariance(self, shm_name, kpts_data):
        """custom kp_sigma_m / pose_cov_diag 가 정확 round-trip."""
        K = kpts_data["K"]
        # custom: 각 keypoint 의 sigma 다름 (hip 5mm, knee 10mm, ankle 25mm)
        sigma = np.array([
            [0.005, 0.005, 0.010],
            [0.005, 0.005, 0.010],
            [0.008, 0.008, 0.012],
            [0.008, 0.008, 0.012],
            [0.015, 0.015, 0.025],
            [0.015, 0.015, 0.025],
        ], dtype=np.float32)
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            pub.publish(
                frame_id=1, rgb_ts_ns=1_000_000_000,
                kpts_3d_m=kpts_data["kpts_3d"],
                kpt_conf=kpts_data["kp_conf"],
                kpts_2d_px=kpts_data["kpts_2d"],
                box_conf=0.9, valid=True,
                kp_sigma_m=sigma,
            )
            reader = ShmReader(name=shm_name, expected_k=K)
            try:
                result = reader.read()
                _, _, _, _, _, _, _, kp_sigma_r, pose_cov_r, *_ = result
                np.testing.assert_array_almost_equal(kp_sigma_r, sigma)
                # pose_cov_diag default = sigma²
                np.testing.assert_array_almost_equal(pose_cov_r, sigma ** 2)
            finally:
                reader.close()
        finally:
            pub.close()


# ----------------------------------------------------------------------
# 3. Seqlock
# ----------------------------------------------------------------------

class TestSeqlock:

    def test_seq_even_after_publish(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            pub.publish(
                frame_id=1, rgb_ts_ns=1_000_000_000,
                kpts_3d_m=kpts_data["kpts_3d"],
                kpt_conf=kpts_data["kp_conf"],
                kpts_2d_px=kpts_data["kpts_2d"],
                box_conf=0.9, valid=True,
            )
            seq = struct.unpack_from("<I", pub._buf, SEQ_OFF)[0]
            assert seq % 2 == 0, f"seq {seq} should be even after publish"
        finally:
            pub.close()

    def test_multiple_publishes_increment_seq(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            seqs = []
            for i in range(5):
                pub.publish(
                    frame_id=i, rgb_ts_ns=1_000_000_000 + i * 8_333_333,
                    kpts_3d_m=kpts_data["kpts_3d"],
                    kpt_conf=kpts_data["kp_conf"],
                    kpts_2d_px=kpts_data["kpts_2d"],
                    box_conf=0.9, valid=True,
                )
                seq = struct.unpack_from("<I", pub._buf, SEQ_OFF)[0]
                seqs.append(seq)
            # all seqs even, monotonically increasing
            for s in seqs:
                assert s % 2 == 0
            for i in range(1, len(seqs)):
                assert seqs[i] > seqs[i-1]
        finally:
            pub.close()


# ----------------------------------------------------------------------
# 4. Validation (input shape / dtype)
# ----------------------------------------------------------------------

class TestValidation:

    def test_wrong_kpts_3d_shape(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            wrong = np.zeros((K + 1, 3), dtype=np.float32)
            with pytest.raises(ValueError):
                pub.publish(
                    frame_id=1, rgb_ts_ns=1_000_000_000,
                    kpts_3d_m=wrong,
                    kpt_conf=kpts_data["kp_conf"],
                    kpts_2d_px=kpts_data["kpts_2d"],
                    box_conf=0.9, valid=True,
                )
        finally:
            pub.close()

    def test_wrong_dtype(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            wrong_dtype = kpts_data["kpts_3d"].astype(np.float64)
            with pytest.raises(ValueError):
                pub.publish(
                    frame_id=1, rgb_ts_ns=1_000_000_000,
                    kpts_3d_m=wrong_dtype,
                    kpt_conf=kpts_data["kp_conf"],
                    kpts_2d_px=kpts_data["kpts_2d"],
                    box_conf=0.9, valid=True,
                )
        finally:
            pub.close()

    def test_kp_sigma_wrong_shape(self, shm_name, kpts_data):
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            wrong_sigma = np.zeros((K, 2), dtype=np.float32)  # should be (K, 3)
            with pytest.raises(ValueError):
                pub.publish(
                    frame_id=1, rgb_ts_ns=1_000_000_000,
                    kpts_3d_m=kpts_data["kpts_3d"],
                    kpt_conf=kpts_data["kp_conf"],
                    kpts_2d_px=kpts_data["kpts_2d"],
                    box_conf=0.9, valid=True,
                    kp_sigma_m=wrong_sigma,
                )
        finally:
            pub.close()


# ----------------------------------------------------------------------
# 5. Reader version mismatch
# ----------------------------------------------------------------------

class TestVersionMismatch:

    def test_reader_rejects_wrong_version(self, shm_name, kpts_data):
        """v2 publisher → version=2. v1 packet (manually written version=1) → reader fail."""
        K = kpts_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        try:
            # publish 정상 (version=2)
            pub.publish(
                frame_id=1, rgb_ts_ns=1_000_000_000,
                kpts_3d_m=kpts_data["kpts_3d"],
                kpt_conf=kpts_data["kp_conf"],
                kpts_2d_px=kpts_data["kpts_2d"],
                box_conf=0.9, valid=True,
            )
            # 강제로 version=1 로 변경 (legacy publisher 시뮬)
            struct.pack_into("<I", pub._buf, VERSION_OFF, 1)
            with pytest.raises(RuntimeError, match="version mismatch"):
                ShmReader(name=shm_name, expected_k=K)
        finally:
            pub.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
