"""SHM v2 timestamp monotonic + frame-to-frame consistency tests.

Codex orchestration bvfvkxo1m (2026-05-11) Q2c.

검증:
    1. publish_done_mono_ns 가 monotonic (sequence 마다 증가)
    2. rgb_ts_ns frame-to-frame difference (caller 의 책임 단 round-trip 검증)
    3. depth_age_us 가 (rgb_ts - depth_ts) / 1000 정확
    4. multiple publishes 의 publish_done_mono 가 strictly increasing

실행:
    python3 -m pytest tests/test_timestamp_monotonic.py -v
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from perception.CUDA_Stream.shm_publisher import (
    ShmPublisher, ShmReader, VALID_OK,
)


@pytest.fixture
def shm_name(request):
    """Per-test SHM name (격리)."""
    safe = request.node.name.replace("[", "_").replace("]", "_")[:48]
    name = f"tts_{os.getpid()}_{safe}"
    yield name
    try:
        os.remove(f"/dev/shm/{name}")
    except FileNotFoundError:
        pass


@pytest.fixture
def sample_data():
    K = 6
    return {
        "K": K,
        "kpts_3d": np.zeros((K, 3), dtype=np.float32),
        "kpts_2d": np.zeros((K, 2), dtype=np.float32),
        "kp_conf": np.full((K,), 0.9, dtype=np.float32),
    }


class TestPublishDoneMonotonic:

    def test_publish_done_monotonic_strict(self, shm_name, sample_data):
        """100 publishes → publish_done_mono_ns 가 strictly increasing."""
        K = sample_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        reader = ShmReader(name=shm_name, expected_k=K)
        try:
            publish_dones = []
            for i in range(100):
                pub.publish(
                    frame_id=i,
                    rgb_ts_ns=1_000_000_000 + i * 8_333_333,
                    kpts_3d_m=sample_data["kpts_3d"],
                    kpt_conf=sample_data["kp_conf"],
                    kpts_2d_px=sample_data["kpts_2d"],
                    box_conf=0.9, valid=True,
                )
                # read immediately to capture publish_done
                result = reader.read()
                assert result is not None
                publish_done = result[13]   # publish_done_mono_ns at index 13
                publish_dones.append(publish_done)
                # tiny delay so monotonic_ns advances
                time.sleep(0.0001)

            # strictly monotonic increasing
            for i in range(1, len(publish_dones)):
                assert publish_dones[i] > publish_dones[i-1], (
                    f"publish_done not monotonic at i={i}: "
                    f"{publish_dones[i-1]} → {publish_dones[i]}"
                )
        finally:
            pub.close()
            reader.close()


class TestDepthAgeConsistency:

    def test_depth_age_zero_same_frame(self, shm_name, sample_data):
        """depth_ts_ns == rgb_ts_ns → depth_age_us == 0."""
        K = sample_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        reader = ShmReader(name=shm_name, expected_k=K)
        try:
            rgb_ts = 1_000_000_000
            pub.publish(
                frame_id=1, rgb_ts_ns=rgb_ts,
                kpts_3d_m=sample_data["kpts_3d"],
                kpt_conf=sample_data["kp_conf"],
                kpts_2d_px=sample_data["kpts_2d"],
                box_conf=0.9, valid=True,
                depth_ts_ns=rgb_ts,    # same frame
            )
            result = reader.read()
            depth_age_us = result[3]
            assert depth_age_us == 0
        finally:
            pub.close()
            reader.close()

    def test_depth_age_one_frame_late(self, shm_name, sample_data):
        """depth_ts = rgb_ts - 8.33ms → depth_age_us ≈ 8333."""
        K = sample_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        reader = ShmReader(name=shm_name, expected_k=K)
        try:
            rgb_ts = 1_000_000_000
            depth_ts = rgb_ts - 8_333_333
            pub.publish(
                frame_id=1, rgb_ts_ns=rgb_ts,
                kpts_3d_m=sample_data["kpts_3d"],
                kpt_conf=sample_data["kp_conf"],
                kpts_2d_px=sample_data["kpts_2d"],
                box_conf=0.9, valid=True,
                depth_ts_ns=depth_ts,
            )
            result = reader.read()
            depth_age_us = result[3]
            assert 8000 <= depth_age_us <= 8500, (
                f"depth_age_us {depth_age_us} not in [8000, 8500]"
            )
        finally:
            pub.close()
            reader.close()

    def test_depth_age_large_stale(self, shm_name, sample_data):
        """depth_ts = rgb_ts - 100ms → depth_age_us == 100000."""
        K = sample_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        reader = ShmReader(name=shm_name, expected_k=K)
        try:
            rgb_ts = 1_000_000_000
            depth_ts = rgb_ts - 100_000_000     # 100ms stale
            pub.publish(
                frame_id=1, rgb_ts_ns=rgb_ts,
                kpts_3d_m=sample_data["kpts_3d"],
                kpt_conf=sample_data["kp_conf"],
                kpts_2d_px=sample_data["kpts_2d"],
                box_conf=0.9, valid=True,
                depth_ts_ns=depth_ts,
            )
            result = reader.read()
            depth_age_us = result[3]
            assert depth_age_us == 100_000
        finally:
            pub.close()
            reader.close()


class TestRgbTimestamp:

    def test_rgb_ts_round_trip(self, shm_name, sample_data):
        """frame N 의 rgb_ts_ns 가 정확 round-trip."""
        K = sample_data["K"]
        pub = ShmPublisher(K, name=shm_name, create=True)
        reader = ShmReader(name=shm_name, expected_k=K)
        try:
            for i in range(20):
                rgb_ts = 1_000_000_000 + i * 8_333_333
                pub.publish(
                    frame_id=i, rgb_ts_ns=rgb_ts,
                    kpts_3d_m=sample_data["kpts_3d"],
                    kpt_conf=sample_data["kp_conf"],
                    kpts_2d_px=sample_data["kpts_2d"],
                    box_conf=0.9, valid=True,
                )
                result = reader.read()
                _, rgb_ts_r, _, _, *_ = result
                assert rgb_ts_r == rgb_ts
        finally:
            pub.close()
            reader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
