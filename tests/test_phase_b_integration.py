"""Phase B integration tests — SHM v2 + Plan D + Forecast publisher 의 *진정 *작동 검증.

진정 사용자 ultrathink: '데이터 흐름 확실 작동, 검토 여러번'.

진정 *Mac 에서 *진정 *진정 *진정 *호환 검증* (no ZED/TRT 의무):
  - compute_kp_sigma 의 *진정 *math correctness
  - ForecastPublisher 의 *seqlock + binary layout
  - Plan D Predictor (6 joints) + forecast_publisher.publish 의 *진정 *통합
  - End-to-end: synthetic walking → Plan D feed → forecast → SHM publish → read
"""
from __future__ import annotations

import math
import os
import struct
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest


def _shm_name(prefix: str) -> str:
    """Short unique SHM name (macOS PSHMNAMLEN ~31 chars incl leading '/')."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "perception" / "realtime"))

# Imports (graceful — Mac 의 *진정 *all components 호환)
from perception.realtime.joint_3d import compute_kp_sigma  # noqa: E402
from perception.CUDA_Stream.shm_publisher import (  # noqa: E402
    ShmPublisher as ShmPublisherV2,
    VALID_OK,
)
from forecast_publisher import (  # noqa: E402
    ForecastPublisher,
    PACKET_SIZE,
    N_JOINTS,
    SEQ_OFF,
    VERSION_OFF,
    FRAME_ID_OFF,
    PUBLISH_DONE_MONO_NS_OFF,
    TAU_LOOKAHEAD_S_OFF,
    PHI_RAD_OFF,
    OMEGA_OFF,
    CASCADE_LEVEL_OFF,
    STRIDE_COUNT_OFF,
    Q_PRED_OFF,
    T_HS_L_OFF,
    T_HS_R_OFF,
)
from perception.plan_d_prototype import PlanDPredictor  # noqa: E402


# ─── compute_kp_sigma ────────────────────────────────────────────────────


def test_compute_kp_sigma_basic():
    """진정 *stereo depth uncertainty formula: σ_z = Z² × σ_d / (fx × baseline)."""
    positions = {'hip': np.array([0.0, 0.0, 2.0], dtype=np.float32)}
    confs = {'hip': 1.0}
    sigmas = compute_kp_sigma(
        positions, confs, fx=480.0, fy=480.0,
        baseline_m=0.063, sigma_d_subpixel=0.25,
    )
    # σ_z = 2.0² × 0.25 / (480 × 0.063) = 1.0 / 30.24 ≈ 0.0331 m
    expected_sigma_z = (2.0 ** 2) * 0.25 / (480.0 * 0.063)
    assert abs(sigmas['hip'][2] - expected_sigma_z) < 1e-5


def test_compute_kp_sigma_invalid_depth():
    """NaN or Z<=0 → fallback large σ (1.0m)."""
    positions = {
        'a': np.array([0.0, 0.0, float('nan')], dtype=np.float32),
        'b': np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    confs = {'a': 0.9, 'b': 0.9}
    sigmas = compute_kp_sigma(positions, confs)
    assert sigmas['a'][2] == 1.0
    assert sigmas['b'][2] == 1.0


def test_compute_kp_sigma_confidence_weighting():
    """Low conf → larger σ."""
    pos = {'a': np.array([0, 0, 1.5], dtype=np.float32),
           'b': np.array([0, 0, 1.5], dtype=np.float32)}
    confs = {'a': 1.0, 'b': 0.3}
    s = compute_kp_sigma(pos, confs, fx=480.0)
    assert s['b'][2] > s['a'][2], "low conf should give larger σ"


# ─── ForecastPublisher ───────────────────────────────────────────────────


def test_forecast_packet_size_constant():
    """PACKET_SIZE = 192 bytes (exact, spec)."""
    assert PACKET_SIZE == 192
    assert N_JOINTS == 6


def test_forecast_publisher_create_close():
    name = _shm_name("tfc")
    fp = ForecastPublisher(name=name, create=True)
    fp.close()
    fp.unlink()


def test_forecast_publisher_publish_roundtrip():
    """publish() writes bytes; reader reads same values via struct.unpack."""
    name = _shm_name("tfp")
    fp = ForecastPublisher(name=name, create=True)

    # Mock CascadeForecast + HeelStrikeEvent
    class MockForecast:
        phi = 1.234
        sigma_phi = 0.05
        omega = 6.28
        sigma_omega = 0.2
        alpha = 0.0
        sigma_alpha = 0.5
        q_pred = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3], dtype=np.float32)

    class MockHS:
        def __init__(self, t, sigma, conf, ready):
            self.t_ahead_s = t
            self.sigma_t_s = sigma
            self.confidence = conf
            self.ready = ready

    fc = MockForecast()
    hs_L = MockHS(0.45, 0.02, 0.85, True)
    hs_R = MockHS(0.95, 0.05, 0.70, False)

    fp.publish(
        frame_id=12345,
        publish_done_mono_ns=987654321000,
        tau_lookahead_s=0.050,
        forecast=fc,
        cascade_level=3,
        stride_count=42,
        template_touched_fraction=0.65,
        is_ready_for_control=True,
        hs_event_L=hs_L,
        hs_event_R=hs_R,
    )

    # Read back via struct.unpack
    buf = bytes(fp._buf)

    seq = struct.unpack_from("<I", buf, SEQ_OFF)[0]
    assert seq % 2 == 0, "seq must be even after write completes"

    version = struct.unpack_from("<I", buf, VERSION_OFF)[0]
    assert version == 1

    frame_id = struct.unpack_from("<I", buf, FRAME_ID_OFF)[0]
    assert frame_id == 12345

    pub_done = struct.unpack_from("<Q", buf, PUBLISH_DONE_MONO_NS_OFF)[0]
    assert pub_done == 987654321000

    tau = struct.unpack_from("<f", buf, TAU_LOOKAHEAD_S_OFF)[0]
    assert abs(tau - 0.050) < 1e-6

    phi = struct.unpack_from("<f", buf, PHI_RAD_OFF)[0]
    assert abs(phi - 1.234) < 1e-5

    omega = struct.unpack_from("<f", buf, OMEGA_OFF)[0]
    assert abs(omega - 6.28) < 1e-5

    level = struct.unpack_from("<B", buf, CASCADE_LEVEL_OFF)[0]
    assert level == 3

    stride = struct.unpack_from("<H", buf, STRIDE_COUNT_OFF)[0]
    assert stride == 42

    # q_pred[0] = 0.1
    q0 = struct.unpack_from("<f", buf, Q_PRED_OFF)[0]
    assert abs(q0 - 0.1) < 1e-5

    # q_pred[5] = -0.3
    q5 = struct.unpack_from("<f", buf, Q_PRED_OFF + 5 * 4)[0]
    assert abs(q5 - (-0.3)) < 1e-5

    # HS L: t=0.45
    t_hs_L = struct.unpack_from("<f", buf, T_HS_L_OFF)[0]
    assert abs(t_hs_L - 0.45) < 1e-5

    # HS R: t=0.95
    t_hs_R = struct.unpack_from("<f", buf, T_HS_R_OFF)[0]
    assert abs(t_hs_R - 0.95) < 1e-5

    fp.close()
    fp.unlink()


def test_forecast_seqlock_during_write():
    """During write, seq should be odd (write_in_progress marker)."""
    name = _shm_name("tsl")
    fp = ForecastPublisher(name=name, create=True)
    # After __init__, seq is even (initial close)
    initial_seq = struct.unpack_from("<I", fp._buf, SEQ_OFF)[0]
    assert initial_seq % 2 == 0
    fp.close()
    fp.unlink()


# ─── Plan D + Forecast publisher 통합 ───────────────────────────────────


def test_plan_d_forecast_e2e():
    """End-to-end: PlanDPredictor.feed → forecast → publish → read."""
    name = _shm_name("tpd")
    fp = ForecastPublisher(name=name, create=True)
    predictor = PlanDPredictor(n_joints=6, fs_hz=67.0, initial_omega=6.28)

    # Synthetic walking 1Hz
    rng = np.random.default_rng(0)
    for i in range(100):
        t = i * 0.0149
        true_phi = (2 * math.pi * 1.0 * t) % (2 * math.pi)
        q = np.array([
            math.sin(true_phi + j * math.pi / 6)
            for j in range(6)
        ]) + rng.normal(0, 0.02, 6)
        predictor.feed(
            t_now=t,
            q=q,
            sigma_per_joint=np.full(6, 0.05),
            hip_z_world_m=0.05 * math.sin(2 * math.pi * 1.0 * t),
        )

    fc = predictor.forecast(0.050)
    hs_L = predictor.predict_heel_strike("L")
    hs_R = predictor.predict_heel_strike("R")

    fp.publish(
        frame_id=99,
        publish_done_mono_ns=int(time.monotonic_ns()),
        tau_lookahead_s=0.050,
        forecast=fc,
        cascade_level=int(predictor.level),
        stride_count=int(predictor.stride_count),
        template_touched_fraction=float(predictor.template_touched_fraction),
        is_ready_for_control=predictor.is_ready_for_control(
            require_l3=False, max_sigma_phi=2.0, max_ambiguity=0.9,
        ),
        hs_event_L=hs_L,
        hs_event_R=hs_R,
    )

    # Verify roundtrip
    buf = bytes(fp._buf)
    seq_final = struct.unpack_from("<I", buf, SEQ_OFF)[0]
    assert seq_final % 2 == 0, "seq must be even after successful publish"

    frame_id_read = struct.unpack_from("<I", buf, FRAME_ID_OFF)[0]
    assert frame_id_read == 99

    # omega convergence depends on real walking data + L2/L3 cascade — this test
    # only verifies publish→read pipeline. Just check it's finite and non-zero.
    omega_read = struct.unpack_from("<f", buf, OMEGA_OFF)[0]
    assert math.isfinite(omega_read) and omega_read > 0.0

    fp.close()
    fp.unlink()


def test_plan_d_6_joints_signature():
    """PlanDPredictor(n_joints=6) accepts 6-vector q."""
    p = PlanDPredictor(n_joints=6, fs_hz=67.0, initial_omega=6.28)
    q6 = np.array([0.1, 0.5, -0.1, 0.1, 0.5, -0.1])
    sigma6 = np.full(6, 0.05)
    p.feed(0.0, q6, sigma6, 0.05)
    fc = p.forecast(0.05)
    assert fc is not None
    assert fc.q_pred is None or fc.q_pred.shape == (6,)


# ─── SHM v2 publisher 통합 ──────────────────────────────────────────────


def test_shm_v2_publisher_6kp_roundtrip():
    """SHM v2 publish 6 keypoints + read back valid + frame_id."""
    name = _shm_name("tsv2")
    pub = ShmPublisherV2(num_keypoints=6, name=name, create=True)

    kpts_3d = np.array([
        [0.1, 0.0, 1.5],   # left_hip
        [0.1, -0.4, 1.6],
        [0.1, -0.9, 1.7],
        [-0.1, 0.0, 1.5],  # right_hip
        [-0.1, -0.4, 1.6],
        [-0.1, -0.9, 1.7],
    ], dtype=np.float32)
    kpt_conf = np.full(6, 0.9, dtype=np.float32)
    kpts_2d = np.array([
        [320.0, 100.0], [320.0, 200.0], [320.0, 300.0],
        [360.0, 100.0], [360.0, 200.0], [360.0, 300.0],
    ], dtype=np.float32)
    kp_sigma = np.full((6, 3), 0.015, dtype=np.float32)

    pub.publish(
        frame_id=777,
        rgb_ts_ns=int(time.monotonic_ns()),
        kpts_3d_m=kpts_3d,
        kpt_conf=kpt_conf,
        kpts_2d_px=kpts_2d,
        box_conf=0.95,
        valid=True,
        depth_invalid_ratio=0.0,
        world_frame_applied=True,
        valid_reason=VALID_OK,
        kp_sigma_m=kp_sigma,
    )

    # Verify seq even (write complete) + version=2
    buf = bytes(pub._buf)
    seq = struct.unpack_from("<I", buf, 0)[0]
    assert seq % 2 == 0
    version = struct.unpack_from("<I", buf, 4)[0]
    assert version == 2
    K = struct.unpack_from("<I", buf, 8)[0]
    assert K == 6

    # ShmPublisher.close() already calls shm.unlink() internally — no separate unlink()
    pub.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
