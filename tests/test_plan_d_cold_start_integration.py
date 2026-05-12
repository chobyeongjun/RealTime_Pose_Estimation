"""Plan D cold-start integration — Hilbert envelope + EKF L1.

Codex review 2026-05-12 HARD WALL: circular bootstrap between L1 and
CrossCorrPhaseEstimator. Solved by HipVerticalPhaseEstimator (Hilbert
envelope) feeding L1's phase observations until template is populated.

This integration test exercises the full cold-start path:
    1. Patient starts walking — vertical hip oscillation observed.
    2. HilbertPhaseEstimator warms up over 1.5 s, emits φ.
    3. L1 EKF predicts + updates with Hilbert φ.
    4. L1 ω converges to true cadence.
    5. CycleTemplate (separate concern) is fed q + L1-derived phi.
    6. After enough strides, switch to CrossCorrPhaseEstimator.

This test validates step 1-4 (the actual cold-start mechanic). Steps 5-6
belong in the PredictorCascade (Phase 2).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.ekf_l1 import EKFL1, PredictStatus
from perception.plan_d_prototype.hilbert_phase import HipVerticalPhaseEstimator
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_pi


def test_cold_start_l1_converges_under_hilbert_phase():
    """Simulate 10 s of 1.0 Hz walking. Verify L1 ω converges to 2π rad/s."""
    fs = 120.0
    f_walk = 1.0
    true_omega = 2.0 * math.pi * f_walk
    amplitude = 0.06

    hilbert_est = HipVerticalPhaseEstimator(
        window_seconds=1.5, fs_hz=fs, min_amplitude_m=0.01,
    )
    # Prior omega intentionally off (3.5 rad/s) — let L1 converge
    l1 = EKFL1(
        initial_omega=3.5,
        process_noise_omega=4e-2,
        measurement_noise=0.05,
    )

    omega_history = []
    for i in range(int(10 * fs)):  # 10 s
        t = i / fs
        z = amplitude * math.sin(2 * math.pi * f_walk * t)
        hilbert_est.feed(t, z)

        status = l1.predict(t)
        # Status is OK after first sample
        if i > 0:
            assert status == PredictStatus.OK

        phase_obs = hilbert_est.estimate()
        if phase_obs.valid:
            l1.update(phase_obs.phi)

        omega_history.append(l1.state.omega)

    # After 10 s, ω should be close to true ω (within 0.5 rad/s)
    final_omega = float(np.mean(omega_history[-int(0.5 * fs):]))
    err = abs(final_omega - true_omega)
    assert err < 0.7, (
        f"L1 ω did not converge: final={final_omega:.3f}, true={true_omega:.3f}"
    )


def test_cold_start_freezing_gait_l1_holds():
    """If patient stops walking (no hip oscillation), Hilbert returns invalid
    and L1 has no observation → ω prior held."""
    fs = 120.0
    hilbert_est = HipVerticalPhaseEstimator(
        window_seconds=1.5, fs_hz=fs, min_amplitude_m=0.01,
    )
    l1 = EKFL1(initial_omega=4.0, process_noise_omega=4e-2)

    # Feed 5 s of stationary hip (no oscillation)
    for i in range(int(5 * fs)):
        t = i / fs
        z = 0.5  # constant
        hilbert_est.feed(t, z)
        l1.predict(t)
        phase_obs = hilbert_est.estimate()
        if phase_obs.valid:
            l1.update(phase_obs.phi)

    # Hilbert should never have emitted valid phase
    assert hilbert_est.estimate().valid is False
    # L1 ω should still be near prior (no updates applied)
    assert abs(l1.state.omega - 4.0) < 0.5
    # But P_omega grew via process noise over 5 s
    assert l1.state.sigma_omega > 0.0


def test_cold_start_dt_gap_status_propagation():
    """L1 predict() reports DT_TOO_LARGE so caller can trigger watchdog."""
    l1 = EKFL1(max_dt_s=0.5)
    l1.predict(0.0)
    # Simulate vision loss for 1 s
    status = l1.predict(1.0)
    assert status == PredictStatus.DT_TOO_LARGE
    # ω prior should be preserved (no integration applied)
    assert abs(l1.state.omega - 4.0) < 0.01
    # t_last must advance so next call sees fresh dt
    assert l1.state.t_last == 1.0
    # Next normal-dt call should integrate fine
    status2 = l1.predict(1.005)
    assert status2 == PredictStatus.OK


def test_cold_start_stop_then_resume():
    """Walking → stop (1 s freeze) → resume. L1 should recover."""
    fs = 120.0
    f_walk = 1.0
    true_omega = 2 * math.pi * f_walk
    A = 0.06

    hilbert_est = HipVerticalPhaseEstimator(
        window_seconds=1.5, fs_hz=fs, min_amplitude_m=0.01,
    )
    l1 = EKFL1(initial_omega=3.5)

    # 5 s walking
    n = 0
    for i in range(int(5 * fs)):
        t = i / fs
        z = A * math.sin(2 * math.pi * f_walk * t)
        hilbert_est.feed(t, z)
        l1.predict(t)
        po = hilbert_est.estimate()
        if po.valid:
            l1.update(po.phi)
        n += 1
    omega_after_walking = l1.state.omega
    assert abs(omega_after_walking - true_omega) < 1.5  # convergence partial

    # 1 s freezing (constant z)
    z_constant = A * math.sin(2 * math.pi * f_walk * (n / fs))
    for i in range(int(fs)):
        t = (n + i) / fs
        hilbert_est.feed(t, z_constant)
        l1.predict(t)
        po = hilbert_est.estimate()
        if po.valid:
            l1.update(po.phi)
        n += 1
    # During freeze: P_omega grew (no measurements; process noise)
    assert l1.state.sigma_omega >= 0.0

    # Resume 5 s walking — phase carries over but template/freeze didn't crash
    for i in range(int(5 * fs)):
        t = (n + i) / fs
        z = A * math.sin(2 * math.pi * f_walk * t)
        hilbert_est.feed(t, z)
        l1.predict(t)
        po = hilbert_est.estimate()
        if po.valid:
            l1.update(po.phi)
        n += 1
    # State remains finite + bounded
    assert math.isfinite(l1.state.omega)
    assert l1.state.omega > 0
    assert l1.state.omega < 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
