"""Plan D EKF L1 tests — const-velocity gait phase Kalman.

Coverage:
    - Initialization (state, covariance shapes, defaults)
    - Predict step (dt validation, F integration, Q discretization)
    - Update step (innovation phase wrap, Joseph PSD, NaN guard)
    - Convergence (steady-state tracking error, omega learning)
    - Forecast (predict_ahead consistency, variance growth)
    - Real-time safety (no allocation, condition number)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.ekf_l1 import (
    _DEFAULT_INITIAL_OMEGA,
    EKFL1,
    EKFL1State,
    PredictStatus,
)
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_pi


# ─── Initialization ──────────────────────────────────────────────────────


def test_l1_init_default_state():
    ekf = EKFL1()
    assert ekf.state.x.shape == (2,)
    assert ekf.state.P.shape == (2, 2)
    assert ekf.state.t_last is None
    assert ekf.state.phi == 0.0
    assert abs(ekf.state.omega - _DEFAULT_INITIAL_OMEGA) < 1e-12


def test_l1_init_custom_omega():
    ekf = EKFL1(initial_omega=5.0)
    assert abs(ekf.state.omega - 5.0) < 1e-12


def test_l1_init_P_psd():
    ekf = EKFL1()
    eigvals = np.linalg.eigvalsh(ekf.state.P)
    assert np.all(eigvals > 0)


def test_l1_init_dtype_float64():
    ekf = EKFL1()
    assert ekf.state.x.dtype == np.float64
    assert ekf.state.P.dtype == np.float64


def test_l1_is_initialized_after_first_predict():
    ekf = EKFL1()
    assert ekf.is_initialized is False
    ekf.predict(0.0)
    assert ekf.is_initialized is True


# ─── Predict step ────────────────────────────────────────────────────────


def test_l1_first_predict_no_state_change():
    """First predict() only stamps t_last; no integration."""
    ekf = EKFL1()
    x0 = ekf.state.x.copy()
    ekf.predict(100.0)
    assert np.allclose(ekf.state.x, x0)
    assert ekf.state.t_last == 100.0


def test_l1_predict_const_velocity_open_loop():
    """No updates; pure forward integration: phi += omega * t."""
    omega = 3.0
    ekf = EKFL1(initial_omega=omega)
    ekf.predict(0.0)
    # Integrate 1s in 0.01s steps
    for k in range(1, 101):
        ekf.predict(k * 0.01)
    # Expected phi = omega * 1.0 = 3.0 rad (within wrap)
    expected = omega * 1.0 % TWO_PI
    err = abs(float(wrap_to_pi(ekf.state.phi - expected)))
    assert err < 1e-6, f"Open-loop integration error: {err}"


def test_l1_predict_negative_dt_skipped():
    """Negative dt should not propagate state but updates t_last."""
    ekf = EKFL1()
    ekf.predict(0.1)
    x_before = ekf.state.x.copy()
    ekf.predict(0.05)  # dt = -0.05
    assert np.allclose(ekf.state.x, x_before)
    assert ekf.state.t_last == 0.05


def test_l1_predict_huge_dt_skipped():
    """dt > max_dt_s should not propagate state."""
    ekf = EKFL1(max_dt_s=0.5)
    ekf.predict(0.0)
    x_before = ekf.state.x.copy()
    ekf.predict(5.0)
    assert np.allclose(ekf.state.x, x_before)
    assert ekf.state.t_last == 5.0


def test_l1_predict_nan_inf_ignored():
    """NaN/inf t_now should not crash or corrupt state."""
    ekf = EKFL1()
    ekf.predict(0.0)
    x_before = ekf.state.x.copy()
    t_before = ekf.state.t_last
    ekf.predict(float("nan"))
    ekf.predict(float("inf"))
    assert np.allclose(ekf.state.x, x_before)
    assert ekf.state.t_last == t_before


def test_l1_predict_P_grows_without_updates():
    """Open-loop: variance should monotonically grow."""
    ekf = EKFL1()
    ekf.predict(0.0)
    P_phi_prev = ekf.state.P[0, 0]
    for k in range(1, 11):
        ekf.predict(k * 0.01)
        P_phi_now = ekf.state.P[0, 0]
        assert P_phi_now >= P_phi_prev - 1e-12, "P_phi should not decrease w/o update"
        P_phi_prev = P_phi_now


def test_l1_predict_keeps_P_symmetric():
    ekf = EKFL1()
    ekf.predict(0.0)
    for k in range(1, 50):
        ekf.predict(k * 0.01)
    assert np.allclose(ekf.state.P, ekf.state.P.T, atol=1e-12)


# ─── Update step ─────────────────────────────────────────────────────────


def test_l1_update_innovation_phase_wrap():
    """Observation just past 2π, state near 0 → small innovation (NOT 2π)."""
    ekf = EKFL1(measurement_noise=0.01)
    ekf.predict(0.0)
    # Force state to phi ≈ 0
    ekf.state.x[:] = [0.05, 4.0]
    z = TWO_PI - 0.05
    phi_before = ekf.state.phi
    ekf.update(z)
    # State should stay near 0 (wrapped), not jump to ~2π
    err = abs(float(wrap_to_pi(ekf.state.phi - phi_before)))
    assert err < 0.5, f"Innovation phase wrap broken: state jumped by {err}"


def test_l1_update_nan_observation_skip():
    """NaN observation should leave state unchanged."""
    ekf = EKFL1()
    ekf.predict(0.0)
    x_before = ekf.state.x.copy()
    P_before = ekf.state.P.copy()
    ekf.update(float("nan"))
    ekf.update(float("inf"))
    assert np.allclose(ekf.state.x, x_before)
    assert np.allclose(ekf.state.P, P_before)


def test_l1_update_keeps_P_psd():
    """Many random updates → P remains PSD."""
    rng = np.random.default_rng(123)
    ekf = EKFL1(measurement_noise=0.05)
    ekf.predict(0.0)
    for k in range(200):
        ekf.predict(k * 0.01)
        z = rng.uniform(0, TWO_PI)
        ekf.update(z)
    eigvals = np.linalg.eigvalsh(ekf.state.P)
    assert np.all(eigvals > -1e-9), f"P not PSD: min eigval {eigvals.min()}"


def test_l1_update_R_override():
    """R_override should be honored for that update."""
    ekf = EKFL1(measurement_noise=1.0)  # poor default
    ekf.predict(0.0)
    P_phi_before = ekf.state.P[0, 0]
    ekf.update(1.0, R_override=1e-6)  # very confident
    # Should pull phi strongly toward observation
    assert abs(ekf.state.phi - 1.0) < 0.1
    # P should shrink dramatically
    assert ekf.state.P[0, 0] < P_phi_before


# ─── Convergence ─────────────────────────────────────────────────────────


def test_l1_steady_state_phase_tracking():
    """True ω=3 rad/s, σ_z=0.05, expect steady-state phase error < 0.1 rad."""
    rng = np.random.default_rng(0)
    true_omega = 3.0
    ekf = EKFL1(initial_omega=true_omega, measurement_noise=0.05**2)
    dt = 0.01
    n = 1000  # 10s
    errors = []
    for k in range(n):
        t = k * dt
        true_phi = (true_omega * t) % TWO_PI
        ekf.predict(t)
        z = true_phi + rng.normal(0, 0.05)
        z = float(np.mod(z, TWO_PI))
        ekf.update(z)
        err = abs(float(wrap_to_pi(ekf.state.phi - true_phi)))
        errors.append(err)
    steady = np.mean(errors[-100:])
    assert steady < 0.1, f"Steady-state phase error too high: {steady:.4f}"


def test_l1_omega_convergence_from_wrong_prior():
    """Prior ω=2.0, true ω=4.0. Should converge within 10s."""
    rng = np.random.default_rng(1)
    true_omega = 4.0
    ekf = EKFL1(
        initial_omega=2.0,
        measurement_noise=0.02**2,
        process_noise_omega=0.2,
    )
    dt = 0.005
    n = 2000  # 10s
    for k in range(n):
        t = k * dt
        true_phi = (true_omega * t) % TWO_PI
        ekf.predict(t)
        z = (true_phi + rng.normal(0, 0.02)) % TWO_PI
        ekf.update(z)
    omega_err = abs(ekf.state.omega - true_omega)
    assert omega_err < 0.5, f"Omega not converged: est={ekf.state.omega:.3f} vs true={true_omega}"


# ─── Forecast ────────────────────────────────────────────────────────────


def test_l1_predict_ahead_zero_tau_returns_current():
    ekf = EKFL1()
    ekf.state.x[:] = [1.2, 3.5]
    phi_f, sig_phi, omega_f, sig_omega = ekf.predict_ahead(0.0)
    assert abs(phi_f - 1.2) < 1e-9
    assert abs(omega_f - 3.5) < 1e-9
    assert sig_phi >= 0 and sig_omega >= 0


def test_l1_predict_ahead_50ms():
    """φ should advance by ω × 0.05."""
    ekf = EKFL1()
    ekf.state.x[:] = [1.0, 3.0]
    phi_f, _, omega_f, _ = ekf.predict_ahead(0.05)
    expected = (1.0 + 3.0 * 0.05) % TWO_PI
    assert abs(phi_f - expected) < 1e-9
    assert abs(omega_f - 3.0) < 1e-9  # omega unchanged in L1


def test_l1_predict_ahead_variance_monotone():
    """σ_phi should grow with τ (no observations to shrink it)."""
    ekf = EKFL1()
    ekf.predict(0.0)
    sigmas = []
    for tau in [0.0, 0.02, 0.05, 0.1, 0.2]:
        _, sig_phi, _, _ = ekf.predict_ahead(tau)
        sigmas.append(sig_phi)
    for a, b in zip(sigmas[:-1], sigmas[1:]):
        assert b >= a - 1e-12, f"sigma not monotone: {sigmas}"


def test_l1_predict_ahead_does_not_mutate():
    """predict_ahead is a forecast — must not change filter state."""
    ekf = EKFL1()
    ekf.predict(0.0)
    x_before = ekf.state.x.copy()
    P_before = ekf.state.P.copy()
    _ = ekf.predict_ahead(0.1)
    assert np.allclose(ekf.state.x, x_before)
    assert np.allclose(ekf.state.P, P_before)


def test_l1_predict_ahead_negative_tau_clamped():
    """Negative τ should be treated as 0."""
    ekf = EKFL1()
    ekf.state.x[:] = [1.0, 3.0]
    phi_f, _, _, _ = ekf.predict_ahead(-0.1)
    assert abs(phi_f - 1.0) < 1e-9


# ─── Reset + diagnostics ─────────────────────────────────────────────────


def test_l1_reset_clears_state():
    ekf = EKFL1()
    ekf.predict(0.0)
    ekf.state.x[:] = [2.0, 6.0]
    ekf.reset()
    assert ekf.state.t_last is None
    assert ekf.state.phi == 0.0
    assert abs(ekf.state.omega - _DEFAULT_INITIAL_OMEGA) < 1e-12


def test_l1_reset_with_omega():
    ekf = EKFL1(initial_omega=4.0)
    ekf.reset(initial_omega=6.0)
    assert abs(ekf.state.omega - 6.0) < 1e-12


def test_l1_condition_number_finite():
    """After many updates, condition number remains finite + reasonable."""
    rng = np.random.default_rng(0)
    ekf = EKFL1(measurement_noise=0.01)
    ekf.predict(0.0)
    for k in range(500):
        ekf.predict(k * 0.005)
        ekf.update(rng.uniform(0, TWO_PI))
    cn = ekf.condition_number_P()
    assert math.isfinite(cn)
    assert cn < 1e10, f"Condition number too high: {cn}"


# ─── Codex Phase 1.5 — NEEDS_FIX coverage tests ───────────────────────────


def test_l1_predict_status_initial():
    """First predict() returns INITIAL — no integration."""
    ekf = EKFL1()
    status = ekf.predict(100.0)
    assert status == PredictStatus.INITIAL


def test_l1_predict_status_ok_normal():
    """Second predict() with valid dt returns OK."""
    ekf = EKFL1()
    ekf.predict(0.0)
    status = ekf.predict(0.01)
    assert status == PredictStatus.OK


def test_l1_predict_status_dt_non_positive():
    """Clock going backward returns DT_NON_POSITIVE."""
    ekf = EKFL1()
    ekf.predict(0.1)
    status = ekf.predict(0.05)
    assert status == PredictStatus.DT_NON_POSITIVE


def test_l1_predict_status_dt_zero_non_positive():
    """dt=0 is also non-positive (no information added)."""
    ekf = EKFL1()
    ekf.predict(0.1)
    status = ekf.predict(0.1)
    assert status == PredictStatus.DT_NON_POSITIVE


def test_l1_predict_status_dt_too_large():
    """dt > max_dt_s returns DT_TOO_LARGE — caller must trigger watchdog."""
    ekf = EKFL1(max_dt_s=0.5)
    ekf.predict(0.0)
    status = ekf.predict(1.0)
    assert status == PredictStatus.DT_TOO_LARGE
    # Critical: even though state did NOT integrate, t_last MUST advance
    # so the next call sees a fresh, small dt.
    assert ekf.state.t_last == 1.0


def test_l1_predict_status_nan_inf():
    """NaN / inf t_now reports T_NOT_FINITE."""
    ekf = EKFL1()
    ekf.predict(0.0)
    assert ekf.predict(float("nan")) == PredictStatus.T_NOT_FINITE
    assert ekf.predict(float("inf")) == PredictStatus.T_NOT_FINITE


def test_l1_update_returns_bool():
    """update() returns True when applied, False on skip — for caller diagnostics."""
    ekf = EKFL1()
    ekf.predict(0.0)
    assert ekf.update(1.0) is True
    # NaN observation
    assert ekf.update(float("nan")) is False
    # Negative R
    assert ekf.update(1.0, R_override=-0.1) is False
    # Zero R
    assert ekf.update(1.0, R_override=0.0) is False
    # NaN R
    assert ekf.update(1.0, R_override=float("nan")) is False
    # Inf R
    assert ekf.update(1.0, R_override=float("inf")) is False


def test_l1_update_negative_R_does_not_corrupt_state():
    """Negative R must NOT poison Joseph form — state must stay sane."""
    ekf = EKFL1()
    ekf.predict(0.0)
    x_before = ekf.state.x.copy()
    P_before = ekf.state.P.copy()
    applied = ekf.update(1.0, R_override=-0.5)
    assert applied is False
    assert np.allclose(ekf.state.x, x_before)
    assert np.allclose(ekf.state.P, P_before)


def test_l1_q_discretization_exact_form():
    """Q_d must match integrated continuous-discrete form (Codex NEEDS_FIX #1).

    For q_omega white noise on omega:
        Q_d[0,0] = q_omega × dt^3/3 + q_phi × dt
        Q_d[0,1] = Q_d[1,0] = q_omega × dt^2/2
        Q_d[1,1] = q_omega × dt
    """
    q_phi = 1e-6
    q_omega = 4e-2
    dt = 0.01
    ekf = EKFL1(process_noise_phi=q_phi, process_noise_omega=q_omega)
    Q_d = ekf._build_Qd(dt)
    expected_00 = q_omega * dt**3 / 3.0 + q_phi * dt
    expected_01 = q_omega * dt**2 / 2.0
    expected_11 = q_omega * dt
    assert abs(Q_d[0, 0] - expected_00) < 1e-15
    assert abs(Q_d[0, 1] - expected_01) < 1e-15
    assert abs(Q_d[1, 0] - expected_01) < 1e-15
    assert abs(Q_d[1, 1] - expected_11) < 1e-15
    # Symmetric
    assert Q_d[0, 1] == Q_d[1, 0]
    # PSD: 2x2 form is PSD iff det >= 0 (already symmetric)
    det = Q_d[0, 0] * Q_d[1, 1] - Q_d[0, 1] * Q_d[1, 0]
    assert det >= -1e-15, f"Q_d not PSD: det={det}"


def test_l1_q_discretization_phi_omega_coupling():
    """Sanity: Q_d[0,1] != 0 — naive diag(q_phi, q_omega) × dt would have 0 here.

    This is the Codex fix: integrated noise couples phi and omega.
    """
    ekf = EKFL1(process_noise_phi=0.0, process_noise_omega=1.0)
    Q_d = ekf._build_Qd(0.01)
    assert Q_d[0, 1] > 0, "Coupling term missing — naive Q_c × dt would zero this"


def test_l1_initial_P_reduced_from_pi_squared():
    """Codex NEEDS_FIX #3: initial P_phi reduced from π² to 1.0."""
    ekf = EKFL1()
    # P_phi should be sigma ~ 1 rad, not sigma ~ π rad
    sigma_phi = ekf.state.sigma_phi
    assert 0.5 <= sigma_phi <= 2.0, f"Initial sigma_phi out of expected range: {sigma_phi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
