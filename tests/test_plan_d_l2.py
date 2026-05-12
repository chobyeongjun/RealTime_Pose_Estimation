"""Plan D EKF L2 tests — const-acceleration 3-state Kalman.

Coverage:
    - Initialization (state shapes, defaults)
    - Cascade promotion from L1 (from_l1)
    - Predict step (F build, Q discretization, dt validation, status enum)
    - Update step (phase wrap, Joseph PSD, NaN guard, R validate)
    - Forecast (predict_ahead variance, monotone)
    - Numerical health (condition number, PSD across many steps)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.ekf_l1 import EKFL1
from perception.plan_d_prototype.ekf_l2 import (
    _DEFAULT_INITIAL_ALPHA,
    _DEFAULT_INITIAL_OMEGA,
    EKFL2,
    EKFL2State,
)
from perception.plan_d_prototype.ekf_l1 import PredictStatus
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_pi


# ─── Initialization ──────────────────────────────────────────────────────


def test_l2_init_default_state():
    ekf = EKFL2()
    assert ekf.state.x.shape == (3,)
    assert ekf.state.P.shape == (3, 3)
    assert ekf.state.t_last is None
    assert ekf.state.phi == 0.0
    assert abs(ekf.state.omega - _DEFAULT_INITIAL_OMEGA) < 1e-12
    assert abs(ekf.state.alpha - _DEFAULT_INITIAL_ALPHA) < 1e-12


def test_l2_init_dtype_float64():
    ekf = EKFL2()
    assert ekf.state.x.dtype == np.float64
    assert ekf.state.P.dtype == np.float64


def test_l2_init_P_psd():
    ekf = EKFL2()
    eigvals = np.linalg.eigvalsh(ekf.state.P)
    assert np.all(eigvals > 0)


# ─── from_l1 cascade promotion ───────────────────────────────────────────


def test_l2_from_l1_copies_phi_omega():
    l1 = EKFL1()
    l1.state.x[0] = 2.5
    l1.state.x[1] = 5.5
    l1.predict(0.0)
    l2 = EKFL2.from_l1(l1)
    assert l2.state.phi == 2.5
    assert l2.state.omega == 5.5
    assert l2.state.alpha == 0.0
    # Same t_last preserved
    assert l2.state.t_last == l1.state.t_last


def test_l2_from_l1_copies_p_block():
    """2x2 P block from L1 must appear in L2 [:2, :2]."""
    l1 = EKFL1()
    # Mutate P to a specific 2x2
    l1.state.P = np.array([[0.5, 0.1], [0.1, 0.3]], dtype=np.float64)
    l2 = EKFL2.from_l1(l1, initial_P_alpha=9.0)
    assert np.allclose(l2.state.P[:2, :2], l1.state.P)
    # α slot has the provided variance, off-diagonals zero
    assert l2.state.P[2, 2] == 9.0
    assert l2.state.P[0, 2] == 0.0
    assert l2.state.P[1, 2] == 0.0
    assert l2.state.P[2, 0] == 0.0
    assert l2.state.P[2, 1] == 0.0


def test_l2_from_l1_initial_alpha_arg():
    l1 = EKFL1()
    l2 = EKFL2.from_l1(l1, initial_alpha=0.3)
    assert l2.state.alpha == 0.3


# ─── Predict status enum ─────────────────────────────────────────────────


def test_l2_predict_status_initial():
    ekf = EKFL2()
    status = ekf.predict(100.0)
    assert status == PredictStatus.INITIAL


def test_l2_predict_status_ok_normal():
    ekf = EKFL2()
    ekf.predict(0.0)
    assert ekf.predict(0.01) == PredictStatus.OK


def test_l2_predict_status_dt_too_large():
    ekf = EKFL2(max_dt_s=0.5)
    ekf.predict(0.0)
    assert ekf.predict(1.0) == PredictStatus.DT_TOO_LARGE


def test_l2_predict_status_dt_non_positive():
    ekf = EKFL2()
    ekf.predict(0.1)
    assert ekf.predict(0.05) == PredictStatus.DT_NON_POSITIVE


# ─── Predict step dynamics ───────────────────────────────────────────────


def test_l2_const_velocity_when_alpha_zero():
    """α=0 → φ advances by ω×t (same as L1 with no accel)."""
    omega = 3.0
    ekf = EKFL2(initial_omega=omega, initial_alpha=0.0,
                process_noise_alpha=0.0, process_noise_phi=0.0)
    ekf.predict(0.0)
    for k in range(1, 101):
        ekf.predict(k * 0.01)
    expected = (omega * 1.0) % TWO_PI
    err = abs(float(wrap_to_pi(ekf.state.phi - expected)))
    assert err < 1e-9, f"L2 zero-α integration error: {err}"


def test_l2_const_accel_dynamics():
    """α=0.5 → ω(t) = ω₀ + α t, φ(t) = ω₀ t + ½ α t² (mod 2π)."""
    omega0 = 2.0
    alpha = 0.5
    ekf = EKFL2(initial_omega=omega0, initial_alpha=alpha,
                process_noise_alpha=0.0, process_noise_phi=0.0)
    ekf.predict(0.0)
    for k in range(1, 101):
        ekf.predict(k * 0.01)
    t = 1.0
    expected_omega = omega0 + alpha * t
    expected_phi = (omega0 * t + 0.5 * alpha * t * t) % TWO_PI
    assert abs(ekf.state.omega - expected_omega) < 1e-9
    err_phi = abs(float(wrap_to_pi(ekf.state.phi - expected_phi)))
    assert err_phi < 1e-9, f"L2 const-accel phi error: {err_phi}"


def test_l2_Q_d_exact_form():
    """Q_d coefficients match analytical integral for q_α white noise."""
    q_phi = 1e-6
    q_alpha = 0.2
    dt = 0.01
    ekf = EKFL2(process_noise_phi=q_phi, process_noise_alpha=q_alpha)
    Q_d = ekf._build_Qd(dt)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    # Top row
    assert abs(Q_d[0, 0] - (q_alpha * dt5 / 20.0 + q_phi * dt)) < 1e-18
    assert abs(Q_d[0, 1] - q_alpha * dt4 / 8.0) < 1e-18
    assert abs(Q_d[0, 2] - q_alpha * dt3 / 6.0) < 1e-18
    # Middle row
    assert abs(Q_d[1, 0] - q_alpha * dt4 / 8.0) < 1e-18
    assert abs(Q_d[1, 1] - q_alpha * dt3 / 3.0) < 1e-18
    assert abs(Q_d[1, 2] - q_alpha * dt2 / 2.0) < 1e-18
    # Bottom row
    assert abs(Q_d[2, 0] - q_alpha * dt3 / 6.0) < 1e-18
    assert abs(Q_d[2, 1] - q_alpha * dt2 / 2.0) < 1e-18
    assert abs(Q_d[2, 2] - q_alpha * dt) < 1e-18
    # Symmetric
    assert np.allclose(Q_d, Q_d.T)
    # PSD
    eigvals = np.linalg.eigvalsh(Q_d)
    assert np.all(eigvals >= -1e-15)


def test_l2_predict_keeps_P_psd():
    """100 predict() calls — P remains PSD + symmetric."""
    ekf = EKFL2()
    ekf.predict(0.0)
    for k in range(1, 101):
        ekf.predict(k * 0.01)
    P = ekf.state.P
    assert np.allclose(P, P.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals > -1e-9)


# ─── Update step ─────────────────────────────────────────────────────────


def test_l2_update_returns_bool():
    ekf = EKFL2()
    ekf.predict(0.0)
    assert ekf.update(1.0) is True
    assert ekf.update(float("nan")) is False
    assert ekf.update(1.0, R_override=-0.5) is False
    assert ekf.update(1.0, R_override=0.0) is False
    assert ekf.update(1.0, R_override=float("nan")) is False


def test_l2_update_innovation_phase_wrap():
    """Observation near 2π, state near 0 → small innovation."""
    ekf = EKFL2(measurement_noise=0.01)
    ekf.predict(0.0)
    ekf.state.x[:] = [0.05, 4.0, 0.0]
    z = TWO_PI - 0.05
    phi_before = ekf.state.phi
    ekf.update(z)
    err = abs(float(wrap_to_pi(ekf.state.phi - phi_before)))
    assert err < 0.5, f"Phase wrap broken in L2 update: state jumped {err}"


def test_l2_update_keeps_P_psd_after_many():
    rng = np.random.default_rng(99)
    ekf = EKFL2(measurement_noise=0.05)
    ekf.predict(0.0)
    for k in range(200):
        ekf.predict(k * 0.01)
        ekf.update(rng.uniform(0, TWO_PI))
    eigvals = np.linalg.eigvalsh(ekf.state.P)
    assert np.all(eigvals > -1e-9)


# ─── Convergence ─────────────────────────────────────────────────────────


def test_l2_converges_omega_under_const_accel_observation():
    """Subject's ω increases linearly. L2 should track both ω and α."""
    rng = np.random.default_rng(0)
    omega0 = 3.0
    alpha = 0.1   # rad/s² — slow accel
    ekf = EKFL2(
        initial_omega=omega0,
        initial_alpha=0.0,
        process_noise_alpha=0.5,
        measurement_noise=0.02 ** 2,
    )
    dt = 0.01
    n = 1000  # 10 s
    for k in range(n):
        t = k * dt
        true_omega = omega0 + alpha * t
        true_phi = ((omega0 * t + 0.5 * alpha * t * t)) % TWO_PI
        ekf.predict(t)
        z = (true_phi + rng.normal(0, 0.02)) % TWO_PI
        ekf.update(z)
    # After 10 s of const-α=0.1, ω should be ~4.0
    assert abs(ekf.state.omega - 4.0) < 0.6, f"L2 ω: {ekf.state.omega}"
    # α should converge near 0.1 (within tolerance — noisy observation)
    assert abs(ekf.state.alpha - alpha) < 0.5, f"L2 α: {ekf.state.alpha}"


# ─── Forecast ────────────────────────────────────────────────────────────


def test_l2_predict_ahead_const_accel_propagation():
    """With known x = [1.0, 3.0, 0.5], τ=0.1 → φ = 1.0 + 0.3 + 0.0025 = 1.3025."""
    ekf = EKFL2(process_noise_phi=0.0, process_noise_alpha=0.0)
    ekf.state.x[:] = [1.0, 3.0, 0.5]
    tau = 0.1
    phi_f, _, omega_f, _, alpha_f, _ = ekf.predict_ahead(tau)
    expected_phi = (1.0 + 3.0 * tau + 0.5 * 0.5 * tau * tau) % TWO_PI
    assert abs(phi_f - expected_phi) < 1e-12
    assert abs(omega_f - (3.0 + 0.5 * tau)) < 1e-12
    assert abs(alpha_f - 0.5) < 1e-12


def test_l2_predict_ahead_does_not_mutate():
    ekf = EKFL2()
    ekf.predict(0.0)
    x_before = ekf.state.x.copy()
    P_before = ekf.state.P.copy()
    _ = ekf.predict_ahead(0.1)
    assert np.allclose(ekf.state.x, x_before)
    assert np.allclose(ekf.state.P, P_before)


def test_l2_predict_ahead_variance_grows():
    ekf = EKFL2()
    ekf.predict(0.0)
    sigmas_phi = []
    for tau in [0.0, 0.05, 0.1, 0.2]:
        _, s_phi, _, _, _, _ = ekf.predict_ahead(tau)
        sigmas_phi.append(s_phi)
    for a, b in zip(sigmas_phi[:-1], sigmas_phi[1:]):
        assert b >= a - 1e-12


# ─── Reset + diagnostics ─────────────────────────────────────────────────


def test_l2_reset_clears():
    ekf = EKFL2()
    ekf.predict(0.0)
    ekf.state.x[:] = [2.0, 6.0, 0.7]
    ekf.reset()
    assert ekf.state.t_last is None
    assert ekf.state.phi == 0.0


def test_l2_condition_number_finite():
    rng = np.random.default_rng(1)
    ekf = EKFL2(measurement_noise=0.05)
    ekf.predict(0.0)
    for k in range(500):
        ekf.predict(k * 0.005)
        ekf.update(rng.uniform(0, TWO_PI))
    cn = ekf.condition_number_P()
    assert math.isfinite(cn)
    assert cn < 1e10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
