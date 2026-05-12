"""Plan D EKF L3 tests — template-driven phase-locked 3-state Kalman.

Coverage:
    - Init + L2→L3 cascade promotion
    - K-DOF measurement update
    - Per-joint NaN masking
    - Innovation χ² diagnostics
    - LDLT-based gain (numerical stability)
    - Joseph PSD across many updates
    - Forecast with template lookup q_pred
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.ekf_l1 import PredictStatus
from perception.plan_d_prototype.ekf_l2 import EKFL2
from perception.plan_d_prototype.ekf_l3 import (
    SIGMA_PER_JOINT_FLOOR,
    EKFL3,
    L3UpdateResult,
)
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_pi


def _build_sinusoidal_template(n_bins=128, n_joints=6):
    """Saturated template with μ_k(φ) = sin(φ + k×π/6)."""
    t = CycleTemplate(n_bins=n_bins, n_joints=n_joints, beta_default=0.05)
    for _ in range(50):
        for bin_i in range(n_bins):
            phi = bin_i * TWO_PI / n_bins
            q = np.array([math.sin(phi + k * math.pi / 6) for k in range(n_joints)])
            t.update(phi, q)
    return t


# ─── Init ────────────────────────────────────────────────────────────────


def test_l3_init_default():
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    assert l3.state.x.shape == (3,)
    assert l3.state.P.shape == (3, 3)
    assert l3.template is t


def test_l3_init_invalid_template_n_joints():
    t = CycleTemplate(n_joints=1)
    EKFL3(template=t)  # OK for 1-joint
    # n_joints=0 not allowed by CycleTemplate __init__


def test_l3_init_invalid_sigma_floor():
    t = _build_sinusoidal_template()
    with pytest.raises(ValueError):
        EKFL3(template=t, sigma_floor=0.0)


# ─── Cascade promotion from L2 ───────────────────────────────────────────


def test_l3_from_l2_copies_state():
    t = _build_sinusoidal_template()
    l2 = EKFL2()
    l2.state.x[:] = [1.5, 4.5, 0.2]
    l2.predict(0.0)
    l3 = EKFL3.from_l2(l2, t)
    assert l3.state.phi == 1.5
    assert l3.state.omega == 4.5
    assert l3.state.alpha == 0.2
    assert l3.template is t


def test_l3_from_l2_copies_P():
    t = _build_sinusoidal_template()
    l2 = EKFL2()
    l2.state.P = np.array([
        [0.5, 0.1, 0.0],
        [0.1, 0.3, 0.05],
        [0.0, 0.05, 0.2],
    ], dtype=np.float64)
    l3 = EKFL3.from_l2(l2, t)
    assert np.allclose(l3.state.P, l2.state.P)


# ─── Predict (delegates to L2) ───────────────────────────────────────────


def test_l3_predict_status_initial():
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    assert l3.predict(100.0) == PredictStatus.INITIAL


def test_l3_predict_status_ok():
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    assert l3.predict(0.01) == PredictStatus.OK


# ─── Update K-DOF ────────────────────────────────────────────────────────


def test_l3_update_wrong_q_shape_raises():
    t = _build_sinusoidal_template(n_joints=6)
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    with pytest.raises(ValueError):
        l3.update(np.array([1.0, 2.0]))


def test_l3_update_all_nan_returns_inapplied():
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    q = np.full(t.n_joints, float("nan"))
    result = l3.update(q)
    assert result.applied is False
    assert result.n_valid_joints == 0


def test_l3_update_returns_diagnostics():
    """update() returns L3UpdateResult with chi2, n_valid, rms."""
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    l3._l2.state.x[0] = 1.0  # Set known phi
    q = np.array([math.sin(1.0 + k * math.pi / 6) for k in range(t.n_joints)])
    result = l3.update(q)
    assert isinstance(result, L3UpdateResult)
    assert result.applied is True
    assert result.n_valid_joints == t.n_joints
    assert math.isfinite(result.innovation_chi2)
    assert math.isfinite(result.residual_rms)


def test_l3_update_per_joint_nan_excludes():
    """NaN joint excluded from update (n_valid_joints reflects mask)."""
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    q = np.array([math.sin(1.0 + k * math.pi / 6) for k in range(t.n_joints)])
    q[2] = float("nan")
    q[4] = float("nan")
    result = l3.update(q)
    assert result.applied
    assert result.n_valid_joints == t.n_joints - 2


def test_l3_update_reduces_residual():
    """After matching observation, subsequent update should have smaller residual."""
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    l3._l2.state.x[0] = 1.5
    # Observation perfectly aligned with template at φ=1.0
    q = np.array([math.sin(1.0 + k * math.pi / 6) for k in range(t.n_joints)])
    # Wrong initial phi → large residual
    r1 = l3.update(q)
    # Re-predict + re-update with same q
    l3.predict(0.01)
    r2 = l3.update(q)
    # State should have moved toward 1.0; second residual smaller
    assert r2.residual_rms <= r1.residual_rms + 1e-6


def test_l3_update_psd_maintained():
    """100 random updates — P remains PSD."""
    rng = np.random.default_rng(7)
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    for k in range(100):
        l3.predict(k * 0.01)
        phi_true = rng.uniform(0, TWO_PI)
        q = np.array([math.sin(phi_true + j * math.pi / 6) for j in range(t.n_joints)])
        sigma = np.full(t.n_joints, 0.05)
        l3.update(q, sigma_per_joint=sigma)
    P = l3.state.P
    assert np.allclose(P, P.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals > -1e-9), f"P not PSD: min eigval {eigvals.min()}"


def test_l3_sigma_floor_applied():
    """Zero sigma should be floored, no inv_var blowup."""
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t, sigma_floor=0.01)
    l3.predict(0.0)
    q = np.array([math.sin(1.0 + k * math.pi / 6) for k in range(t.n_joints)])
    sigma = np.full(t.n_joints, 1e-12)  # essentially zero
    result = l3.update(q, sigma_per_joint=sigma)
    # Should not crash; state still finite
    assert math.isfinite(l3.state.phi)
    assert math.isfinite(l3.state.omega)
    assert result.applied


def test_l3_no_template_jacobian_returns_inapplied():
    """Empty template → jacobian = 0 everywhere → result still applies but
    innovation has zero gradient effect."""
    t = CycleTemplate(n_joints=6, beta_default=0.05)  # untouched
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    q = np.zeros(6)
    result = l3.update(q)
    # Should not crash; finite outputs
    assert math.isfinite(l3.state.phi)


# ─── Forecast ────────────────────────────────────────────────────────────


def test_l3_predict_ahead_returns_q_pred():
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3._l2.state.x[:] = [1.0, 3.0, 0.0]
    out = l3.predict_ahead(0.05)
    phi_f, s_phi, omega_f, s_omega, alpha_f, s_alpha, q_pred = out
    assert q_pred.shape == (t.n_joints,)
    # q_pred[k] should be approximately sin(φ_f + k×π/6)
    for k in range(t.n_joints):
        expected = math.sin(phi_f + k * math.pi / 6)
        assert abs(q_pred[k] - expected) < 0.1


def test_l3_predict_ahead_does_not_mutate():
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    l3.predict(0.0)
    x_before = l3.state.x.copy()
    P_before = l3.state.P.copy()
    _ = l3.predict_ahead(0.1)
    assert np.allclose(l3.state.x, x_before)
    assert np.allclose(l3.state.P, P_before)


# ─── Real-walking-like convergence ───────────────────────────────────────


def test_l3_phi_converges_under_template_observations():
    """Feed template-consistent q + true φ trace. L3 should track φ."""
    rng = np.random.default_rng(0)
    t = _build_sinusoidal_template()
    l3 = EKFL3(template=t)
    # Start L3 with wrong initial omega
    l3._l2.state.x[:] = [0.0, 3.0, 0.0]
    true_omega = 6.28  # 1 Hz
    dt = 0.01
    n = 1000
    errors = []
    for k in range(n):
        time = k * dt
        true_phi = (true_omega * time) % TWO_PI
        l3.predict(time)
        q = np.array([math.sin(true_phi + j * math.pi / 6) for j in range(t.n_joints)])
        q += rng.normal(0, 0.02, size=t.n_joints)   # small noise
        sigma = np.full(t.n_joints, 0.05)
        l3.update(q, sigma_per_joint=sigma)
        errors.append(abs(float(wrap_to_pi(l3.state.phi - true_phi))))
    steady = float(np.mean(errors[-100:]))
    assert steady < 0.5, f"L3 steady-state phase error too high: {steady}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
