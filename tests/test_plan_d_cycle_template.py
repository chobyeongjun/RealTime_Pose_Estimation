"""Plan D CycleTemplate tests — recursive μ(φ) with cubic Hermite interp."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.cycle_template import (
    BETA_MAX,
    BETA_MIN,
    CycleTemplate,
)
from perception.plan_d_prototype.utils import TWO_PI, bin_of_phase


# ─── Initialization ──────────────────────────────────────────────────────


def test_template_init_default():
    t = CycleTemplate()
    assert t.n_bins == 128
    assert t.n_joints == 6
    assert t.mu.shape == (128, 6)
    assert np.all(t.mu == 0.0)
    assert t.is_initialized is False
    assert t.touched_fraction == 0.0
    assert t.total_updates == 0


def test_template_init_custom():
    t = CycleTemplate(n_bins=64, n_joints=4, beta_default=0.07)
    assert t.n_bins == 64
    assert t.n_joints == 4
    assert t.mu.shape == (64, 4)


def test_template_init_invalid_n_bins():
    with pytest.raises(ValueError):
        CycleTemplate(n_bins=4)


def test_template_init_invalid_beta():
    with pytest.raises(ValueError):
        CycleTemplate(beta_default=0.5)
    with pytest.raises(ValueError):
        CycleTemplate(beta_default=0.01)


# ─── Update ──────────────────────────────────────────────────────────────


def test_template_update_single_bin():
    """Update at φ=0 with β=0.10 → bin[0] = 0.10 × q."""
    t = CycleTemplate(beta_default=0.10)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t.update(0.0, q)
    assert np.allclose(t.mu[0], 0.10 * q)
    assert t.is_initialized
    assert t.total_updates == 1


def test_template_update_recursive_convergence():
    """Repeated same q → μ converges to q."""
    t = CycleTemplate(beta_default=0.10)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for _ in range(200):
        t.update(0.5, q)
    # After 200 updates at β=0.10, error should be < (0.9)^200 ≈ 7e-10
    assert np.allclose(t.mu[bin_of_phase(0.5, 128)], q, atol=1e-6)


def test_template_update_wrong_q_shape_raises():
    t = CycleTemplate(n_joints=6)
    with pytest.raises(ValueError):
        t.update(0.0, np.array([1.0, 2.0, 3.0]))


def test_template_update_nan_phi_no_op():
    t = CycleTemplate()
    t.update(float("nan"), np.ones(6))
    assert t.total_updates == 0


def test_template_update_nan_q_partial():
    """NaN entries in q skip that joint but update others."""
    t = CycleTemplate(beta_default=0.10)
    q = np.array([1.0, float("nan"), 3.0, 4.0, 5.0, 6.0])
    t.update(0.5, q)
    idx = bin_of_phase(0.5, 128)
    # Joint 0, 2, 3, 4, 5 updated; joint 1 unchanged (still 0)
    assert t.mu[idx, 0] != 0.0
    assert t.mu[idx, 1] == 0.0
    assert t.mu[idx, 2] != 0.0


def test_template_update_all_nan_q_no_change():
    t = CycleTemplate()
    q = np.full(6, float("nan"))
    t.update(0.5, q)
    # No bin updates
    assert t.total_updates == 0


def test_template_beta_clamped_to_clinical():
    """β outside [0.03, 0.10] is clamped (defensive)."""
    t = CycleTemplate(beta_default=0.05)
    q = np.array([1.0] * 6)
    # Try β=0.5 (too high) — should clamp to BETA_MAX
    t.update(0.0, q, beta=0.5)
    idx = bin_of_phase(0.0, 128)
    assert np.allclose(t.mu[idx], BETA_MAX * q)
    # Try β=0.0 (too low) — should clamp to BETA_MIN
    t.reset()
    t.update(0.0, q, beta=0.0)
    assert np.allclose(t.mu[idx], BETA_MIN * q)


# ─── Lookup (cubic Hermite) ──────────────────────────────────────────────


def test_template_lookup_empty_returns_zero():
    t = CycleTemplate()
    result = t.lookup(1.0)
    assert np.allclose(result, 0.0)


def test_template_lookup_single_bin_falls_back_to_nearest():
    """One bin updated, lookup elsewhere should return that bin's value."""
    t = CycleTemplate(beta_default=0.10)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Saturate bin 0
    for _ in range(100):
        t.update(0.0, q)
    # Lookup at a faraway phase (bin 60) — nearest fallback
    result = t.lookup(math.pi)
    assert result.shape == (6,)
    # Should fall back to nearest touched (bin 0) — non-zero
    assert np.linalg.norm(result) > 0.0


def test_template_lookup_at_bin_center_matches():
    """If two adjacent bins same value, Hermite returns that value at center."""
    t = CycleTemplate(beta_default=0.10)
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Fill bins 0-3 to saturation
    phases = [b * TWO_PI / 128 for b in range(8)]
    for phi in phases:
        for _ in range(100):
            t.update(phi, q)
    # Lookup at bin 4 center (well inside filled region)
    phi_test = 4.0 * TWO_PI / 128
    result = t.lookup(phi_test)
    assert np.allclose(result, q, atol=0.01)


def test_template_lookup_smoothness():
    """Sample lookup along φ axis — output should be C¹ (no kinks)."""
    rng = np.random.default_rng(0)
    t = CycleTemplate(beta_default=0.10)
    # Train with a sinusoid
    for _ in range(500):
        phi = rng.uniform(0, TWO_PI)
        q = np.array([math.sin(phi + i * 0.1) for i in range(6)])
        t.update(phi, q)
    # Sample densely
    phis = np.linspace(0.5, 1.0, 200)
    samples = np.array([t.lookup(p) for p in phis])
    # Second differences should be bounded (smoothness check)
    second_diff = np.diff(samples, n=2, axis=0)
    # If there were kinks, |second_diff| could blow up
    assert np.max(np.abs(second_diff)) < 0.5, "Template lookup not smooth"


def test_template_lookup_jacobian_zero_when_constant():
    """μ constant in φ → ∂μ/∂φ = 0."""
    t = CycleTemplate(beta_default=0.10)
    q = np.ones(6)
    for bin_i in range(128):
        phi = bin_i * TWO_PI / 128
        for _ in range(50):
            t.update(phi, q)
    jac = t.lookup_jacobian(1.0)
    assert np.allclose(jac, 0.0, atol=1e-2)


def test_template_lookup_jacobian_sign():
    """μ(φ) = sin(φ) → ∂μ/∂φ ≈ cos(φ)."""
    t = CycleTemplate(beta_default=0.10, n_joints=1)
    for bin_i in range(128):
        phi = bin_i * TWO_PI / 128
        q = np.array([math.sin(phi)])
        for _ in range(100):
            t.update(phi, q)
    # At φ=0, cos(0)=1
    jac_at_0 = t.lookup_jacobian(0.0)[0]
    assert jac_at_0 > 0.3, f"Expected positive slope, got {jac_at_0}"
    # At φ=π, cos(π)=-1
    jac_at_pi = t.lookup_jacobian(math.pi)[0]
    assert jac_at_pi < -0.3, f"Expected negative slope, got {jac_at_pi}"


def test_template_lookup_nan_phi():
    t = CycleTemplate()
    result = t.lookup(float("nan"))
    assert np.all(np.isnan(result))


def test_template_lookup_wraps_at_2pi():
    """lookup(0) == lookup(2π)."""
    rng = np.random.default_rng(0)
    t = CycleTemplate(beta_default=0.10)
    for _ in range(500):
        phi = rng.uniform(0, TWO_PI)
        q = rng.normal(size=6)
        t.update(phi, q)
    a = t.lookup(0.0)
    b = t.lookup(TWO_PI)
    assert np.allclose(a, b, atol=1e-10)


# ─── Status + reset ──────────────────────────────────────────────────────


def test_template_touched_fraction():
    t = CycleTemplate()
    q = np.ones(6)
    t.update(0.0, q)
    t.update(math.pi, q)
    # 2 of 128 bins touched
    assert abs(t.touched_fraction - 2 / 128) < 1e-12


def test_template_reset_clears():
    t = CycleTemplate()
    q = np.ones(6)
    for phi in np.linspace(0, TWO_PI, 50):
        t.update(phi, q)
    t.reset()
    assert t.total_updates == 0
    assert t.touched_fraction == 0.0
    assert np.all(t.mu == 0.0)


def test_template_mu_is_defensive_copy():
    """External mutation of .mu should not affect internal state."""
    t = CycleTemplate()
    t.update(0.0, np.ones(6))
    mu_copy = t.mu
    mu_copy[0, 0] = 999.0
    # Internal should be unchanged
    assert t.mu[0, 0] != 999.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
