"""Plan D CrossCorrPhaseEstimator tests — template matching for φ observation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.phase_estimator import (
    CrossCorrPhaseEstimator,
    PhaseEstimate,
)
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_pi


def _build_sinusoidal_template(n_bins: int = 128, n_joints: int = 6, n_iter: int = 200):
    """Build a saturated template with μ_k(φ) = sin(φ + k×π/6)."""
    rng = np.random.default_rng(0)
    t = CycleTemplate(n_bins=n_bins, n_joints=n_joints, beta_default=0.10)
    for _ in range(n_iter):
        # Train over all bins
        for bin_i in range(n_bins):
            phi = bin_i * TWO_PI / n_bins
            q = np.array([math.sin(phi + k * math.pi / 6) for k in range(n_joints)])
            t.update(phi, q)
    return t


# ─── Init + readiness gate ───────────────────────────────────────────────


def test_estimator_uninitialized_template_returns_invalid():
    """Empty template → estimator returns invalid."""
    t = CycleTemplate()
    est = CrossCorrPhaseEstimator(t)
    q = np.ones(6)
    result = est.estimate(q)
    assert isinstance(result, PhaseEstimate)
    assert result.valid is False
    assert math.isnan(result.phi)


def test_estimator_below_min_touched_invalid():
    """Touching < 25% of bins → invalid."""
    t = CycleTemplate(beta_default=0.10)
    # Update only 10 bins (out of 128 default)
    for bin_i in range(10):
        t.update(bin_i * TWO_PI / 128, np.ones(6))
    est = CrossCorrPhaseEstimator(t, min_touched_fraction=0.25)
    result = est.estimate(np.ones(6))
    assert result.valid is False


def test_estimator_above_min_touched_valid():
    """Touching ≥ 25% → valid."""
    t = CycleTemplate(beta_default=0.10)
    # Touch 40 bins
    for bin_i in range(40):
        t.update(bin_i * TWO_PI / 128, np.ones(6))
    est = CrossCorrPhaseEstimator(t, min_touched_fraction=0.25)
    result = est.estimate(np.ones(6))
    assert result.valid is True


# ─── Phase recovery ──────────────────────────────────────────────────────


def test_estimator_recovers_known_phase():
    """Train sinusoidal template, query at known φ → estimate matches."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    n_joints = t.n_joints
    for phi_true in [0.5, 1.0, math.pi, 4.5, 5.8]:
        q = np.array(
            [math.sin(phi_true + k * math.pi / 6) for k in range(n_joints)]
        )
        result = est.estimate(q)
        assert result.valid
        err = abs(float(wrap_to_pi(result.phi - phi_true)))
        # Cycle template recovery: should be within ~1 bin (2π/128 ≈ 0.05 rad)
        # Subpixel + saturated template → tighter
        assert err < 0.08, f"phi_true={phi_true}, estimate={result.phi}, err={err}"


def test_estimator_subpixel_better_than_bin():
    """At a phase NOT on a bin center, subpixel should yield error < bin width."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    bin_width = TWO_PI / 128
    rng = np.random.default_rng(7)
    errors = []
    for _ in range(20):
        # Pick phase 1/3 between two bin centers (off-grid)
        bin_i = rng.integers(0, 128)
        phi_true = (bin_i + 0.33) * bin_width
        q = np.array([math.sin(phi_true + k * math.pi / 6) for k in range(t.n_joints)])
        result = est.estimate(q)
        if result.valid:
            err = abs(float(wrap_to_pi(result.phi - phi_true)))
            errors.append(err)
    mean_err = np.mean(errors)
    assert mean_err < bin_width / 2, f"Subpixel error too high: {mean_err}"


def test_estimator_noise_robustness():
    """Add σ=0.05 observation noise — error should stay bounded."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    rng = np.random.default_rng(11)
    errors = []
    for _ in range(50):
        phi_true = rng.uniform(0, TWO_PI)
        q_clean = np.array(
            [math.sin(phi_true + k * math.pi / 6) for k in range(t.n_joints)]
        )
        q_noisy = q_clean + rng.normal(0, 0.05, size=t.n_joints)
        result = est.estimate(q_noisy)
        if result.valid:
            errors.append(abs(float(wrap_to_pi(result.phi - phi_true))))
    p95 = float(np.percentile(errors, 95))
    assert p95 < 0.2, f"Noisy estimate p95 error too high: {p95}"


# ─── Confidence ──────────────────────────────────────────────────────────


def test_estimator_ambiguity_low_for_sharp_match():
    """At clean φ_true on a well-trained sinusoidal template, ambiguity_ratio
    should be LOW (sharp minimum vs second-best — low = sharp = trustworthy)."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    q = np.array([math.sin(1.0 + k * math.pi / 6) for k in range(t.n_joints)])
    result = est.estimate(q)
    assert result.ambiguity_ratio < 0.5, (
        f"ambiguity_ratio not sharp: {result.ambiguity_ratio}"
    )


def test_estimator_ambiguity_high_for_ambiguous():
    """Flat template (no phase info) → high ambiguity_ratio (no preferred φ)."""
    t = CycleTemplate(beta_default=0.10)
    # Fill all bins with same q → no phase discrimination
    for bin_i in range(128):
        t.update(bin_i * TWO_PI / 128, np.ones(6))
    est = CrossCorrPhaseEstimator(t)
    # Saturate
    for _ in range(50):
        for bin_i in range(128):
            t.update(bin_i * TWO_PI / 128, np.ones(6))
    result = est.estimate(np.ones(6))
    # All bins identical → no preferred φ → ambiguity_ratio ~1
    assert result.ambiguity_ratio > 0.5, (
        f"ambiguity_ratio not high for ambiguous: {result.ambiguity_ratio}"
    )


def test_estimator_sigma_floor_prevents_numerical_blowup():
    """Codex NEEDS_FIX #9: sigma_per_joint=0 (numerically near-zero) must not
    cause inverse-variance explosion."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    phi_true = 1.0
    q = np.array([math.sin(phi_true + k * math.pi / 6) for k in range(t.n_joints)])
    # Extremely tight sigma — would have produced inv_var=1e12 pre-fix
    sigma = np.full(t.n_joints, 1e-12)
    result = est.estimate(q, sigma_per_joint=sigma)
    assert result.valid
    assert math.isfinite(result.phi)
    assert math.isfinite(result.cost)
    # Despite extreme tight sigma, phase recovery should still work
    err = abs(float((result.phi - phi_true + math.pi) % (2 * math.pi) - math.pi))
    assert err < 0.1, f"Phase recovery broken under sigma floor: {err}"


def test_estimator_sigma_nan_treated_as_floor():
    """NaN sigma entries should be clamped to floor, not propagated."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    q = np.array([math.sin(1.0 + k * math.pi / 6) for k in range(t.n_joints)])
    sigma = np.array([1.0, float("nan"), 1.0, 1.0, 1.0, 1.0])
    result = est.estimate(q, sigma_per_joint=sigma)
    assert result.valid
    assert math.isfinite(result.phi)


def test_estimator_invalid_sigma_floor_raises():
    """sigma_floor must be > 0."""
    t = _build_sinusoidal_template()
    with pytest.raises(ValueError):
        CrossCorrPhaseEstimator(t, sigma_floor=0.0)
    with pytest.raises(ValueError):
        CrossCorrPhaseEstimator(t, sigma_floor=-0.01)


# ─── Sigma weighting ─────────────────────────────────────────────────────


def test_estimator_per_joint_sigma_weighting():
    """A joint with huge σ should be ignored; estimate should rely on others."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    phi_true = 1.2
    q = np.array([math.sin(phi_true + k * math.pi / 6) for k in range(t.n_joints)])
    # Corrupt joint 0 with garbage
    q[0] = 999.0
    # Without weighting → estimate ruined
    result_unweighted = est.estimate(q)
    # With huge σ on joint 0 → estimate rescued
    sigma = np.array([1e3, 1.0, 1.0, 1.0, 1.0, 1.0])
    result_weighted = est.estimate(q, sigma_per_joint=sigma)
    err_unweighted = abs(float(wrap_to_pi(result_unweighted.phi - phi_true)))
    err_weighted = abs(float(wrap_to_pi(result_weighted.phi - phi_true)))
    assert err_weighted < err_unweighted, (
        f"Weighting didn't help: unw={err_unweighted}, w={err_weighted}"
    )
    assert err_weighted < 0.15


# ─── NaN handling ────────────────────────────────────────────────────────


def test_estimator_nan_observation_excludes_joint():
    """NaN in q should not propagate to result."""
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    phi_true = 1.5
    q = np.array([math.sin(phi_true + k * math.pi / 6) for k in range(t.n_joints)])
    q[2] = float("nan")
    result = est.estimate(q)
    # Should still produce finite estimate
    assert result.valid
    assert math.isfinite(result.phi)
    err = abs(float(wrap_to_pi(result.phi - phi_true)))
    assert err < 0.2


def test_estimator_all_nan_invalid():
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    q = np.full(t.n_joints, float("nan"))
    result = est.estimate(q)
    assert result.valid is False


# ─── Shape errors ────────────────────────────────────────────────────────


def test_estimator_wrong_q_shape_raises():
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    with pytest.raises(ValueError):
        est.estimate(np.array([1.0, 2.0]))


def test_estimator_wrong_sigma_shape_raises():
    t = _build_sinusoidal_template()
    est = CrossCorrPhaseEstimator(t)
    q = np.ones(t.n_joints)
    with pytest.raises(ValueError):
        est.estimate(q, sigma_per_joint=np.array([1.0, 2.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
