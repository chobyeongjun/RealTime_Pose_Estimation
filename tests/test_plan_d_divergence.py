"""Plan D divergence module tests — innovation gates, chi², jump detectors."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.divergence import (
    CHI2_THRESHOLD_99,
    CHI2_THRESHOLD_999,
    cadence_jump_detector,
    chi2_threshold,
    innovation_gate,
    mahalanobis_chi2,
    template_residual_chi2,
    vision_loss_detector,
)


# ─── chi2_threshold ──────────────────────────────────────────────────────


def test_chi2_threshold_99():
    assert chi2_threshold(1, 0.99) == CHI2_THRESHOLD_99[1]
    assert chi2_threshold(3, 0.99) == CHI2_THRESHOLD_99[3]
    assert chi2_threshold(6, 0.99) == CHI2_THRESHOLD_99[6]


def test_chi2_threshold_999():
    assert chi2_threshold(1, 0.999) == CHI2_THRESHOLD_999[1]
    assert chi2_threshold(6, 0.999) == CHI2_THRESHOLD_999[6]


def test_chi2_threshold_monotone_in_dof():
    """Higher DOF → higher threshold."""
    prev = -1.0
    for dof in [1, 2, 3, 4, 5, 6, 8, 10]:
        thr = chi2_threshold(dof, 0.99)
        assert thr > prev
        prev = thr


def test_chi2_threshold_invalid_confidence():
    with pytest.raises(ValueError):
        chi2_threshold(3, 0.95)
    with pytest.raises(ValueError):
        chi2_threshold(3, 0.5)


def test_chi2_threshold_unsupported_dof():
    with pytest.raises(KeyError):
        chi2_threshold(7, 0.99)
    with pytest.raises(KeyError):
        chi2_threshold(100, 0.99)


# ─── mahalanobis_chi2 ────────────────────────────────────────────────────


def test_mahalanobis_chi2_identity():
    """y=(1,1,1), S=I → χ² = 3."""
    inn = np.array([1.0, 1.0, 1.0])
    S = np.eye(3)
    assert abs(mahalanobis_chi2(inn, S) - 3.0) < 1e-12


def test_mahalanobis_chi2_diagonal_unequal():
    """y=(2,2), S=diag(1,4) → χ² = 4/1 + 4/4 = 5."""
    inn = np.array([2.0, 2.0])
    S = np.diag([1.0, 4.0])
    assert abs(mahalanobis_chi2(inn, S) - 5.0) < 1e-12


def test_mahalanobis_chi2_zero_innov():
    inn = np.zeros(4)
    S = np.eye(4)
    assert mahalanobis_chi2(inn, S) == 0.0


def test_mahalanobis_chi2_nan_innov():
    """NaN in innovation → NaN result."""
    inn = np.array([1.0, float("nan")])
    S = np.eye(2)
    assert math.isnan(mahalanobis_chi2(inn, S))


def test_mahalanobis_chi2_singular_S():
    """Singular S → NaN, no crash."""
    inn = np.array([1.0, 1.0])
    S = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
    result = mahalanobis_chi2(inn, S)
    # Either NaN or huge — both acceptable; chi must NOT raise
    assert math.isnan(result) or math.isfinite(result)


def test_mahalanobis_chi2_shape_mismatch_raises():
    inn = np.array([1.0, 1.0])
    S = np.eye(3)
    with pytest.raises(ValueError):
        mahalanobis_chi2(inn, S)


# ─── innovation_gate ─────────────────────────────────────────────────────


def test_innovation_gate_accept_small_chi2():
    """χ² well below threshold → accept (False = no divergence)."""
    assert innovation_gate(1.0, dof=1) is False
    assert innovation_gate(5.0, dof=3) is False


def test_innovation_gate_reject_large_chi2():
    """χ² above threshold → diverge (True)."""
    # 1-DOF threshold = 6.64; 10 > 6.64
    assert innovation_gate(10.0, dof=1) is True
    # 6-DOF threshold = 16.81; 20 > 16.81
    assert innovation_gate(20.0, dof=6) is True


def test_innovation_gate_nan_treated_as_diverged():
    """NaN χ² → treat as diverged (defensive)."""
    assert innovation_gate(float("nan"), dof=3) is True


def test_innovation_gate_negative_chi2_diverged():
    """Negative χ² is impossible mathematically — treat as diverged."""
    assert innovation_gate(-1.0, dof=3) is True


def test_innovation_gate_confidence_999_tighter():
    """At 99.9% confidence, threshold is higher → some accepts at 99 become accepts at 999."""
    chi2 = 7.0
    # 1-DOF: 99% threshold 6.63, 99.9% threshold 10.83
    assert innovation_gate(chi2, dof=1, confidence=0.99) is True
    assert innovation_gate(chi2, dof=1, confidence=0.999) is False


# ─── template_residual_chi2 ──────────────────────────────────────────────


def test_template_residual_chi2_perfect_match():
    q = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.0, 2.0, 3.0])
    sig = np.array([0.1, 0.1, 0.1])
    assert template_residual_chi2(q, mu, sig) == 0.0


def test_template_residual_chi2_one_sigma():
    """All joints off by exactly 1σ → χ² = K."""
    q = np.array([1.1, 2.1, 3.1])
    mu = np.array([1.0, 2.0, 3.0])
    sig = np.array([0.1, 0.1, 0.1])
    assert abs(template_residual_chi2(q, mu, sig) - 3.0) < 1e-12


def test_template_residual_chi2_nan_joint_excluded():
    """NaN joint contributes 0 to χ²."""
    q = np.array([1.1, float("nan"), 3.1])
    mu = np.array([1.0, 2.0, 3.0])
    sig = np.array([0.1, 0.1, 0.1])
    # Only joint 0 and 2 — each 1σ → χ² = 2
    assert abs(template_residual_chi2(q, mu, sig) - 2.0) < 1e-12


def test_template_residual_chi2_all_nan_returns_nan():
    q = np.full(3, float("nan"))
    mu = np.zeros(3)
    sig = np.ones(3)
    assert math.isnan(template_residual_chi2(q, mu, sig))


# ─── cadence_jump_detector ───────────────────────────────────────────────


def test_cadence_jump_small_change_no_trigger():
    assert cadence_jump_detector(5.0, 5.1, 0.20) is False
    assert cadence_jump_detector(5.0, 4.5, 0.20) is False  # 10% change


def test_cadence_jump_large_change_triggers():
    assert cadence_jump_detector(5.0, 4.0, 0.20) is True   # 25% change


def test_cadence_jump_zero_prev_no_trigger():
    """ω_prev ≈ 0 (freezing) → no trigger (avoid divide-by-zero)."""
    assert cadence_jump_detector(5.0, 0.0001, 0.20) is False


def test_cadence_jump_non_finite_no_trigger():
    assert cadence_jump_detector(float("nan"), 5.0) is False
    assert cadence_jump_detector(5.0, float("nan")) is False


# ─── vision_loss_detector ────────────────────────────────────────────────


def test_vision_loss_below_threshold_no_trigger():
    assert vision_loss_detector(0.030, max_gap_s=0.060) is False


def test_vision_loss_above_threshold_triggers():
    assert vision_loss_detector(0.100, max_gap_s=0.060) is True


def test_vision_loss_nan_treats_as_lost():
    assert vision_loss_detector(float("nan")) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
