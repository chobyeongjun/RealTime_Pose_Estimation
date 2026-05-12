"""Plan D divergence detection — innovation gates + chi² + cadence-jump detector.

Codex review 2026-05-12 adjustments applied:
    - Chi² thresholds BY DOF (NOT hardcoded 9.0):
        1-DOF (phase observation):  χ² > 6.64  (99% confidence)
        3-DOF:                        χ² > 11.34
        6-DOF (full joint vector):  χ² > 16.81
    - Mahalanobis solve via np.linalg.solve (LDLT-equivalent in C++ Eigen),
      NOT explicit `np.linalg.inv` (numerically poison).
    - Template residual gate uses per-joint σ for chi-square test.

Failure mode catalog (plan_d_predictor_spec.md §4):
    - Cycle assumption breaks (freezing, shuffling, turning, start/stop)
    - Template residual spike (3σ)
    - Phase innovation > 25% cycle
    - Cadence jump > 20% in one stride
    - Missing core joints > 60 ms gap
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


# Chi-square upper critical values at α=0.01 (99% confidence interval).
# These are the thresholds above which an innovation is "too large".
# Values from chi-square distribution inverse CDF at 0.99 quantile.
CHI2_THRESHOLD_99: Dict[int, float] = {
    1: 6.6349,
    2: 9.2103,
    3: 11.3449,
    4: 13.2767,
    5: 15.0863,
    6: 16.8119,
    8: 20.0902,
    10: 23.2093,
}

# Default tighter threshold for clinical safety (99.9% confidence).
CHI2_THRESHOLD_999: Dict[int, float] = {
    1: 10.8276,
    2: 13.8155,
    3: 16.2662,
    4: 18.4668,
    5: 20.5150,
    6: 22.4577,
    8: 26.1245,
    10: 29.5883,
}


def chi2_threshold(dof: int, confidence: float = 0.99) -> float:
    """Lookup χ² critical value for given degrees of freedom + confidence.

    Args:
        dof: integer degrees of freedom (1..10 in table).
        confidence: 0.99 or 0.999. Other values raise.

    Raises:
        KeyError: if dof not in table.
        ValueError: if confidence not 0.99 or 0.999.
    """
    if confidence == 0.99:
        return CHI2_THRESHOLD_99[int(dof)]
    if confidence == 0.999:
        return CHI2_THRESHOLD_999[int(dof)]
    raise ValueError(f"confidence must be 0.99 or 0.999, got {confidence}")


def mahalanobis_chi2(
    innovation: np.ndarray,
    S: np.ndarray,
) -> float:
    """Compute innovation Mahalanobis χ² = yᵀ S⁻¹ y.

    Uses np.linalg.solve (LDLT-equivalent), NOT explicit inverse.

    Args:
        innovation: (m,) innovation vector.
        S: (m, m) innovation covariance.

    Returns:
        χ² value. NaN if S is singular or innovation has NaN.
    """
    inn = np.asarray(innovation, dtype=np.float64).ravel()
    if not np.all(np.isfinite(inn)):
        return float("nan")
    if S.shape[0] != S.shape[1] or S.shape[0] != inn.shape[0]:
        raise ValueError(
            f"shape mismatch: S {S.shape}, innov {inn.shape}"
        )
    try:
        x = np.linalg.solve(S, inn)
    except np.linalg.LinAlgError:
        return float("nan")
    val = float(inn @ x)
    if not math.isfinite(val) or val < 0.0:
        return float("nan")
    return val


def innovation_gate(
    chi2: float,
    dof: int,
    confidence: float = 0.99,
) -> bool:
    """Return True if chi2 EXCEEDS the threshold for given DOF (= divergence).

    Args:
        chi2: Mahalanobis χ² (from mahalanobis_chi2 or EKF update).
        dof: measurement DOF.
        confidence: 0.99 (default) or 0.999.

    Returns:
        True → diverged (caller should fall back). False → accept.
    """
    if not math.isfinite(chi2) or chi2 < 0.0:
        return True  # Pathological — treat as diverged
    threshold = chi2_threshold(dof, confidence=confidence)
    return chi2 > threshold


def template_residual_chi2(
    q: np.ndarray,
    mu_at_phi: np.ndarray,
    sigma_per_joint: np.ndarray,
    sigma_floor: float = 0.01,
) -> float:
    """Per-joint residual chi-square (used by L3 divergence check).

    χ² = Σ_k ((q_k - μ_k) / σ_k)²

    NaN entries in q are excluded (each subtracts 1 from effective DOF).

    Args:
        q: (K,) observation.
        mu_at_phi: (K,) template prediction.
        sigma_per_joint: (K,) per-joint σ (rad).
        sigma_floor: minimum σ for numerical safety.

    Returns:
        χ². NaN if all joints invalid.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    mu = np.asarray(mu_at_phi, dtype=np.float64)
    sig = np.asarray(sigma_per_joint, dtype=np.float64)
    if not (q_arr.shape == mu.shape == sig.shape):
        raise ValueError("shape mismatch in template_residual_chi2")
    valid = np.isfinite(q_arr) & np.isfinite(mu)
    if not valid.any():
        return float("nan")
    sig_safe = np.where(np.isfinite(sig), sig, sigma_floor)
    sig_safe = np.maximum(sig_safe, sigma_floor)
    diff = (q_arr - mu) / sig_safe
    return float(np.sum(diff[valid] ** 2))


def cadence_jump_detector(
    omega_now: float,
    omega_prev: float,
    relative_threshold: float = 0.20,
) -> bool:
    """Return True if cadence changed by > threshold (fractional).

    20% jump in 1 stride (plan_d_predictor_spec.md §4.2 default).

    Args:
        omega_now: current ω estimate (rad/s).
        omega_prev: previous ω estimate (one stride earlier).
        relative_threshold: fractional cadence change threshold.

    Returns:
        True if jump exceeds threshold; False if smooth.
        False also if either input is non-finite or omega_prev ≈ 0.
    """
    if not (math.isfinite(omega_now) and math.isfinite(omega_prev)):
        return False
    if abs(omega_prev) < 1e-3:
        # Resuming from freeze; cannot compute ratio
        return False
    ratio = abs(omega_now - omega_prev) / abs(omega_prev)
    return ratio > relative_threshold


def vision_loss_detector(
    dt_since_last_valid_s: float,
    max_gap_s: float = 0.060,
) -> bool:
    """Return True if vision pose has been missing for too long.

    Default 60 ms = "missed detections >60ms gap" (plan_d_predictor_spec.md §4.2).
    Triggers fallback / pretension watchdog upstream.

    Args:
        dt_since_last_valid_s: seconds since last valid pose.
        max_gap_s: threshold.

    Returns:
        True if vision considered lost.
    """
    if not math.isfinite(dt_since_last_valid_s):
        return True
    return dt_since_last_valid_s > max_gap_s
