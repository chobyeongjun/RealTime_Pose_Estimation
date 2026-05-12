"""Plan D utility helpers — phase wrap, dt validation, Joseph covariance.

All floats are np.float64. EKF condition number issues observed with float32
in similar prior-art implementations (Thatte 2015 errata).
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI: float = 2.0 * math.pi
"""2π constant in float64. Use this everywhere for consistency."""


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle (rad) to [-π, π].

    For innovation: shortest signed angular difference.
    Idempotent: wrap_to_pi(wrap_to_pi(x)) == wrap_to_pi(x).

    Args:
        angle: scalar or numpy array (rad)

    Returns:
        Same shape as input. NaN passes through.
    """
    # ((x + π) mod 2π) - π gives [-π, π) consistently
    # Use np.mod for vectorization + NaN propagation
    return np.mod(angle + math.pi, TWO_PI) - math.pi


def wrap_to_2pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle (rad) to [0, 2π).

    For state representation: phase φ always in canonical range.

    Args:
        angle: scalar or numpy array (rad)

    Returns:
        Same shape as input. NaN passes through.
    """
    return np.mod(angle, TWO_PI)


def validate_dt(dt: float, max_dt_s: float = 0.5) -> bool:
    """Check dt is positive, finite, and reasonable.

    Rejects:
        - dt <= 0 (clock went backward, or first sample)
        - dt > max_dt_s (stale frame, missed multiple cycles)
        - NaN, inf

    Args:
        dt: elapsed seconds
        max_dt_s: upper bound (default 0.5s = 2 strides worst case)

    Returns:
        True if dt is usable for EKF integration.
    """
    if not isinstance(dt, (int, float, np.floating)):
        return False
    if not math.isfinite(dt):
        return False
    if dt <= 0.0:
        return False
    if dt > max_dt_s:
        return False
    return True


def joseph_update(
    P_prior: np.ndarray,
    K: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Joseph form covariance update — positive-definite-safe.

    Computes P_post = (I - K H) P_prior (I - K H)^T + K R K^T

    This is numerically superior to the naive P_post = (I - K H) P_prior:
    - Symmetric by construction
    - Positive semi-definite even with finite-precision K
    - Robust under near-singular S

    Args:
        P_prior: (n, n) prior covariance
        K: (n, m) Kalman gain
        H: (m, n) measurement Jacobian
        R: (m, m) measurement noise covariance

    Returns:
        P_post: (n, n) posterior covariance (symmetric PSD)
    """
    n = P_prior.shape[0]
    I_KH = np.eye(n, dtype=np.float64) - K @ H
    P_post = I_KH @ P_prior @ I_KH.T + K @ R @ K.T
    # Force exact symmetry (round-off can introduce tiny asymmetry)
    return 0.5 * (P_post + P_post.T)


def bin_of_phase(phi: float, n_bins: int) -> int:
    """Map phase φ ∈ [0, 2π) to bin index ∈ [0, n_bins).

    Boundary handling:
        bin_of_phase(0.0, 128) = 0
        bin_of_phase(2π - ε, 128) = 127
        bin_of_phase(2π, 128) = 0 (wrap)
        bin_of_phase(bin_i × 2π/n_bins, n_bins) = bin_i (roundoff-safe)

    The roundoff-safe property is critical: e.g. for n_bins=128,
        bin_i=22: phi = 22*2π/128, phi*128/2π = 21.999999999999996 (float)
    Naive int() would map this to bin 21 instead of 22.

    We add a small bias (1e-9, ~2e-8 of a bin width at n_bins=128) before
    floor() to correct the roundoff while preserving the (2π - tiny_eps) → 127
    boundary contract.

    Args:
        phi: phase in radians (any range; wrapped internally)
        n_bins: number of bins (typically 128)

    Returns:
        Integer bin index in [0, n_bins).
    """
    phi_wrapped = float(wrap_to_2pi(phi))
    idx_float = phi_wrapped * n_bins / TWO_PI
    # Bias floor by 1e-9 to round-up bin centers that landed at integer-eps.
    idx = int(math.floor(idx_float + 1e-9))
    if idx >= n_bins:
        idx = 0
    if idx < 0:
        idx = 0
    return idx
