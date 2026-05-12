"""Plan D utils tests — phase wrap, dt validation, Joseph covariance, bin map."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.utils import (
    TWO_PI,
    bin_of_phase,
    joseph_update,
    validate_dt,
    wrap_to_2pi,
    wrap_to_pi,
)


# ─── wrap_to_pi ──────────────────────────────────────────────────────────


def test_wrap_to_pi_identity_in_range():
    """Values already in [-π, π] should be unchanged (modulo floating roundoff)."""
    for x in [0.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0]:
        result = wrap_to_pi(x)
        assert abs(result - x) < 1e-12, f"wrap_to_pi({x}) = {result}, expected {x}"


def test_wrap_to_pi_basic_boundaries():
    """Edges and just-outside [-π, π]."""
    # π wraps to -π (left-closed convention)
    assert abs(wrap_to_pi(math.pi) - (-math.pi)) < 1e-12 or abs(wrap_to_pi(math.pi) - math.pi) < 1e-12
    # 2π should wrap to 0
    assert abs(wrap_to_pi(TWO_PI)) < 1e-12
    # 3π wraps to -π or π (boundary)
    r = wrap_to_pi(3 * math.pi)
    assert abs(abs(r) - math.pi) < 1e-12


def test_wrap_to_pi_negative():
    """Negative angles."""
    assert abs(wrap_to_pi(-3 * math.pi / 2) - math.pi / 2) < 1e-12
    assert abs(wrap_to_pi(-math.pi / 2) - (-math.pi / 2)) < 1e-12


def test_wrap_to_pi_innovation_use_case():
    """Critical use case: shortest difference across 2π boundary.

    If x_pred ≈ 0 and z ≈ 2π, innovation should be tiny (~0),
    NOT 2π (which would catastrophically drive EKF state).
    """
    x_pred = 0.01
    z = TWO_PI - 0.01
    raw_diff = z - x_pred  # = 2π - 0.02 ≈ 6.26
    innov = wrap_to_pi(raw_diff)
    # Innovation should be small negative (z is "just behind" x_pred wrapped)
    assert abs(innov) < 0.05, f"Innovation across 2π not wrapped: {innov}"


def test_wrap_to_pi_idempotent():
    """wrap(wrap(x)) == wrap(x) for many values."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        x = rng.uniform(-20.0, 20.0)
        once = wrap_to_pi(x)
        twice = wrap_to_pi(once)
        assert abs(once - twice) < 1e-12


def test_wrap_to_pi_vectorized():
    """Numpy array input → array output."""
    arr = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi, -math.pi / 2])
    result = wrap_to_pi(arr)
    assert result.shape == arr.shape
    assert np.all(np.abs(result) <= math.pi + 1e-12)


def test_wrap_to_pi_nan_passthrough():
    """NaN should pass through (don't silently zero)."""
    result = wrap_to_pi(float("nan"))
    assert math.isnan(result)


# ─── wrap_to_2pi ─────────────────────────────────────────────────────────


def test_wrap_to_2pi_basic():
    assert abs(wrap_to_2pi(0.0)) < 1e-12
    assert abs(wrap_to_2pi(math.pi) - math.pi) < 1e-12
    assert abs(wrap_to_2pi(TWO_PI)) < 1e-12  # 2π → 0
    assert abs(wrap_to_2pi(3 * math.pi) - math.pi) < 1e-12


def test_wrap_to_2pi_negative():
    assert abs(wrap_to_2pi(-math.pi / 2) - (3 * math.pi / 2)) < 1e-12
    assert abs(wrap_to_2pi(-TWO_PI)) < 1e-12


def test_wrap_to_2pi_range():
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.uniform(-100.0, 100.0)
        r = float(wrap_to_2pi(x))
        assert 0.0 <= r < TWO_PI + 1e-12


# ─── validate_dt ─────────────────────────────────────────────────────────


def test_validate_dt_positive():
    assert validate_dt(0.001) is True
    assert validate_dt(0.05) is True
    assert validate_dt(0.1) is True


def test_validate_dt_zero_rejected():
    assert validate_dt(0.0) is False


def test_validate_dt_negative_rejected():
    assert validate_dt(-0.001) is False
    assert validate_dt(-1.0) is False


def test_validate_dt_too_large_rejected():
    assert validate_dt(1.0) is False  # > 0.5 default
    assert validate_dt(100.0) is False


def test_validate_dt_custom_max():
    assert validate_dt(0.8, max_dt_s=1.0) is True
    assert validate_dt(1.2, max_dt_s=1.0) is False


def test_validate_dt_nan_inf_rejected():
    assert validate_dt(float("nan")) is False
    assert validate_dt(float("inf")) is False
    assert validate_dt(float("-inf")) is False


def test_validate_dt_type_safety():
    assert validate_dt("0.01") is False  # type: ignore[arg-type]
    assert validate_dt(None) is False  # type: ignore[arg-type]


# ─── joseph_update ───────────────────────────────────────────────────────


def test_joseph_update_simple_scalar_like():
    """1-state Kalman as 1x1 matrices."""
    P = np.array([[1.0]])
    H = np.array([[1.0]])
    R = np.array([[0.5]])
    # K = P H^T / (H P H^T + R) = 1/1.5 = 0.6667
    K = np.array([[2.0 / 3.0]])
    P_post = joseph_update(P, K, H, R)
    # Posterior variance = (1 - K) P = 0.333
    # Also = P R / (P + R) = 1 × 0.5 / 1.5 = 0.333
    assert abs(P_post[0, 0] - 1.0 / 3.0) < 1e-9


def test_joseph_update_keeps_psd_random():
    """Random K, H, R, P should always yield PSD result."""
    rng = np.random.default_rng(7)
    n, m = 3, 2
    for trial in range(20):
        # Random PSD P
        A = rng.standard_normal((n, n))
        P = A @ A.T + np.eye(n) * 1e-3
        H = rng.standard_normal((m, n))
        # Random PSD R
        B = rng.standard_normal((m, m))
        R = B @ B.T + np.eye(m) * 1e-3
        K = rng.standard_normal((n, m))
        P_post = joseph_update(P, K, H, R)
        # Symmetry
        assert np.allclose(P_post, P_post.T, atol=1e-10), f"Trial {trial}: P_post not symmetric"
        # PSD: all eigenvalues ≥ 0 (allow small negative from roundoff)
        eigvals = np.linalg.eigvalsh(P_post)
        assert np.all(eigvals > -1e-9), f"Trial {trial}: negative eigenvalue {eigvals.min()}"


def test_joseph_update_zero_gain_returns_prior():
    """K = 0 means no measurement applied → P_post = P_prior."""
    P = np.diag([1.0, 2.0, 3.0])
    K = np.zeros((3, 2))
    H = np.ones((2, 3))
    R = np.eye(2)
    P_post = joseph_update(P, K, H, R)
    assert np.allclose(P_post, P)


def test_joseph_update_dtype_float64():
    """Output dtype should be float64 even with float32 input (numerical safety)."""
    P = np.eye(2, dtype=np.float64)
    K = np.array([[0.5], [0.1]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    R = np.array([[1.0]], dtype=np.float64)
    P_post = joseph_update(P, K, H, R)
    assert P_post.dtype == np.float64


# ─── bin_of_phase ────────────────────────────────────────────────────────


def test_bin_of_phase_zero():
    assert bin_of_phase(0.0, 128) == 0


def test_bin_of_phase_just_below_2pi():
    eps = 1e-9
    assert bin_of_phase(TWO_PI - eps, 128) == 127


def test_bin_of_phase_exactly_2pi_wraps():
    """2π wraps to 0 (same as phase 0)."""
    assert bin_of_phase(TWO_PI, 128) == 0


def test_bin_of_phase_quadrants():
    """π/2 → bin 32, π → bin 64, 3π/2 → bin 96 (n_bins=128)."""
    assert bin_of_phase(math.pi / 2, 128) == 32
    assert bin_of_phase(math.pi, 128) == 64
    assert bin_of_phase(3 * math.pi / 2, 128) == 96


def test_bin_of_phase_negative_wraps():
    """-π/2 should wrap to 3π/2 → bin 96."""
    assert bin_of_phase(-math.pi / 2, 128) == 96


def test_bin_of_phase_different_n_bins():
    assert bin_of_phase(math.pi, 64) == 32
    assert bin_of_phase(math.pi, 256) == 128


def test_bin_of_phase_roundoff_safe_at_all_centers():
    """Regression: bin_i × 2π/128 must map back to bin_i for all i.

    Float arithmetic can cause 22 × 2π/128 × 128 / 2π = 21.999... → bin 21.
    bin_of_phase must be robust to this systematic roundoff.
    """
    n_bins = 128
    for bin_i in range(n_bins):
        phi = bin_i * TWO_PI / n_bins
        got = bin_of_phase(phi, n_bins)
        assert got == bin_i, f"bin_i={bin_i}: phi={phi}, got bin={got}"


def test_bin_of_phase_roundoff_safe_n_bins_256():
    """Same regression at higher resolution."""
    n_bins = 256
    for bin_i in range(n_bins):
        phi = bin_i * TWO_PI / n_bins
        assert bin_of_phase(phi, n_bins) == bin_i, f"bin_i={bin_i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
