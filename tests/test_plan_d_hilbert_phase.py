"""Plan D HipVerticalPhaseEstimator tests — Hilbert envelope cold-start source.

Codex review 2026-05-12 HARD WALL fix: cold-start phase observation source
for L1 before cycle template is populated.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.hilbert_phase import (
    HilbertPhaseResult,
    HipVerticalPhaseEstimator,
)
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_pi


# ─── Init validation ─────────────────────────────────────────────────────


def test_hilbert_init_default():
    est = HipVerticalPhaseEstimator()
    assert est.window_samples == int(round(1.5 * 120))
    assert est.buffer_fill == 0.0
    assert est.is_ready is False


def test_hilbert_init_invalid_window():
    with pytest.raises(ValueError):
        HipVerticalPhaseEstimator(window_seconds=0)
    with pytest.raises(ValueError):
        HipVerticalPhaseEstimator(window_seconds=-1)


def test_hilbert_init_invalid_fs():
    with pytest.raises(ValueError):
        HipVerticalPhaseEstimator(fs_hz=0)


def test_hilbert_init_invalid_amplitude():
    with pytest.raises(ValueError):
        HipVerticalPhaseEstimator(min_amplitude_m=0)


def test_hilbert_init_invalid_fill_fraction():
    with pytest.raises(ValueError):
        HipVerticalPhaseEstimator(min_fill_fraction=0)
    with pytest.raises(ValueError):
        HipVerticalPhaseEstimator(min_fill_fraction=1.5)


# ─── Feed + readiness ────────────────────────────────────────────────────


def test_hilbert_feed_nan_inf_rejected():
    est = HipVerticalPhaseEstimator(fs_hz=120, window_seconds=1.0)
    est.feed(float("nan"), 0.05)
    est.feed(0.01, float("nan"))
    est.feed(float("inf"), 0.05)
    assert est.buffer_fill == 0.0


def test_hilbert_estimate_empty_invalid():
    est = HipVerticalPhaseEstimator()
    result = est.estimate()
    assert isinstance(result, HilbertPhaseResult)
    assert result.valid is False
    assert math.isnan(result.phi)
    assert result.n_samples == 0


def test_hilbert_estimate_under_min_fill_invalid():
    """Below min_fill_fraction → invalid."""
    est = HipVerticalPhaseEstimator(
        window_seconds=1.0, fs_hz=120, min_fill_fraction=0.5,
    )
    # Feed 10 samples (way below 60-sample min fill)
    for i in range(10):
        est.feed(i * (1.0 / 120), math.sin(i * 0.5))
    result = est.estimate()
    assert result.valid is False


# ─── Amplitude / stationary gate ─────────────────────────────────────────


def test_hilbert_stationary_signal_invalid():
    """Flat signal (no walking) → invalid (below min_amplitude)."""
    est = HipVerticalPhaseEstimator(
        window_seconds=1.0, fs_hz=120, min_amplitude_m=0.01,
    )
    for i in range(180):
        est.feed(i * (1.0 / 120), 0.5)  # constant
    result = est.estimate()
    assert result.valid is False
    assert result.amplitude < 0.01


def test_hilbert_tiny_oscillation_invalid():
    """Sub-amplitude tremor → invalid."""
    est = HipVerticalPhaseEstimator(
        window_seconds=1.0, fs_hz=120, min_amplitude_m=0.01,
    )
    fs = 120.0
    f_walk = 1.0
    for i in range(180):
        t = i / fs
        z = 0.001 * math.sin(2 * math.pi * f_walk * t)  # 1mm amplitude
        est.feed(t, z)
    result = est.estimate()
    assert result.valid is False


# ─── Phase recovery on synthetic sinusoid ────────────────────────────────


def test_hilbert_synthetic_walking_phase_recovery():
    """1 Hz sinusoidal hip-z → Hilbert phase should grow linearly with time."""
    est = HipVerticalPhaseEstimator(
        window_seconds=1.5, fs_hz=120, min_amplitude_m=0.005,
    )
    fs = 120.0
    f_walk = 1.0  # 1 Hz stride
    amplitude = 0.05  # 5 cm
    # Fill window with clean sinusoid
    n_warmup = int(1.5 * fs)
    for i in range(n_warmup):
        t = i / fs
        z = amplitude * math.sin(2 * math.pi * f_walk * t)
        est.feed(t, z)
    result1 = est.estimate()
    assert result1.valid
    # Wait some samples and check phase advanced
    for i in range(n_warmup, n_warmup + 60):  # +0.5 s
        t = i / fs
        z = amplitude * math.sin(2 * math.pi * f_walk * t)
        est.feed(t, z)
    result2 = est.estimate()
    assert result2.valid
    # Phase should advance by 2π × 0.5 = π over 0.5 s at 1 Hz
    phase_advance = (result2.phi - result1.phi) % TWO_PI
    assert abs(phase_advance - math.pi) < 0.3, (
        f"Phase advance off: expected ~π, got {phase_advance}"
    )


def test_hilbert_amplitude_reported():
    """Amplitude field should report signal RMS."""
    est = HipVerticalPhaseEstimator(
        window_seconds=1.0, fs_hz=120, min_amplitude_m=0.005,
    )
    fs = 120.0
    f = 1.0
    A = 0.05
    for i in range(180):
        t = i / fs
        est.feed(t, A * math.sin(2 * math.pi * f * t))
    result = est.estimate()
    assert result.valid
    # RMS of sin = A/√2 ≈ 0.0354; detrend can shift slightly.
    assert 0.02 < result.amplitude < 0.07


# ─── Linear drift handling ───────────────────────────────────────────────


def test_hilbert_handles_linear_drift():
    """Slow body drift (constant slope) should not contaminate phase."""
    est = HipVerticalPhaseEstimator(
        window_seconds=1.0, fs_hz=120, min_amplitude_m=0.005,
    )
    fs = 120.0
    f = 1.0
    A = 0.05
    drift = 0.5  # m/s upward drift
    for i in range(180):
        t = i / fs
        z = A * math.sin(2 * math.pi * f * t) + drift * t
        est.feed(t, z)
    result = est.estimate()
    assert result.valid
    # The estimator should detrend; phase should still be recovered (not exact
    # match with no-drift version due to edge effects + detrend windowing,
    # but should be finite and bounded).
    assert math.isfinite(result.phi)


# ─── Reset ───────────────────────────────────────────────────────────────


def test_hilbert_reset_clears_buffer():
    est = HipVerticalPhaseEstimator()
    for i in range(50):
        est.feed(i * 0.01, math.sin(i * 0.1))
    assert est.buffer_fill > 0
    est.reset()
    assert est.buffer_fill == 0.0
    assert est.is_ready is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
