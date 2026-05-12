"""Hilbert-envelope phase estimator — cold-start phase source for Plan D.

Codex review 2026-05-12 identified a HARD WALL: Plan D L1/L2 need φ_meas,
but CrossCorrPhaseEstimator cannot produce φ_meas until the cycle template
is touched, and CycleTemplate.update() needs φ. Circular bootstrap.

This module is the **external cold-start phase source** that breaks the loop.

Theory:
    During walking, hip vertical position (z_hip in world frame) oscillates
    quasi-sinusoidally with the stride period. The Hilbert transform of a
    sufficiently band-limited signal yields its analytic signal:
        s_analytic(t) = s(t) + j × H{s}(t)
        instantaneous_phase = angle(s_analytic(t))

    For walking, instantaneous_phase advances ~2π per stride. This is a
    cycle-template-independent phase observation suitable for cold-start.

Operation:
    1. Caller feeds (t, z_hip) per frame.
    2. Estimator maintains a sliding window (default 1.5 s ~ 1-2 strides).
    3. estimate() detrends + Hilbert + returns instantaneous phase at latest sample.
    4. If amplitude too low (stationary patient) or window too short → returns None.

Boundary effects:
    Hilbert transform has end-effect distortion (~half-window). For online
    use, the latest sample is the most distorted. Codex IMPROVE: use a
    longer window than strictly needed and only emit phase if the latest
    sample is sufficiently inside it (e.g., last 5% buffer trustworthy).

Convention:
    Returned φ ∈ [0, 2π). Absolute alignment with template HS convention
    (HS_L at φ=0, HS_R at φ=π) is NOT guaranteed — Hilbert phase has an
    arbitrary offset depending on starting sample. The cascade is expected
    to use this only for cold-start (L1) where absolute phase is unimportant
    (only relative consistency for ω estimation).

Real-time:
    - O(n_window × log n_window) per estimate() call due to FFT in scipy.
    - For n_window=128 at 120 Hz, ~50 µs on Jetson Orin (negligible).
    - In production C++, use a FIR Hilbert filter for deterministic timing.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from perception.plan_d_prototype.utils import wrap_to_2pi


@dataclass
class HilbertPhaseResult:
    """Output of HipVerticalPhaseEstimator.estimate().

    Attributes:
        phi: instantaneous phase (rad in [0, 2π)) — NaN if not estimable.
        amplitude: signal RMS in window (m) — for caller's freezing-gait gate.
        valid: True if phase was computed (buffer full, amplitude > floor).
        n_samples: window length used.
    """

    phi: float
    amplitude: float
    valid: bool
    n_samples: int


class HipVerticalPhaseEstimator:
    """Hilbert-envelope phase estimator on hip vertical position signal.

    Public API:
        feed(t_s, z_hip_m) → None
        estimate() → HilbertPhaseResult
        reset() → None

    Designed for L1 cold-start. Switch to CrossCorrPhaseEstimator after
    cycle template is touched and stride count ≥ 3 (PredictorCascade duty).
    """

    def __init__(
        self,
        window_seconds: float = 1.5,
        fs_hz: float = 120.0,
        min_amplitude_m: float = 0.01,
        min_fill_fraction: float = 0.5,
    ) -> None:
        """
        Args:
            window_seconds: sliding window length (default 1.5 s ≈ 1.5 strides).
                Longer window = better Hilbert accuracy but more lag.
            fs_hz: nominal sampling rate (used to size the buffer).
            min_amplitude_m: minimum signal std (RMS) to consider walking.
                Below this → freezing or stationary → return invalid.
            min_fill_fraction: estimate() requires at least this fraction of
                the buffer filled. Default 0.5 = half-window before first phase.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if fs_hz <= 0:
            raise ValueError("fs_hz must be > 0")
        if min_amplitude_m <= 0:
            raise ValueError("min_amplitude_m must be > 0")
        if not (0.0 < min_fill_fraction <= 1.0):
            raise ValueError("min_fill_fraction must be in (0, 1]")
        self._window_samples = max(8, int(round(window_seconds * fs_hz)))
        self._min_fill = int(self._window_samples * min_fill_fraction)
        self._fs_hz = float(fs_hz)
        self._min_amplitude = float(min_amplitude_m)
        self._buffer: deque = deque(maxlen=self._window_samples)
        self._t_buffer: deque = deque(maxlen=self._window_samples)

    # ─── Feed ────────────────────────────────────────────────────────────

    def feed(self, t_s: float, z_hip_m: float) -> None:
        """Append a sample. NaN / inf rejected silently."""
        if not (math.isfinite(t_s) and math.isfinite(z_hip_m)):
            return
        self._buffer.append(float(z_hip_m))
        self._t_buffer.append(float(t_s))

    # ─── Estimate ────────────────────────────────────────────────────────

    def estimate(self) -> HilbertPhaseResult:
        """Compute instantaneous Hilbert phase at the latest sample.

        Returns invalid HilbertPhaseResult if:
            - Buffer not yet at min_fill_fraction
            - Signal std below min_amplitude (freezing / stationary)
            - Numerical failure (rare)
        """
        n = len(self._buffer)
        if n < self._min_fill:
            return HilbertPhaseResult(
                phi=float("nan"),
                amplitude=0.0,
                valid=False,
                n_samples=n,
            )
        signal = np.asarray(self._buffer, dtype=np.float64)
        # Detrend (remove DC + slope to handle slow body drift)
        signal = signal - signal.mean()
        # Linear detrend
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, signal, 1)[0]
        signal = signal - slope * x
        rms = float(np.sqrt(np.mean(signal * signal)))
        if rms < self._min_amplitude:
            return HilbertPhaseResult(
                phi=float("nan"),
                amplitude=rms,
                valid=False,
                n_samples=n,
            )
        # Hilbert (deferred import — scipy is heavy on import)
        try:
            from scipy.signal import hilbert as _hilbert
        except ImportError:
            return HilbertPhaseResult(
                phi=float("nan"),
                amplitude=rms,
                valid=False,
                n_samples=n,
            )
        analytic = _hilbert(signal)
        # Instantaneous phase at latest sample (the one we control on)
        phi_raw = float(np.angle(analytic[-1]))
        phi = float(wrap_to_2pi(phi_raw))
        if not math.isfinite(phi):
            return HilbertPhaseResult(
                phi=float("nan"),
                amplitude=rms,
                valid=False,
                n_samples=n,
            )
        return HilbertPhaseResult(
            phi=phi,
            amplitude=rms,
            valid=True,
            n_samples=n,
        )

    # ─── Status ──────────────────────────────────────────────────────────

    @property
    def buffer_fill(self) -> float:
        """Fraction of buffer filled (0..1)."""
        return float(len(self._buffer)) / self._window_samples

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= self._min_fill

    @property
    def window_samples(self) -> int:
        return self._window_samples

    def reset(self) -> None:
        self._buffer.clear()
        self._t_buffer.clear()
