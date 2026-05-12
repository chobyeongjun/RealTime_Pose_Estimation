"""Plan D Level 1 EKF — Constant velocity gait phase tracker.

State:
    x = [φ, ω]ᵀ
      φ: gait phase (rad, wrapped to [0, 2π))
      ω: cadence (rad/s; typical walking ~3-7 rad/s for 0.5-1.1 Hz stride)

Dynamics (continuous-time, ω driven by white noise):
    φ̇ = ω
    ω̇ = w(t),   w(t) ~ N(0, q_ω)   (white noise spectral density q_ω)

Discrete form (Δt > 0):
    φ_{k+1} = φ_k + ω_k Δt                   (mod 2π)
    ω_{k+1} = ω_k
    F = [[1, Δt], [0, 1]]

Correct discrete process noise covariance (Codex 2026-05-12 fix):
    For F_c = [[0,1],[0,0]], G_c = [0,1]ᵀ, Q_c = q_ω scalar:
        Q_d = ∫₀^Δt e^{F_c τ} G_c Q_c G_cᵀ e^{F_c τ}ᵀ dτ
            = q_ω × [[Δt³/3, Δt²/2], [Δt²/2, Δt]]
    Plus optional independent φ noise (e.g., quantization):
        Q_d_total = Q_d + q_φ × Δt × [[1, 0], [0, 0]]

Measurement:
    z = φ_observed (rad), from CrossCorrPhaseEstimator
    h(x) = [1, 0] x = φ
    Innovation uses shortest-angle wrap to [-π, π].

References:
    docs/lessons/plan_d_predictor_spec.md §3 (cold-start cascade)
    docs/lessons/codex_review_phase1_2026_05_12.md (NEEDS_FIX #1, 2, 3)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from perception.plan_d_prototype.utils import (
    TWO_PI,
    joseph_update,
    validate_dt,
    wrap_to_2pi,
    wrap_to_pi,
)


# Sensible default cadence: 4.0 rad/s ≈ 0.64 Hz stride (slow walk).
# Caller can override via initial_omega for session calibration.
_DEFAULT_INITIAL_OMEGA = 4.0

# Cold-start covariance: σ = 1 rad (~57°) for φ, σ = 2 rad/s for ω.
# π² (Phase 1 original) was too aggressive — first observation dominated.
# Codex 2026-05-12: "fix the comment and gate first update".
_DEFAULT_INITIAL_P_PHI = 1.0
_DEFAULT_INITIAL_P_OMEGA = 4.0


class PredictStatus(Enum):
    """Why predict() returned without integrating."""

    OK = "ok"                            # state evolved normally
    INITIAL = "initial"                  # first call — only t_last stamped
    DT_NON_POSITIVE = "dt_non_positive"  # clock went backward
    DT_TOO_LARGE = "dt_too_large"        # > max_dt_s — STALE / WATCHDOG
    DT_NOT_FINITE = "dt_not_finite"      # NaN / inf
    T_NOT_FINITE = "t_not_finite"        # caller passed bad time


@dataclass
class EKFL1State:
    """L1 state snapshot — exposes φ, ω with named accessors."""

    x: np.ndarray = field(
        default_factory=lambda: np.array([0.0, _DEFAULT_INITIAL_OMEGA], dtype=np.float64)
    )
    P: np.ndarray = field(
        default_factory=lambda: np.diag(
            [_DEFAULT_INITIAL_P_PHI, _DEFAULT_INITIAL_P_OMEGA]
        ).astype(np.float64)
    )
    t_last: Optional[float] = None

    @property
    def phi(self) -> float:
        return float(self.x[0])

    @property
    def omega(self) -> float:
        return float(self.x[1])

    @property
    def sigma_phi(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], 0.0)))

    @property
    def sigma_omega(self) -> float:
        return float(math.sqrt(max(self.P[1, 1], 0.0)))


class EKFL1:
    """Level 1 EKF — Constant velocity (φ, ω).

    Public API:
        predict(t_now) -> PredictStatus
        update(z_phi, R_override=None) -> bool   (True if applied)
        predict_ahead(tau_s) -> (phi, sigma_phi, omega, sigma_omega)
        reset(initial_omega=None)
        condition_number_P() -> float

    Real-time safety:
        - All ops are O(1) (2-state, no allocation in steady state).
        - No I/O, no logging in hot path.
        - All arrays float64.
    """

    def __init__(
        self,
        process_noise_phi: float = 1e-6,
        process_noise_omega: float = 4e-2,
        measurement_noise: float = 0.05,
        initial_omega: float = _DEFAULT_INITIAL_OMEGA,
        initial_P_phi: float = _DEFAULT_INITIAL_P_PHI,
        initial_P_omega: float = _DEFAULT_INITIAL_P_OMEGA,
        max_dt_s: float = 0.5,
    ) -> None:
        """Initialize L1 EKF.

        Args:
            process_noise_phi: σ²_φ (rad² / s) — independent φ noise (quantization).
                Small (1e-6) since φ is integrated state.
            process_noise_omega: q_ω ((rad/s)² / s) — cadence white-noise spectral density.
                4e-2 corresponds to σ_ω growth ≈ 0.2 rad/s per √s.
            measurement_noise: σ²_z (rad²) — observed phase noise variance.
                0.05 rad² = σ ≈ 13° per observation. Codex IMPROVE: calibrate from
                template-match cost curvature when available.
            initial_omega: prior cadence (rad/s). 4.0 ≈ 0.64 Hz slow walk.
                Codex IMPROVE: expose as session calibration in production caller.
            initial_P_phi: σ²_φ at cold-start (default 1.0 rad² = σ ≈ 57°).
            initial_P_omega: σ²_ω at cold-start (default 4.0 (rad/s)² = σ ≈ 2 rad/s).
            max_dt_s: dt validation upper bound. Above this, predict() returns
                DT_TOO_LARGE so the caller can trigger watchdog/fallback.
        """
        # Two separable spectral densities (kept as scalars for fast hot-path math).
        self._q_phi = float(process_noise_phi)
        self._q_omega = float(process_noise_omega)
        self._R = np.array([[float(measurement_noise)]], dtype=np.float64)
        self._H = np.array([[1.0, 0.0]], dtype=np.float64)
        self._initial_omega = float(initial_omega)
        self._initial_P_phi = float(initial_P_phi)
        self._initial_P_omega = float(initial_P_omega)
        self._max_dt_s = float(max_dt_s)
        self.state = EKFL1State(
            x=np.array([0.0, initial_omega], dtype=np.float64),
            P=np.diag([initial_P_phi, initial_P_omega]).astype(np.float64),
            t_last=None,
        )

    # ─── Predict ─────────────────────────────────────────────────────────

    def predict(self, t_now: float) -> PredictStatus:
        """Time-update step. Returns status so caller can drive watchdog.

        First call only stamps t_last (no state evolution; status INITIAL).
        Invalid dt → returns DT_* status, stamps t_last to avoid huge next dt.
        Codex NEEDS_FIX: callers MUST react to DT_TOO_LARGE (stale vision).
        """
        if not math.isfinite(t_now):
            return PredictStatus.T_NOT_FINITE
        if self.state.t_last is None:
            self.state.t_last = float(t_now)
            return PredictStatus.INITIAL
        dt = float(t_now) - self.state.t_last
        if not math.isfinite(dt):
            self.state.t_last = float(t_now)
            return PredictStatus.DT_NOT_FINITE
        if dt <= 0.0:
            self.state.t_last = float(t_now)
            return PredictStatus.DT_NON_POSITIVE
        if dt > self._max_dt_s:
            self.state.t_last = float(t_now)
            return PredictStatus.DT_TOO_LARGE
        # ─── Normal integration ──────────────────────────────────────────
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        x_pred = F @ self.state.x
        x_pred[0] = float(wrap_to_2pi(x_pred[0]))
        # Correct continuous-discrete Q (Codex NEEDS_FIX #1)
        Q_d = self._build_Qd(dt)
        P_pred = F @ self.state.P @ F.T + Q_d
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.state.x = x_pred
        self.state.P = P_pred
        self.state.t_last = float(t_now)
        return PredictStatus.OK

    def _build_Qd(self, dt: float) -> np.ndarray:
        """Integrated discrete process-noise covariance.

        Continuous-time: ω driven by white noise of spectral density q_ω,
        plus optional independent φ noise q_φ (e.g., quantization).

        Q_d = q_ω × [[dt³/3, dt²/2], [dt²/2, dt]] + q_φ × dt × [[1,0],[0,0]]
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        return np.array(
            [
                [self._q_omega * dt3 / 3.0 + self._q_phi * dt,
                 self._q_omega * dt2 / 2.0],
                [self._q_omega * dt2 / 2.0,
                 self._q_omega * dt],
            ],
            dtype=np.float64,
        )

    # ─── Update ──────────────────────────────────────────────────────────

    def update(
        self,
        z_phi: float,
        R_override: Optional[float] = None,
    ) -> bool:
        """Measurement update. Returns True if applied, False otherwise.

        Codex NEEDS_FIX #2: R_override must be validated.
        Codex NEEDS_FIX #3: large initial P_phi addressed via reduced default,
            so no special-case first-update gate needed. Caller is expected
            to validate observation upstream (use ambiguity_ratio).

        Skip conditions (return False):
            - z_phi not finite
            - R_override not finite, or ≤ 0
            - innovation covariance S not positive finite (pathological state)
        """
        if not math.isfinite(z_phi):
            return False
        if R_override is None:
            R = self._R
        else:
            R_val = float(R_override)
            if not math.isfinite(R_val) or R_val <= 0.0:
                return False
            R = np.array([[R_val]], dtype=np.float64)
        # Innovation: shortest angular distance
        z_pred = float((self._H @ self.state.x)[0])
        innov = float(wrap_to_pi(z_phi - z_pred))
        # Innovation covariance and Kalman gain
        S = self._H @ self.state.P @ self._H.T + R     # (1, 1)
        S_scalar = float(S[0, 0])
        if S_scalar <= 0.0 or not math.isfinite(S_scalar):
            return False
        K = (self.state.P @ self._H.T) / S_scalar      # (2, 1)
        # State update
        x_new = self.state.x + K.flatten() * innov
        x_new[0] = float(wrap_to_2pi(x_new[0]))
        # Joseph form covariance update
        P_new = joseph_update(self.state.P, K, self._H, R)
        self.state.x = x_new
        self.state.P = P_new
        return True

    # ─── Predict ahead (forecast at τ seconds) ───────────────────────────

    def predict_ahead(self, tau_s: float) -> Tuple[float, float, float, float]:
        """Forecast state at τ seconds ahead of the last predict() time.

        Pure read-only forecast — does NOT mutate filter state.

        Args:
            tau_s: lookahead in seconds (>= 0; τ < 0 clamped to 0).

        Returns:
            (phi_pred, sigma_phi, omega_pred, sigma_omega)
            phi_pred in [0, 2π), sigmas non-negative.
        """
        if not math.isfinite(tau_s) or tau_s < 0:
            tau_s = 0.0
        F_tau = np.array([[1.0, tau_s], [0.0, 1.0]], dtype=np.float64)
        x_f = F_tau @ self.state.x
        phi_f = float(wrap_to_2pi(x_f[0]))
        omega_f = float(x_f[1])
        # Same integrated Q form used in predict() — consistency.
        Q_tau = self._build_Qd(tau_s) if tau_s > 0 else np.zeros((2, 2))
        P_f = F_tau @ self.state.P @ F_tau.T + Q_tau
        sigma_phi = float(math.sqrt(max(P_f[0, 0], 0.0)))
        sigma_omega = float(math.sqrt(max(P_f[1, 1], 0.0)))
        return phi_f, sigma_phi, omega_f, sigma_omega

    # ─── Reset (for fallback re-init) ────────────────────────────────────

    def reset(self, initial_omega: Optional[float] = None) -> None:
        """Reinitialize state. Use when divergence detected by upstream cascade."""
        omega0 = self._initial_omega if initial_omega is None else float(initial_omega)
        self.state = EKFL1State(
            x=np.array([0.0, omega0], dtype=np.float64),
            P=np.diag(
                [self._initial_P_phi, self._initial_P_omega]
            ).astype(np.float64),
            t_last=None,
        )

    # ─── Diagnostics ─────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self.state.t_last is not None

    def condition_number_P(self) -> float:
        """Largest / smallest eigenvalue of P. >1e10 = numerically unhealthy."""
        eigvals = np.linalg.eigvalsh(self.state.P)
        eigvals = np.maximum(eigvals, 1e-30)
        return float(eigvals.max() / eigvals.min())
