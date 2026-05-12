"""Plan D Level 1 EKF — Constant velocity gait phase tracker.

State:
    x = [φ, ω]ᵀ
      φ: gait phase (rad, wrapped to [0, 2π))
      ω: cadence (rad/s; typical walking ~3-7 rad/s for 0.5-1.1 Hz stride)

Dynamics (continuous-velocity model, α implicitly 0):
    φ̇ = ω
    ω̇ = 0  (driven only by process noise)

Discrete form (Δt > 0):
    φ_{k+1} = φ_k + ω_k Δt           (mod 2π)
    ω_{k+1} = ω_k
    F = [[1, Δt], [0, 1]]
    Q = diag(σ_φ², σ_ω²) × Δt     (continuous-time discretization)

Measurement:
    z = φ_observed (rad), from CrossCorrPhaseEstimator
    h(x) = [1, 0] x = φ
    Innovation uses shortest-angle wrap to [-π, π].

Use cases:
    1. Cold-start (vision pose first frame, no template yet) — fallback to L1.
    2. Fallback when L3 phase-locked diverges (innovation gate triggers).
    3. Pathological gait where cycle template is invalid (start/stop, turning).

References:
    docs/lessons/plan_d_predictor_spec.md §3 (cold-start cascade)
    Thatte N. EKF gait phase, prosthesis control (cited)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from perception.plan_d_prototype.utils import (
    TWO_PI,
    joseph_update,
    validate_dt,
    wrap_to_2pi,
    wrap_to_pi,
)


# Sensible default cadence: 4.0 rad/s ≈ 0.64 Hz stride (slow walk)
# Range will adapt during tracking; the prior only affects cold-start.
_DEFAULT_INITIAL_OMEGA = 4.0

# Default initial covariance: large to express cold-start uncertainty
_DEFAULT_INITIAL_P_PHI = (math.pi) ** 2  # σ ~ π (1 rad), maximally uncertain
_DEFAULT_INITIAL_P_OMEGA = (2.0) ** 2    # σ ~ 2 rad/s


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
        predict(t_now)
        update(z_phi, R_override=None)
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
            process_noise_phi: σ²_φ (rad² / s) — continuous-time spectral density.
                Small (1e-6) since φ is integrated state.
            process_noise_omega: σ²_ω ((rad/s)² / s) — cadence drift rate.
                Reasonable for adaptive walking: ~0.2 rad/s std per second.
            measurement_noise: σ²_z (rad²) — observed phase noise variance.
                ~0.05 rad² = σ ≈ 13° per observation (template match quality).
            initial_omega: prior cadence (rad/s). 4.0 ≈ 0.64 Hz slow walk.
            initial_P_phi/omega: prior variances (large for cold-start).
            max_dt_s: dt validation upper bound.
        """
        self._Q_continuous = np.diag(
            [process_noise_phi, process_noise_omega]
        ).astype(np.float64)
        self._R = np.array([[measurement_noise]], dtype=np.float64)
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

    def predict(self, t_now: float) -> None:
        """Time-update step. Caller provides monotonic seconds.

        First call only stamps t_last (no state evolution).
        Invalid dt (negative, NaN, > max_dt_s) → silently skip but update t_last
        to current to avoid huge dt on next call.
        """
        if not math.isfinite(t_now):
            return
        if self.state.t_last is None:
            self.state.t_last = float(t_now)
            return
        dt = float(t_now) - self.state.t_last
        if not validate_dt(dt, max_dt_s=self._max_dt_s):
            self.state.t_last = float(t_now)
            return
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        # State propagation
        x_pred = F @ self.state.x
        x_pred[0] = float(wrap_to_2pi(x_pred[0]))
        # Covariance propagation with continuous-time Q
        Q_d = self._Q_continuous * dt
        P_pred = F @ self.state.P @ F.T + Q_d
        P_pred = 0.5 * (P_pred + P_pred.T)  # exact symmetry
        self.state.x = x_pred
        self.state.P = P_pred
        self.state.t_last = float(t_now)

    # ─── Update ──────────────────────────────────────────────────────────

    def update(
        self,
        z_phi: float,
        R_override: Optional[float] = None,
    ) -> None:
        """Measurement update. z_phi = observed phase (rad).

        NaN/inf observation → skip silently (e.g., template not yet trained,
        or cross-correlation failed).
        """
        if not math.isfinite(z_phi):
            return
        R = self._R if R_override is None else np.array(
            [[float(R_override)]], dtype=np.float64
        )
        # Innovation: shortest angular distance
        z_pred = float((self._H @ self.state.x)[0])
        innov = float(wrap_to_pi(z_phi - z_pred))
        # Innovation covariance and Kalman gain
        S = self._H @ self.state.P @ self._H.T + R     # (1, 1)
        S_scalar = float(S[0, 0])
        if S_scalar <= 0.0 or not math.isfinite(S_scalar):
            # Pathological — bail
            return
        K = (self.state.P @ self._H.T) / S_scalar      # (2, 1)
        # State update
        x_new = self.state.x + K.flatten() * innov
        x_new[0] = float(wrap_to_2pi(x_new[0]))
        # Joseph form covariance update
        P_new = joseph_update(self.state.P, K, self._H, R)
        self.state.x = x_new
        self.state.P = P_new

    # ─── Predict ahead (forecast at τ seconds) ───────────────────────────

    def predict_ahead(self, tau_s: float) -> Tuple[float, float, float, float]:
        """Forecast state at τ seconds ahead of the last predict() time.

        This does NOT mutate the filter state — pure read-only forecast for
        downstream consumer (Plan D control loop).

        Args:
            tau_s: lookahead in seconds (>= 0; τ < 0 returns current).

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
        # Covariance forecast (no process noise to add for pure forecast,
        # but predictor downstream may add σ_model²×τ² per spec §2.6)
        Q_tau = self._Q_continuous * tau_s
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
