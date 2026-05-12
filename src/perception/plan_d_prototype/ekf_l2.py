"""Plan D Level 2 EKF — Constant acceleration gait phase tracker.

State:
    x = [φ, ω, α]ᵀ
      φ: gait phase (rad, wrapped to [0, 2π))
      ω: cadence (rad/s)
      α: cadence acceleration (rad/s²) — random-walk driven by white noise

Dynamics (continuous-time, α driven by white noise q_α):
    φ̇ = ω
    ω̇ = α
    α̇ = w(t),   w(t) ~ N(0, q_α)

Discrete form (Δt > 0):
    φ_{k+1} = φ_k + ω_k Δt + ½ α_k Δt²    (mod 2π)
    ω_{k+1} = ω_k + α_k Δt
    α_{k+1} = α_k

    F = [[1, Δt, Δt²/2],
         [0, 1,  Δt   ],
         [0, 0,  1    ]]

Process noise Q_d (analytical integral — Codex PASSED, NOT van Loan for RT):
    For G_c = [0, 0, 1]ᵀ, Q_c = q_α scalar:
        Q_d = q_α × [[Δt⁵/20, Δt⁴/8,  Δt³/6],
                     [Δt⁴/8,  Δt³/3,  Δt²/2],
                     [Δt³/6,  Δt²/2,  Δt   ]]
    Plus optional independent φ noise q_φ × Δt × diag(1,0,0).

Measurement (single-phase observation, same as L1):
    z = φ_observed (rad)
    h(x) = [1, 0, 0] x = φ
    Innovation uses shortest-angle wrap to [-π, π].

Use cases:
    - Promoted from L1 after ≥ 1 stride (Codex Q1).
    - Demoted from L3 when template diverges but cadence is changing fast
      (start/stop, pathological accel patterns).

Codex PASSED (`plan_d_phase2_design.md:26-32`): analytical Q form is the
embedded RT choice. van Loan is non-deterministic and slower in hot path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from perception.plan_d_prototype.ekf_l1 import EKFL1, EKFL1State, PredictStatus
from perception.plan_d_prototype.utils import (
    TWO_PI,
    joseph_update,
    wrap_to_2pi,
    wrap_to_pi,
)


# Cold-start defaults (used when L2 is not promoted from L1).
_DEFAULT_INITIAL_OMEGA = 4.0           # rad/s ≈ 0.64 Hz
_DEFAULT_INITIAL_ALPHA = 0.0           # rad/s²
_DEFAULT_INITIAL_P_PHI = 1.0           # σ ≈ 1 rad (matches L1)
_DEFAULT_INITIAL_P_OMEGA = 4.0         # σ ≈ 2 rad/s
_DEFAULT_INITIAL_P_ALPHA = 9.0         # σ ≈ 3 rad/s² (cold-start max plausible)


@dataclass
class EKFL2State:
    """L2 state snapshot — exposes φ, ω, α with named accessors."""

    x: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, _DEFAULT_INITIAL_OMEGA, _DEFAULT_INITIAL_ALPHA], dtype=np.float64
        )
    )
    P: np.ndarray = field(
        default_factory=lambda: np.diag(
            [_DEFAULT_INITIAL_P_PHI, _DEFAULT_INITIAL_P_OMEGA, _DEFAULT_INITIAL_P_ALPHA]
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
    def alpha(self) -> float:
        return float(self.x[2])

    @property
    def sigma_phi(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], 0.0)))

    @property
    def sigma_omega(self) -> float:
        return float(math.sqrt(max(self.P[1, 1], 0.0)))

    @property
    def sigma_alpha(self) -> float:
        return float(math.sqrt(max(self.P[2, 2], 0.0)))


class EKFL2:
    """Level 2 EKF — Constant acceleration (φ, ω, α).

    Public API mirrors L1 for cascade interchangeability:
        predict(t_now) -> PredictStatus
        update(z_phi, R_override=None) -> bool
        predict_ahead(tau_s) -> (phi, sigma_phi, omega, sigma_omega, alpha, sigma_alpha)
        reset(initial_omega=None, initial_alpha=None)
        condition_number_P() -> float
        from_l1(l1: EKFL1) -> EKFL2 (classmethod, cascade promotion)

    Real-time safety:
        - All ops are O(1) (3-state).
        - No allocation in steady state.
        - float64 throughout.
    """

    def __init__(
        self,
        process_noise_phi: float = 1e-6,
        process_noise_alpha: float = 2e-1,
        measurement_noise: float = 0.05,
        initial_omega: float = _DEFAULT_INITIAL_OMEGA,
        initial_alpha: float = _DEFAULT_INITIAL_ALPHA,
        initial_P_phi: float = _DEFAULT_INITIAL_P_PHI,
        initial_P_omega: float = _DEFAULT_INITIAL_P_OMEGA,
        initial_P_alpha: float = _DEFAULT_INITIAL_P_ALPHA,
        max_dt_s: float = 0.5,
    ) -> None:
        """Initialize L2 EKF.

        Args:
            process_noise_phi: q_φ (rad²/s) — independent φ noise (quantization).
            process_noise_alpha: q_α ((rad/s²)²/s) — cadence-acceleration white-
                noise spectral density. Default 0.2 corresponds to σ_α growth
                ~0.45 rad/s² per √s (clinical adaptive walking).
            measurement_noise: σ²_z (rad²) — phase observation variance.
            initial_omega/alpha: cold-start state means.
            initial_P_*: cold-start state variances.
            max_dt_s: dt validation upper bound (caller drives watchdog on
                DT_TOO_LARGE).
        """
        self._q_phi = float(process_noise_phi)
        self._q_alpha = float(process_noise_alpha)
        self._R = np.array([[float(measurement_noise)]], dtype=np.float64)
        self._H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        self._initial_omega = float(initial_omega)
        self._initial_alpha = float(initial_alpha)
        self._initial_P_phi = float(initial_P_phi)
        self._initial_P_omega = float(initial_P_omega)
        self._initial_P_alpha = float(initial_P_alpha)
        self._max_dt_s = float(max_dt_s)
        self.state = EKFL2State(
            x=np.array([0.0, initial_omega, initial_alpha], dtype=np.float64),
            P=np.diag(
                [initial_P_phi, initial_P_omega, initial_P_alpha]
            ).astype(np.float64),
            t_last=None,
        )

    # ─── Cascade promotion from L1 ──────────────────────────────────────

    @classmethod
    def from_l1(
        cls,
        l1: EKFL1,
        initial_alpha: float = 0.0,
        initial_P_alpha: float = _DEFAULT_INITIAL_P_ALPHA,
        process_noise_alpha: Optional[float] = None,
    ) -> "EKFL2":
        """Promote an L1 filter to L2, copying φ, ω, P_φφ, P_φω, P_ωω.

        The new α state starts at `initial_alpha` (default 0) with large
        variance `initial_P_alpha`. Cross-terms P_φα, P_ωα start at 0
        (no prior coupling).
        """
        l2 = cls(
            process_noise_alpha=(
                process_noise_alpha if process_noise_alpha is not None else 2e-1
            ),
            initial_P_alpha=initial_P_alpha,
        )
        # Copy φ, ω
        l2.state.x[0] = l1.state.x[0]
        l2.state.x[1] = l1.state.x[1]
        l2.state.x[2] = float(initial_alpha)
        # Copy 2×2 block; expand to 3×3 with α slot
        P3 = np.zeros((3, 3), dtype=np.float64)
        P3[:2, :2] = l1.state.P
        P3[2, 2] = float(initial_P_alpha)
        l2.state.P = P3
        l2.state.t_last = l1.state.t_last
        return l2

    # ─── Predict ─────────────────────────────────────────────────────────

    def predict(self, t_now: float) -> PredictStatus:
        """Time-update step. Returns status (same enum as L1)."""
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
        # Normal integration
        F = self._build_F(dt)
        x_pred = F @ self.state.x
        x_pred[0] = float(wrap_to_2pi(x_pred[0]))
        Q_d = self._build_Qd(dt)
        P_pred = F @ self.state.P @ F.T + Q_d
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.state.x = x_pred
        self.state.P = P_pred
        self.state.t_last = float(t_now)
        return PredictStatus.OK

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """State transition matrix for [φ, ω, α] under constant-α model."""
        dt2_half = 0.5 * dt * dt
        return np.array(
            [[1.0, dt, dt2_half],
             [0.0, 1.0, dt],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _build_Qd(self, dt: float) -> np.ndarray:
        """Integrated discrete process-noise covariance.

        For α driven by white noise of spectral density q_α:
            Q_d = q_α × [[dt⁵/20, dt⁴/8,  dt³/6],
                         [dt⁴/8,  dt³/3,  dt²/2],
                         [dt³/6,  dt²/2,  dt   ]]
        Plus independent φ noise: q_φ × dt × diag(1, 0, 0).
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        qa = self._q_alpha
        Q_alpha = np.array(
            [[qa * dt5 / 20.0, qa * dt4 / 8.0, qa * dt3 / 6.0],
             [qa * dt4 / 8.0,  qa * dt3 / 3.0, qa * dt2 / 2.0],
             [qa * dt3 / 6.0,  qa * dt2 / 2.0, qa * dt]],
            dtype=np.float64,
        )
        Q_alpha[0, 0] += self._q_phi * dt
        return Q_alpha

    # ─── Update ──────────────────────────────────────────────────────────

    def update(
        self,
        z_phi: float,
        R_override: Optional[float] = None,
    ) -> bool:
        """Phase-observation update. Returns True if applied."""
        if not math.isfinite(z_phi):
            return False
        if R_override is None:
            R = self._R
        else:
            R_val = float(R_override)
            if not math.isfinite(R_val) or R_val <= 0.0:
                return False
            R = np.array([[R_val]], dtype=np.float64)
        z_pred = float((self._H @ self.state.x)[0])
        innov = float(wrap_to_pi(z_phi - z_pred))
        S = self._H @ self.state.P @ self._H.T + R
        S_scalar = float(S[0, 0])
        if S_scalar <= 0.0 or not math.isfinite(S_scalar):
            return False
        # LDLT for 1-DOF is trivial — division. For consistency with C++ Eigen
        # path, we use scalar inverse here (Codex IMPROVE for K-DOF in L3).
        K = (self.state.P @ self._H.T) / S_scalar     # (3, 1)
        x_new = self.state.x + K.flatten() * innov
        x_new[0] = float(wrap_to_2pi(x_new[0]))
        P_new = joseph_update(self.state.P, K, self._H, R)
        self.state.x = x_new
        self.state.P = P_new
        return True

    # ─── Predict ahead ───────────────────────────────────────────────────

    def predict_ahead(
        self, tau_s: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Forecast at τ seconds ahead. Pure read-only.

        Returns:
            (phi, sigma_phi, omega, sigma_omega, alpha, sigma_alpha)
        """
        if not math.isfinite(tau_s) or tau_s < 0:
            tau_s = 0.0
        F_tau = self._build_F(tau_s)
        x_f = F_tau @ self.state.x
        phi_f = float(wrap_to_2pi(x_f[0]))
        omega_f = float(x_f[1])
        alpha_f = float(x_f[2])
        Q_tau = self._build_Qd(tau_s) if tau_s > 0 else np.zeros((3, 3))
        P_f = F_tau @ self.state.P @ F_tau.T + Q_tau
        sigma_phi = float(math.sqrt(max(P_f[0, 0], 0.0)))
        sigma_omega = float(math.sqrt(max(P_f[1, 1], 0.0)))
        sigma_alpha = float(math.sqrt(max(P_f[2, 2], 0.0)))
        return phi_f, sigma_phi, omega_f, sigma_omega, alpha_f, sigma_alpha

    # ─── Reset ───────────────────────────────────────────────────────────

    def reset(
        self,
        initial_omega: Optional[float] = None,
        initial_alpha: Optional[float] = None,
    ) -> None:
        omega0 = self._initial_omega if initial_omega is None else float(initial_omega)
        alpha0 = self._initial_alpha if initial_alpha is None else float(initial_alpha)
        self.state = EKFL2State(
            x=np.array([0.0, omega0, alpha0], dtype=np.float64),
            P=np.diag(
                [self._initial_P_phi, self._initial_P_omega, self._initial_P_alpha]
            ).astype(np.float64),
            t_last=None,
        )

    # ─── Diagnostics ─────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self.state.t_last is not None

    def condition_number_P(self) -> float:
        eigvals = np.linalg.eigvalsh(self.state.P)
        eigvals = np.maximum(eigvals, 1e-30)
        return float(eigvals.max() / eigvals.min())
