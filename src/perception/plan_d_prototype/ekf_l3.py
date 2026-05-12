"""Plan D Level 3 EKF — Template-driven phase-locked tracker.

State (same as L2):
    x = [φ, ω, α]ᵀ

Measurement model (the L3 difference vs L1/L2):
    z = q ∈ ℝ^K              (K-joint observation, e.g. K=6)
    h(x) = μ(φ) ∈ ℝ^K        (cycle template lookup)
    H = ∂h/∂x = [∂μ/∂φ, 0, 0]    ∈ ℝ^(K×3)
        where ∂μ/∂φ is from CycleTemplate.lookup_jacobian(φ)
    R = diag(σ_per_joint²)  ∈ ℝ^(K×K)   (per-keypoint noise from SHM v2)

Innovation:
    ỹ = z - h(x_pred) = q - μ(φ_pred)
        (no phase wrap needed — measurement is in joint-angle space)

Codex review 2026-05-12 adjustments:
    - Chi² thresholds **BY DOF** (not hardcoded 9.0) — used by external
      divergence module; this class exposes innovation Mahalanobis χ² so
      caller can gate.
    - LDLT solve for K-DOF S (NOT np.linalg.inv) — numerically robust.
    - Template post-update is the CALLER's responsibility — L3.update()
      returns innovation diagnostics; caller checks gate then calls
      template.update(...) only after innovation accepted.
    - Per-joint validity mask supported (NaN entries in q → that joint
      excluded from update).

Use cases:
    - Promoted from L2 after ≥ 3 strides with template ready
      (touched_fraction ≥ 0.5 per cascade activation criteria).
    - Demoted on divergence (large innov χ², missed HS, cadence jump).

References:
    docs/lessons/plan_d_predictor_spec.md §2.4-2.5 (template + EKF)
    docs/lessons/plan_d_phase2_design.md Adjustment 3-5 (Codex)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.ekf_l1 import PredictStatus
from perception.plan_d_prototype.ekf_l2 import EKFL2, EKFL2State
from perception.plan_d_prototype.utils import (
    TWO_PI,
    joseph_update,
    wrap_to_2pi,
    wrap_to_pi,
)


# Realistic per-joint σ floor (matches phase_estimator.SIGMA_PER_JOINT_FLOOR)
SIGMA_PER_JOINT_FLOOR: float = 0.01   # rad


@dataclass
class L3UpdateResult:
    """Return value of EKFL3.update — diagnostics for cascade divergence check.

    Attributes:
        applied: True if the EKF state was updated (innovation passed sanity).
        innovation_chi2: Mahalanobis χ² of the K-DOF innovation, or NaN.
        n_valid_joints: how many entries of q were finite + included.
        residual_rms: ‖innov‖ / √n_valid (joint-angle RMS in rad), or NaN.
    """

    applied: bool
    innovation_chi2: float
    n_valid_joints: int
    residual_rms: float


class EKFL3:
    """Level 3 EKF — template-driven phase-locked.

    Public API:
        predict(t_now) -> PredictStatus
        update(q, sigma_per_joint=None) -> L3UpdateResult
        predict_ahead(tau_s) -> (phi, sigma_phi, omega, sigma_omega,
                                  alpha, sigma_alpha, q_pred)
        reset(initial_omega=None, initial_alpha=None)
        from_l2(l2: EKFL2, template: CycleTemplate) -> EKFL3 (classmethod)
        condition_number_P() -> float

    Template is REFERENCED (not copied) — caller owns lifetime + updates.
    """

    def __init__(
        self,
        template: CycleTemplate,
        process_noise_phi: float = 1e-6,
        process_noise_alpha: float = 2e-1,
        initial_omega: float = 4.0,
        initial_alpha: float = 0.0,
        initial_P_phi: float = 1.0,
        initial_P_omega: float = 4.0,
        initial_P_alpha: float = 9.0,
        max_dt_s: float = 0.5,
        sigma_floor: float = SIGMA_PER_JOINT_FLOOR,
    ) -> None:
        if template.n_joints < 1:
            raise ValueError("template must have ≥ 1 joint")
        if sigma_floor <= 0.0:
            raise ValueError("sigma_floor must be > 0")
        # Internally hold an L2 (predict/state machinery) + template ref.
        self._l2 = EKFL2(
            process_noise_phi=process_noise_phi,
            process_noise_alpha=process_noise_alpha,
            measurement_noise=1.0,  # unused — L3 uses K-DOF measurement
            initial_omega=initial_omega,
            initial_alpha=initial_alpha,
            initial_P_phi=initial_P_phi,
            initial_P_omega=initial_P_omega,
            initial_P_alpha=initial_P_alpha,
            max_dt_s=max_dt_s,
        )
        self._template = template
        self._sigma_floor = float(sigma_floor)
        self._n_joints = template.n_joints

    # ─── State accessor ──────────────────────────────────────────────────

    @property
    def state(self) -> EKFL2State:
        return self._l2.state

    @property
    def template(self) -> CycleTemplate:
        return self._template

    # ─── Cascade promotion ───────────────────────────────────────────────

    @classmethod
    def from_l2(
        cls,
        l2: EKFL2,
        template: CycleTemplate,
        sigma_floor: float = SIGMA_PER_JOINT_FLOOR,
    ) -> "EKFL3":
        """Promote an L2 filter to L3, sharing the cycle template."""
        l3 = cls(
            template=template,
            sigma_floor=sigma_floor,
        )
        # Copy L2 state in-place
        l3._l2.state.x = l2.state.x.copy()
        l3._l2.state.P = l2.state.P.copy()
        l3._l2.state.t_last = l2.state.t_last
        return l3

    # ─── Predict ─────────────────────────────────────────────────────────

    def predict(self, t_now: float) -> PredictStatus:
        return self._l2.predict(t_now)

    # ─── Update (K-DOF template-driven) ──────────────────────────────────

    def update(
        self,
        q: np.ndarray,
        sigma_per_joint: Optional[np.ndarray] = None,
    ) -> L3UpdateResult:
        """Template-driven EKF update.

        Per Codex Adjustment 5: this method returns diagnostics, but does NOT
        update the cycle template. The cascade is expected to call
        `template.update(phi_post, q, β)` ONLY IF the innovation passes the
        divergence gate (innovation_chi2 below threshold per DOF).

        Args:
            q: (K,) joint-angle observation.
            sigma_per_joint: (K,) per-joint σ. None = unit variance for all.
                Floored at sigma_floor for numerical safety.

        Returns:
            L3UpdateResult with innovation_chi2 (for caller's gate),
            n_valid_joints, residual_rms, and applied flag.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self._n_joints,):
            raise ValueError(
                f"q shape {q_arr.shape} != ({self._n_joints},)"
            )

        # Per-joint validity mask
        valid_mask = np.isfinite(q_arr)
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            return L3UpdateResult(
                applied=False, innovation_chi2=float("nan"),
                n_valid_joints=0, residual_rms=float("nan"),
            )

        # Sigma per joint (with floor)
        if sigma_per_joint is None:
            sig = np.ones(self._n_joints, dtype=np.float64)
        else:
            sig = np.asarray(sigma_per_joint, dtype=np.float64)
            if sig.shape != (self._n_joints,):
                raise ValueError("sigma_per_joint shape mismatch")
            sig = np.where(np.isfinite(sig), sig, self._sigma_floor)
            sig = np.maximum(sig, self._sigma_floor)

        # Template prediction at current phi
        phi_pred = float(self.state.phi)
        mu_at_phi = self._template.lookup(phi_pred)            # (K,)
        if not np.all(np.isfinite(mu_at_phi)):
            return L3UpdateResult(
                applied=False, innovation_chi2=float("nan"),
                n_valid_joints=n_valid, residual_rms=float("nan"),
            )
        # Jacobian ∂μ/∂φ (K,)
        dmu_dphi = self._template.lookup_jacobian(phi_pred)    # (K,)
        if not np.all(np.isfinite(dmu_dphi)):
            return L3UpdateResult(
                applied=False, innovation_chi2=float("nan"),
                n_valid_joints=n_valid, residual_rms=float("nan"),
            )

        # Build H = [∂μ/∂φ, 0, 0]  (K, 3)
        H = np.zeros((self._n_joints, 3), dtype=np.float64)
        H[:, 0] = dmu_dphi

        # Build R = diag(σ²) — but mask out invalid joints by setting σ huge
        # (effectively zero weight).
        sig_eff = np.where(valid_mask, sig, 1e6)
        R = np.diag(sig_eff * sig_eff)                          # (K, K)

        # Innovation
        innov = q_arr - mu_at_phi                                # (K,)
        # Zero out invalid joints in the innovation so they contribute nothing
        innov = np.where(valid_mask, innov, 0.0)

        # S = H P H^T + R
        P = self.state.P
        HP = H @ P                                               # (K, 3)
        S = HP @ H.T + R                                         # (K, K)

        # K_gain = P H^T S^{-1}  via LDLT solve (Codex Adjustment 4)
        # Equivalent: K_T = solve(S, H P);  K = K_T.T
        # K has shape (3, K)
        try:
            # Use Cholesky if S is PD; fall back to general solve
            L = np.linalg.cholesky(S)
            # Solve L Lᵀ X = HP   for X = (3, K)
            # NumPy doesn't have cho_solve directly without scipy; use solve.
            HPt = HP                                             # (K, 3)
            S_inv_HP = np.linalg.solve(S, HPt)                   # (K, 3)
            K_gain = S_inv_HP.T                                  # (3, K)
        except np.linalg.LinAlgError:
            return L3UpdateResult(
                applied=False, innovation_chi2=float("nan"),
                n_valid_joints=n_valid, residual_rms=float("nan"),
            )

        # Mahalanobis χ² of innovation: yᵀ S⁻¹ y (only over valid joints)
        # We use the full S and full innov (invalid zeroed); the χ² is
        # informative because invalid contributions are zero in innov AND
        # the corresponding R is huge so S handles it gracefully.
        try:
            S_inv_innov = np.linalg.solve(S, innov)
            chi2 = float(innov @ S_inv_innov)
        except np.linalg.LinAlgError:
            chi2 = float("nan")

        # State update
        x_new = self.state.x + K_gain @ innov                     # (3,)
        x_new[0] = float(wrap_to_2pi(x_new[0]))
        # Joseph update for P
        P_new = joseph_update(P, K_gain, H, R)

        self._l2.state.x = x_new
        self._l2.state.P = P_new

        # Residual RMS (in radians, joint-angle space)
        if n_valid > 0:
            residual_rms = float(
                math.sqrt(float(np.sum(innov[valid_mask] ** 2)) / n_valid)
            )
        else:
            residual_rms = float("nan")

        return L3UpdateResult(
            applied=True,
            innovation_chi2=chi2,
            n_valid_joints=n_valid,
            residual_rms=residual_rms,
        )

    # ─── Predict ahead ───────────────────────────────────────────────────

    def predict_ahead(
        self, tau_s: float
    ) -> Tuple[float, float, float, float, float, float, np.ndarray]:
        """Forecast at τ seconds ahead, including template prediction.

        Returns:
            (phi, sigma_phi, omega, sigma_omega, alpha, sigma_alpha, q_pred)
            q_pred = μ(φ_forecast) — predicted joint angles.
        """
        phi_f, s_phi, omega_f, s_omega, alpha_f, s_alpha = \
            self._l2.predict_ahead(tau_s)
        q_pred = self._template.lookup(phi_f)
        return phi_f, s_phi, omega_f, s_omega, alpha_f, s_alpha, q_pred

    # ─── Reset ───────────────────────────────────────────────────────────

    def reset(
        self,
        initial_omega: Optional[float] = None,
        initial_alpha: Optional[float] = None,
    ) -> None:
        self._l2.reset(initial_omega=initial_omega, initial_alpha=initial_alpha)

    # ─── Diagnostics ─────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._l2.is_initialized

    def condition_number_P(self) -> float:
        return self._l2.condition_number_P()
