"""Plan D PredictorCascade — L1 → L2 → L3 activation + fallback chain.

Wires together every Phase 1.5 + Phase 2 module:
    HipVerticalPhaseEstimator (cold-start phase source)
    EKFL1 (const velocity) ─→ EKFL2 (const accel) ─→ EKFL3 (template-driven)
    CycleTemplate (μ over φ)
    CrossCorrPhaseEstimator (template phase observation)
    divergence module (innovation gate, chi² by DOF)

Codex review 2026-05-12 design:
    Adjustment 2: stride detection INSIDE cascade state (NOT external).
    Adjustment 4: LDLT solve via divergence.mahalanobis_chi2.
    Adjustment 5: template post-update ONLY after innovation gate passes.
    Adjustment 6: Hilbert envelope cold-start until 3 strides + template ready.

Activation criteria (Codex Q1, plan_d_predictor_spec.md §3):
    L1 → L2: stride_count ≥ 1
    L2 → L3: stride_count ≥ 3 AND template.touched_fraction ≥ 0.5
             AND residual RMS < threshold

Demotion (divergence detection):
    L3 → L2: innovation χ² > threshold for K joints
    L2 → L1: cadence_jump > 20% OR template residual χ² > 3σ
    L1 → hold: vision_loss > 60ms → pretension safe mode (upstream watchdog)

Public API:
    PredictorCascade(n_joints, ...)
    step(t_now, q, sigma_per_joint, hip_z_world_m) → CascadeStepResult
    predict_ahead(tau_s) → CascadeForecast
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.divergence import (
    cadence_jump_detector,
    chi2_threshold,
    innovation_gate,
    template_residual_chi2,
    vision_loss_detector,
)
from perception.plan_d_prototype.ekf_l1 import EKFL1, PredictStatus
from perception.plan_d_prototype.ekf_l2 import EKFL2
from perception.plan_d_prototype.ekf_l3 import EKFL3, L3UpdateResult
from perception.plan_d_prototype.hilbert_phase import (
    HilbertPhaseResult,
    HipVerticalPhaseEstimator,
)
from perception.plan_d_prototype.phase_estimator import (
    CrossCorrPhaseEstimator,
    PhaseEstimate,
)
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_2pi, wrap_to_pi


class CascadeLevel(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3


@dataclass
class CascadeStepResult:
    """One step output (per frame)."""

    level: CascadeLevel
    phi: float                           # post-update phase (rad)
    omega: float                         # rad/s
    alpha: float                         # rad/s² (0 in L1)
    sigma_phi: float                     # rad
    stride_count: int
    template_touched_fraction: float
    template_updated: bool               # True if template received this frame
    hilbert_phase: HilbertPhaseResult    # cold-start observation
    estimator_estimate: Optional[PhaseEstimate]   # template-based, when active
    l3_result: Optional[L3UpdateResult]
    vision_lost: bool
    promoted_this_step: bool
    demoted_this_step: bool
    predict_status: PredictStatus


@dataclass
class CascadeForecast:
    """predict_ahead result (read-only)."""

    phi: float
    sigma_phi: float
    omega: float
    sigma_omega: float
    alpha: float
    sigma_alpha: float
    q_pred: Optional[np.ndarray]   # only when L3 active


class PredictorCascade:
    """Top-level Plan D state machine.

    Construction parameters cover the entire stack — caller provides knobs
    for tuning. Defaults are clinical-walking sensible.
    """

    def __init__(
        self,
        n_joints: int = 6,
        # Camera / Hilbert source
        fs_hz: float = 60.0,
        hilbert_window_s: float = 1.5,
        hilbert_min_amp_m: float = 0.005,
        # Template
        n_bins: int = 128,
        beta_default: float = 0.05,
        beta_cold: float = 0.10,
        # Activation criteria
        l2_promote_strides: int = 1,
        l3_promote_strides: int = 3,
        l3_template_min_fraction: float = 0.5,
        l3_residual_rms_max: float = 0.30,    # rad
        # Demotion / divergence gates
        chi2_confidence: float = 0.99,
        cadence_jump_threshold: float = 0.20,
        vision_loss_max_gap_s: float = 0.060,
        # EKF tuning
        initial_omega: float = 4.0,
        process_noise_omega_l1: float = 4e-2,
        process_noise_alpha_l2: float = 2e-1,
        process_noise_alpha_l3: float = 2e-1,
        sigma_floor_rad: float = 0.01,
        max_dt_s: float = 0.5,
    ) -> None:
        if n_joints < 1:
            raise ValueError("n_joints must be ≥ 1")
        self._n_joints = int(n_joints)
        self._fs_hz = float(fs_hz)
        self._chi2_conf = float(chi2_confidence)
        self._cadence_jump_thr = float(cadence_jump_threshold)
        self._vision_loss_max_s = float(vision_loss_max_gap_s)
        self._l2_promote_strides = int(l2_promote_strides)
        self._l3_promote_strides = int(l3_promote_strides)
        self._l3_template_min_frac = float(l3_template_min_fraction)
        self._l3_residual_rms_max = float(l3_residual_rms_max)

        # Building blocks (Phase 1.5 + Phase 2)
        self.hilbert = HipVerticalPhaseEstimator(
            window_seconds=hilbert_window_s,
            fs_hz=fs_hz,
            min_amplitude_m=hilbert_min_amp_m,
        )
        self.template = CycleTemplate(
            n_bins=n_bins,
            n_joints=n_joints,
            beta_default=beta_default,
            beta_cold=beta_cold,
        )
        self.estimator = CrossCorrPhaseEstimator(
            self.template,
            min_touched_fraction=l3_template_min_fraction,
            sigma_floor=sigma_floor_rad,
        )
        self.l1 = EKFL1(
            initial_omega=initial_omega,
            process_noise_omega=process_noise_omega_l1,
            max_dt_s=max_dt_s,
        )
        self.l2: Optional[EKFL2] = None
        self.l3: Optional[EKFL3] = None
        self._process_noise_alpha_l2 = float(process_noise_alpha_l2)
        self._process_noise_alpha_l3 = float(process_noise_alpha_l3)
        self._sigma_floor = float(sigma_floor_rad)
        self._max_dt_s = float(max_dt_s)

        # State machine
        self._level = CascadeLevel.L1
        self._stride_count = 0
        self._prev_phi: Optional[float] = None
        self._prev_stride_omega: Optional[float] = None  # ω at last HS, for jump check
        self._t_last_valid_pose: Optional[float] = None

    # ─── Public properties ───────────────────────────────────────────────

    @property
    def level(self) -> CascadeLevel:
        return self._level

    @property
    def stride_count(self) -> int:
        return self._stride_count

    @property
    def phi(self) -> float:
        return self._active_filter_phi()

    @property
    def omega(self) -> float:
        return self._active_filter_omega()

    @property
    def alpha(self) -> float:
        if self._level == CascadeLevel.L1:
            return 0.0
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.state.alpha
        assert self.l3 is not None
        return self.l3.state.alpha

    # ─── Step ────────────────────────────────────────────────────────────

    def step(
        self,
        t_now: float,
        q: Optional[np.ndarray] = None,
        sigma_per_joint: Optional[np.ndarray] = None,
        hip_z_world_m: float = float("nan"),
    ) -> CascadeStepResult:
        """One frame of fusion.

        Args:
            t_now: monotonic seconds.
            q: (K,) joint angles, or None if pose not available this frame.
            sigma_per_joint: (K,) per-joint σ (rad), or None for unit weight.
            hip_z_world_m: hip vertical position (m), or NaN if unavailable.

        Returns CascadeStepResult.
        """
        # 1. Hilbert envelope sampling
        self.hilbert.feed(t_now, hip_z_world_m)
        hilbert_phase = self.hilbert.estimate()

        # 2. Active filter predict
        predict_status = self._active_filter_predict(t_now)

        # 3. Vision loss detection
        if q is not None and np.any(np.isfinite(np.asarray(q))):
            self._t_last_valid_pose = t_now
            vision_lost = False
        else:
            if self._t_last_valid_pose is None:
                vision_lost = vision_loss_detector(
                    float("inf"), self._vision_loss_max_s
                )
            else:
                gap = t_now - self._t_last_valid_pose
                vision_lost = vision_loss_detector(gap, self._vision_loss_max_s)

        # 4. Phase observation
        promoted_this_step = False
        demoted_this_step = False
        estimator_estimate: Optional[PhaseEstimate] = None
        l3_result: Optional[L3UpdateResult] = None
        template_updated = False

        cold_start = (
            self._stride_count < self._l3_promote_strides
            or self.template.touched_fraction < self._l3_template_min_frac
        )

        if self._level == CascadeLevel.L3:
            assert self.l3 is not None
            if q is not None:
                l3_result = self.l3.update(
                    np.asarray(q, dtype=np.float64),
                    sigma_per_joint=(
                        np.asarray(sigma_per_joint, dtype=np.float64)
                        if sigma_per_joint is not None else None
                    ),
                )
                # Divergence check (Codex Adjustment 5)
                dof = l3_result.n_valid_joints
                if (
                    l3_result.applied and dof >= 1
                    and dof in chi2_threshold_supported_dofs()
                    and not innovation_gate(
                        l3_result.innovation_chi2, dof, self._chi2_conf,
                    )
                ):
                    # Accepted — also update template
                    self.template.update(
                        self.l3.state.phi,
                        np.asarray(q, dtype=np.float64),
                    )
                    template_updated = True
                elif l3_result.applied and dof >= 1:
                    # Diverged → demote to L2
                    self._demote_to_l2()
                    demoted_this_step = True
        else:
            # L1 / L2 path — phase observation source
            if cold_start:
                if hilbert_phase.valid:
                    self._active_filter_update(hilbert_phase.phi)
            else:
                # Template ready (stride_count ≥ 3 but still in L1/L2)
                if q is not None:
                    est = self.estimator.estimate(
                        np.asarray(q, dtype=np.float64),
                        sigma_per_joint=(
                            np.asarray(sigma_per_joint, dtype=np.float64)
                            if sigma_per_joint is not None else None
                        ),
                    )
                    estimator_estimate = est
                    if est.valid and est.ambiguity_ratio < 0.5:
                        self._active_filter_update(est.phi)
            # In any case, accumulate template if frame is valid
            if q is not None:
                phi_now = self._active_filter_phi()
                self.template.update(
                    phi_now, np.asarray(q, dtype=np.float64),
                )
                template_updated = True

        # 5. Stride detection (inside cascade — Codex Adjustment 2)
        phi_now = self._active_filter_phi()
        if (
            self._prev_phi is not None
            and self._prev_phi > 3.0 * math.pi / 2.0
            and phi_now < 0.5 * math.pi
        ):
            # Phase wrap — count stride
            # Cadence-jump confirmation (Codex § divergence):
            current_omega = self._active_filter_omega()
            if (
                self._prev_stride_omega is not None
                and cadence_jump_detector(
                    current_omega,
                    self._prev_stride_omega,
                    self._cadence_jump_thr,
                )
            ):
                # Spurious wrap — demote (cadence jump)
                if self._level == CascadeLevel.L3:
                    self._demote_to_l2()
                    demoted_this_step = True
                elif self._level == CascadeLevel.L2:
                    self._demote_to_l1()
                    demoted_this_step = True
            else:
                self._stride_count += 1
                self._prev_stride_omega = current_omega
        self._prev_phi = phi_now

        # 6. Promotion
        if (
            self._level == CascadeLevel.L1
            and self._stride_count >= self._l2_promote_strides
        ):
            self._promote_to_l2()
            promoted_this_step = True
        elif (
            self._level == CascadeLevel.L2
            and self._stride_count >= self._l3_promote_strides
            and self.template.touched_fraction >= self._l3_template_min_frac
        ):
            self._promote_to_l3()
            promoted_this_step = True

        return CascadeStepResult(
            level=self._level,
            phi=self._active_filter_phi(),
            omega=self._active_filter_omega(),
            alpha=(0.0 if self._level == CascadeLevel.L1 else self._active_filter_alpha()),
            sigma_phi=self._active_filter_sigma_phi(),
            stride_count=self._stride_count,
            template_touched_fraction=self.template.touched_fraction,
            template_updated=template_updated,
            hilbert_phase=hilbert_phase,
            estimator_estimate=estimator_estimate,
            l3_result=l3_result,
            vision_lost=vision_lost,
            promoted_this_step=promoted_this_step,
            demoted_this_step=demoted_this_step,
            predict_status=predict_status,
        )

    # ─── Forecast ────────────────────────────────────────────────────────

    def predict_ahead(self, tau_s: float) -> CascadeForecast:
        """Forecast at τ seconds ahead. Pure read-only."""
        if self._level == CascadeLevel.L1:
            phi, s_phi, omega, s_omega = self.l1.predict_ahead(tau_s)
            return CascadeForecast(
                phi=phi, sigma_phi=s_phi, omega=omega, sigma_omega=s_omega,
                alpha=0.0, sigma_alpha=0.0, q_pred=None,
            )
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            phi, s_phi, omega, s_omega, alpha, s_alpha = self.l2.predict_ahead(tau_s)
            return CascadeForecast(
                phi=phi, sigma_phi=s_phi, omega=omega, sigma_omega=s_omega,
                alpha=alpha, sigma_alpha=s_alpha, q_pred=None,
            )
        assert self.l3 is not None
        phi, s_phi, omega, s_omega, alpha, s_alpha, q_pred = self.l3.predict_ahead(tau_s)
        return CascadeForecast(
            phi=phi, sigma_phi=s_phi, omega=omega, sigma_omega=s_omega,
            alpha=alpha, sigma_alpha=s_alpha, q_pred=q_pred,
        )

    # ─── Internal: active filter dispatch ────────────────────────────────

    def _active_filter_predict(self, t_now: float) -> PredictStatus:
        if self._level == CascadeLevel.L1:
            return self.l1.predict(t_now)
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.predict(t_now)
        assert self.l3 is not None
        return self.l3.predict(t_now)

    def _active_filter_update(self, z_phi: float) -> bool:
        if self._level == CascadeLevel.L1:
            return self.l1.update(z_phi)
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.update(z_phi)
        # L3 uses K-DOF update only — single-φ update not appropriate
        return False

    def _active_filter_phi(self) -> float:
        if self._level == CascadeLevel.L1:
            return self.l1.state.phi
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.state.phi
        assert self.l3 is not None
        return self.l3.state.phi

    def _active_filter_omega(self) -> float:
        if self._level == CascadeLevel.L1:
            return self.l1.state.omega
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.state.omega
        assert self.l3 is not None
        return self.l3.state.omega

    def _active_filter_alpha(self) -> float:
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.state.alpha
        if self._level == CascadeLevel.L3:
            assert self.l3 is not None
            return self.l3.state.alpha
        return 0.0

    def _active_filter_sigma_phi(self) -> float:
        if self._level == CascadeLevel.L1:
            return self.l1.state.sigma_phi
        if self._level == CascadeLevel.L2:
            assert self.l2 is not None
            return self.l2.state.sigma_phi
        assert self.l3 is not None
        return self.l3.state.sigma_phi

    # ─── Promote / demote ────────────────────────────────────────────────

    def _promote_to_l2(self) -> None:
        self.l2 = EKFL2.from_l1(
            self.l1, process_noise_alpha=self._process_noise_alpha_l2,
        )
        self._level = CascadeLevel.L2

    def _promote_to_l3(self) -> None:
        assert self.l2 is not None
        self.l3 = EKFL3.from_l2(
            self.l2, self.template, sigma_floor=self._sigma_floor,
        )
        self._level = CascadeLevel.L3

    def _demote_to_l2(self) -> None:
        """L3 → L2 — keep [φ, ω, α], drop template observation path."""
        if self.l3 is None:
            return
        # Copy L3 state back into L2 (L3 wraps L2 — state already there)
        # Re-init an L2 from the current L3 state
        self.l2 = EKFL2(
            process_noise_alpha=self._process_noise_alpha_l2,
            max_dt_s=self._max_dt_s,
        )
        self.l2.state.x = self.l3.state.x.copy()
        self.l2.state.P = self.l3.state.P.copy()
        self.l2.state.t_last = self.l3.state.t_last
        self.l3 = None
        self._level = CascadeLevel.L2

    def _demote_to_l1(self) -> None:
        """L2 → L1 — keep φ, ω; drop α."""
        if self.l2 is None:
            return
        self.l1.state.x[0] = self.l2.state.phi
        self.l1.state.x[1] = self.l2.state.omega
        self.l1.state.P = self.l2.state.P[:2, :2].copy()
        self.l1.state.t_last = self.l2.state.t_last
        self.l2 = None
        self._level = CascadeLevel.L1


def chi2_threshold_supported_dofs() -> set:
    """Return DOFs supported by chi2 threshold table — used for safe gating."""
    from perception.plan_d_prototype.divergence import CHI2_THRESHOLD_99
    return set(CHI2_THRESHOLD_99.keys())
