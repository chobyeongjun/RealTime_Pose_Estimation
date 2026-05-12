"""Plan D top-level facade — single public class for the control loop.

This class is the **production-facing API**. The C++ port (control repo)
will mirror this surface 1:1.

Workflow per frame:
    1. predictor.feed(t, q, sigma_per_joint, hip_z)        — ingest pose
    2. forecast = predictor.forecast(tau_s=0.05)           — 50 ms-ahead
    3. predictor.is_ready_for_control()                    — gate check
    4. event = predictor.predict_heel_strike(side="L")     — when to trigger

References:
    docs/lessons/plan_d_predictor_spec.md §2.7 (heel-strike prediction)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from perception.plan_d_prototype.cascade import (
    CascadeForecast,
    CascadeLevel,
    CascadeStepResult,
    PredictorCascade,
)
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_2pi


# Heel-strike phase convention (plan_d_predictor_spec.md §2.7):
# Left HS at φ = 0; Right HS at φ = π.
PHI_HS_L: float = 0.0
PHI_HS_R: float = math.pi


@dataclass
class HeelStrikeEvent:
    """Predicted heel-strike for one side.

    Attributes:
        side: "L" or "R"
        t_ahead_s: predicted time until next HS (seconds, ≥ 0).
                   inf if cadence too slow to estimate, NaN if not ready.
        sigma_t_s: 1σ uncertainty estimate (seconds).
        confidence: 0..1 — combination of ambiguity, vision integrity,
                    cadence stability.
        ready: True if predictor.is_ready_for_control() AND t_ahead_s finite.
    """

    side: Literal["L", "R"]
    t_ahead_s: float
    sigma_t_s: float
    confidence: float
    ready: bool


class PlanDPredictor:
    """Top-level Plan D phase-locked predictor.

    Thin facade over PredictorCascade with HS prediction + control gate.
    """

    def __init__(
        self,
        n_joints: int = 6,
        fs_hz: float = 60.0,
        **cascade_kwargs,
    ) -> None:
        self.cascade = PredictorCascade(
            n_joints=n_joints,
            fs_hz=fs_hz,
            **cascade_kwargs,
        )
        self._last_step: Optional[CascadeStepResult] = None

    # ─── Ingest ──────────────────────────────────────────────────────────

    def feed(
        self,
        t_now: float,
        q: Optional[np.ndarray] = None,
        sigma_per_joint: Optional[np.ndarray] = None,
        hip_z_world_m: float = float("nan"),
    ) -> CascadeStepResult:
        """One frame of pose + hip-z. Returns step diagnostics."""
        result = self.cascade.step(
            t_now=t_now,
            q=q,
            sigma_per_joint=sigma_per_joint,
            hip_z_world_m=hip_z_world_m,
        )
        self._last_step = result
        return result

    # ─── Forecast ────────────────────────────────────────────────────────

    def forecast(self, tau_s: float) -> CascadeForecast:
        """Read-only state forecast at τ seconds ahead."""
        return self.cascade.predict_ahead(tau_s)

    # ─── Control-readiness gate ──────────────────────────────────────────

    def is_ready_for_control(
        self,
        require_l3: bool = True,
        max_sigma_phi: float = 0.5,
        max_ambiguity: float = 0.5,
    ) -> bool:
        """Whether the predictor's state is trustworthy for actuator control.

        Gates (all must pass):
            - last step exists and vision was not lost
            - cascade at the required level (L3 by default)
            - phase σ below threshold
            - latest CrossCorrPhaseEstimator ambiguity below threshold (if L3)

        Returns False until the cascade has settled.
        """
        if self._last_step is None:
            return False
        step = self._last_step
        if step.vision_lost:
            return False
        if require_l3 and step.level < CascadeLevel.L3:
            return False
        if step.sigma_phi > max_sigma_phi:
            return False
        # Ambiguity is only available when estimator ran this step
        if step.estimator_estimate is not None:
            if step.estimator_estimate.ambiguity_ratio > max_ambiguity:
                return False
        return True

    # ─── Heel-strike prediction ──────────────────────────────────────────

    def predict_heel_strike(
        self,
        side: Literal["L", "R"],
        max_t_ahead_s: float = 0.150,
        min_omega_rad_s: float = 1.0,
    ) -> HeelStrikeEvent:
        """Predict time until next heel-strike on the given side.

        Solves:  φ_HS = (φ_now + ω × t_HS) mod 2π
        →        t_HS = wrap((φ_HS - φ_now)) / ω    if ω > min

        Wraps to [0, 2π/ω) — picks the FIRST upcoming HS.

        Args:
            side: "L" (HS at φ=0) or "R" (HS at φ=π).
            max_t_ahead_s: clip predictions beyond this (return inf).
                Default 150 ms — beyond this is not actionable for control.
            min_omega_rad_s: below this ω, prediction is meaningless
                (freezing patient) — return inf.

        Returns:
            HeelStrikeEvent with t_ahead_s, sigma_t_s, confidence, ready.
        """
        if self._last_step is None:
            return HeelStrikeEvent(
                side=side, t_ahead_s=float("nan"),
                sigma_t_s=float("nan"), confidence=0.0, ready=False,
            )
        step = self._last_step
        phi_now = step.phi
        omega = step.omega
        if not (math.isfinite(phi_now) and math.isfinite(omega)):
            return HeelStrikeEvent(
                side=side, t_ahead_s=float("nan"),
                sigma_t_s=float("nan"), confidence=0.0, ready=False,
            )
        if omega < min_omega_rad_s:
            return HeelStrikeEvent(
                side=side, t_ahead_s=float("inf"),
                sigma_t_s=float("inf"), confidence=0.0, ready=False,
            )
        phi_hs = PHI_HS_L if side == "L" else PHI_HS_R
        # Δφ in [0, 2π) — wrap forward
        delta_phi = (phi_hs - phi_now) % TWO_PI
        t_ahead = delta_phi / omega
        if t_ahead > max_t_ahead_s:
            return HeelStrikeEvent(
                side=side, t_ahead_s=float("inf"),
                sigma_t_s=float("inf"), confidence=0.0, ready=False,
            )
        # Uncertainty: σ_t = σ_phi / ω  (1st-order approximation)
        sigma_t = step.sigma_phi / omega if omega > 0 else float("inf")
        # Confidence proxy: 1 - σ_t / max_t_ahead (clipped)
        confidence = 1.0 - min(1.0, sigma_t / max_t_ahead_s)
        confidence = max(0.0, confidence)
        ready = self.is_ready_for_control()
        return HeelStrikeEvent(
            side=side,
            t_ahead_s=float(t_ahead),
            sigma_t_s=float(sigma_t),
            confidence=float(confidence),
            ready=ready,
        )

    # ─── Diagnostics passthroughs ────────────────────────────────────────

    @property
    def level(self) -> CascadeLevel:
        return self.cascade.level

    @property
    def stride_count(self) -> int:
        return self.cascade.stride_count

    @property
    def phi(self) -> float:
        return self.cascade.phi

    @property
    def omega(self) -> float:
        return self.cascade.omega

    @property
    def template_touched_fraction(self) -> float:
        return self.cascade.template.touched_fraction
