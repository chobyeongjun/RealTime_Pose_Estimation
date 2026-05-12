"""Plan D EKF Predictor — Python prototype (algorithm validation).

⚠️ PROTOTYPE ONLY ⚠️

This package is the **Python algorithm validation** for the Plan D phase-locked
EKF predictor. Production location = 사용자 control repo (C++ with Eigen).

Phase 1.5 modules (commits fb7e1b5 → 37ee3f4):
    utils:              phase wrap, dt validation, Joseph covariance, bin_of_phase
    ekf_l1:             Level 1 (const velocity, 2-state Kalman)
    cycle_template:     μ(φ) 128 bins × K joints, per-joint touched, β scheduling
    phase_estimator:    cross-correlation phase observation (ambiguity_ratio)
    hilbert_phase:      Hilbert envelope cold-start phase source (Hard Wall fix)

Phase 2 modules (this session):
    ekf_l2:             Level 2 (const accel, 3-state, analytical Q)
    ekf_l3:             Level 3 (template-driven, K-DOF, LDLT solve)
    divergence:         innovation gate, chi2 by DOF, residual chi2,
                        cadence_jump, vision_loss detectors
    cascade:            PredictorCascade — L1→L2→L3 + fallback + stride-inside
    predictor:          PlanDPredictor — top-level facade + HS prediction

References:
    docs/lessons/plan_d_predictor_spec.md          (full spec, 332 lines)
    docs/lessons/plan_d_phase2_design.md           (Codex adjustments)
    docs/lessons/codex_review_phase1_2026_05_12.md (Codex review verdict)
    docs/lessons/plan_d_phase1_review.md           (Phase 1.5 checklist)
"""

from perception.plan_d_prototype.utils import (
    TWO_PI,
    wrap_to_pi,
    wrap_to_2pi,
    validate_dt,
    joseph_update,
    bin_of_phase,
)
from perception.plan_d_prototype.ekf_l1 import EKFL1, EKFL1State, PredictStatus
from perception.plan_d_prototype.ekf_l2 import EKFL2, EKFL2State
from perception.plan_d_prototype.ekf_l3 import EKFL3, L3UpdateResult
from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.phase_estimator import (
    CrossCorrPhaseEstimator,
    PhaseEstimate,
)
from perception.plan_d_prototype.hilbert_phase import (
    HilbertPhaseResult,
    HipVerticalPhaseEstimator,
)
from perception.plan_d_prototype.divergence import (
    cadence_jump_detector,
    chi2_threshold,
    innovation_gate,
    mahalanobis_chi2,
    template_residual_chi2,
    vision_loss_detector,
    CHI2_THRESHOLD_99,
    CHI2_THRESHOLD_999,
)
from perception.plan_d_prototype.cascade import (
    CascadeForecast,
    CascadeLevel,
    CascadeStepResult,
    PredictorCascade,
)
from perception.plan_d_prototype.predictor import (
    HeelStrikeEvent,
    PHI_HS_L,
    PHI_HS_R,
    PlanDPredictor,
)

__all__ = [
    # utils
    "TWO_PI", "wrap_to_pi", "wrap_to_2pi", "validate_dt",
    "joseph_update", "bin_of_phase",
    # ekf_l1
    "EKFL1", "EKFL1State", "PredictStatus",
    # ekf_l2
    "EKFL2", "EKFL2State",
    # ekf_l3
    "EKFL3", "L3UpdateResult",
    # cycle_template + phase_estimator
    "CycleTemplate", "CrossCorrPhaseEstimator", "PhaseEstimate",
    # hilbert
    "HilbertPhaseResult", "HipVerticalPhaseEstimator",
    # divergence
    "cadence_jump_detector", "chi2_threshold", "innovation_gate",
    "mahalanobis_chi2", "template_residual_chi2", "vision_loss_detector",
    "CHI2_THRESHOLD_99", "CHI2_THRESHOLD_999",
    # cascade
    "CascadeForecast", "CascadeLevel", "CascadeStepResult", "PredictorCascade",
    # predictor
    "HeelStrikeEvent", "PHI_HS_L", "PHI_HS_R", "PlanDPredictor",
]
