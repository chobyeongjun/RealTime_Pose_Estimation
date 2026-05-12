"""Plan D EKF Predictor — Python prototype (algorithm validation).

⚠️ PROTOTYPE ONLY ⚠️

This package is the **Python algorithm validation** for the Plan D phase-locked EKF
predictor. Production location = 사용자 control repo (C++ with Eigen).

Why Python prototype:
- Unit-testable algorithm (pytest)
- Fast iteration on numerical edge cases
- Reference implementation for C++ port verification
- Independent algorithm correctness from the C++ scaffolding work

Phase 1 modules (this session):
- utils: phase wrap, dt validation, Joseph covariance update
- ekf_l1: Level 1 (constant velocity, 2-state Kalman)
- cycle_template: μ(φ) 128 bins × K joints, recursive β update, cubic Hermite
- phase_estimator: cross-correlation phase observation (not FFT/Hilbert)

Phase 2 modules (next session):
- ekf_l2: Level 2 (constant acceleration, 3-state)
- ekf_l3: Level 3 (phase-locked with template measurement)
- cascade: L1→L2→L3 + divergence detection + fallback chain
- predictor: top-level facade + predict_ahead(τ)

References:
- docs/lessons/plan_d_predictor_spec.md (332 lines, full spec)
- docs/handovers/2026-05-12-post-v4l2-abandon-roadmap.md
- Thatte N. EKF gait phase (prosthesis prior art)
- Righetti L., Ijspeert A. 2008 (adaptive frequency oscillators)
"""

from perception.plan_d_prototype.utils import (
    TWO_PI,
    wrap_to_pi,
    wrap_to_2pi,
    validate_dt,
    joseph_update,
    bin_of_phase,
)
from perception.plan_d_prototype.ekf_l1 import EKFL1, EKFL1State
from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.phase_estimator import (
    CrossCorrPhaseEstimator,
    PhaseEstimate,
)

__all__ = [
    "TWO_PI",
    "wrap_to_pi",
    "wrap_to_2pi",
    "validate_dt",
    "joseph_update",
    "bin_of_phase",
    "EKFL1",
    "EKFL1State",
    "CycleTemplate",
    "CrossCorrPhaseEstimator",
    "PhaseEstimate",
]
