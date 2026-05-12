#!/usr/bin/env bash
# Plan D Phase 2 (complete) — Jetson 환경 검증.
#
# Phase 1.5 (126 tests) + Phase 2 (89 tests) = 215 total.
#
# 사용 (Jetson):
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   bash scripts/jetson_phase2_verify.sh 2>&1 | tee /tmp/phase2_verify.log
#   echo "exit=$?"
#
# 기대 결과:
#   ✓ scipy installed
#   ✓ 22 Plan D imports
#   ✓ 215 unit tests PASS

set +e
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "  Plan D Phase 2 (complete) — Jetson verification"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo "============================================================"
echo ""

# ─── 1. scipy availability ──────────────────────────────────────────────
echo "── 1. scipy availability ──"
if python3 -c "import scipy; print('scipy', scipy.__version__)" 2>/dev/null; then
    echo -e "  ${GREEN}✓ scipy installed${NC}"
else
    echo -e "  ${RED}✗ scipy NOT installed${NC}"
    echo "  Install: pip3 install --user scipy"
    exit 1
fi

# ─── 2. baseline tools ──────────────────────────────────────────────────
echo ""
echo "── 2. baseline tools ──"
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | sed 's/^/  /'
python3 -c "import pytest; print('pytest', pytest.__version__)" 2>&1 | sed 's/^/  /'

# ─── 3. Plan D Phase 1.5 + Phase 2 imports (22 symbols) ─────────────────
echo ""
echo "── 3. Plan D imports (Phase 1.5 + Phase 2) ──"
PYTHONPATH=src python3 -c "
from perception.plan_d_prototype import (
    # utils
    TWO_PI, wrap_to_pi, wrap_to_2pi, validate_dt, joseph_update, bin_of_phase,
    # ekf_l1
    EKFL1, PredictStatus,
    # ekf_l2
    EKFL2,
    # ekf_l3
    EKFL3, L3UpdateResult,
    # template + estimator
    CycleTemplate, CrossCorrPhaseEstimator, PhaseEstimate,
    # hilbert
    HipVerticalPhaseEstimator, HilbertPhaseResult,
    # divergence
    chi2_threshold, mahalanobis_chi2, innovation_gate,
    template_residual_chi2, cadence_jump_detector, vision_loss_detector,
    # cascade
    PredictorCascade, CascadeLevel, CascadeStepResult, CascadeForecast,
    # predictor (top-level)
    PlanDPredictor, HeelStrikeEvent, PHI_HS_L, PHI_HS_R,
)
print('  ✓ all 30+ Plan D Phase 2 symbols imported')
" 2>&1

# ─── 4. 215 unit tests (Phase 1.5 + Phase 2) ────────────────────────────
echo ""
echo "── 4. 215 Plan D unit tests (Phase 1.5: 126 + Phase 2: 89) ──"

PYTHONPATH=src python3 -m pytest \
    -p no:anyio -p no:asyncio \
    tests/test_plan_d_utils.py \
    tests/test_plan_d_l1.py \
    tests/test_plan_d_cycle_template.py \
    tests/test_plan_d_phase_estimator.py \
    tests/test_plan_d_hilbert_phase.py \
    tests/test_plan_d_cold_start_integration.py \
    tests/test_plan_d_l2.py \
    tests/test_plan_d_l3.py \
    tests/test_plan_d_divergence.py \
    tests/test_plan_d_cascade.py \
    tests/test_plan_d_predictor.py \
    -q 2>&1 | tail -15

RC=$?
echo ""
if [ "$RC" -eq 0 ]; then
    echo -e "${GREEN}=== ALL Phase 2 VERIFIED on Jetson — Plan D ready for walking ===${NC}"
    echo ""
    echo "  Top-level API ready:"
    echo "    from perception.plan_d_prototype import PlanDPredictor"
    echo "    predictor = PlanDPredictor(n_joints=6, fs_hz=60)"
    echo "    for each frame:"
    echo "        predictor.feed(t, q, sigma_per_joint, hip_z_world_m)"
    echo "    forecast = predictor.forecast(tau_s=0.05)"
    echo "    if predictor.is_ready_for_control():"
    echo "        event_l = predictor.predict_heel_strike(\"L\")"
    echo "        event_r = predictor.predict_heel_strike(\"R\")"
    echo ""
    echo "  Tomorrow walking test:"
    echo "    1. ZED Recorder walking_60s.svo2 (5 min)"
    echo "    2. PYTHONPATH=src python3 src/perception/realtime/pipeline_main.py \\"
    echo "         --svo2 walking_60s.svo2 --method B --no-display \\"
    echo "         --record-pose-npz walking_60s.npz"
    echo "    3. PYTHONPATH=src python3 scripts/run_plan_d_offline.py \\"
    echo "         walking_60s.npz --plot"
    exit 0
else
    echo -e "${RED}=== Phase 2 verification FAILED ===${NC}"
    exit 1
fi
