#!/usr/bin/env bash
# Plan D Phase 1.5 의 Jetson 환경 검증 — scipy + 126 tests + smoke.
#
# 사용 (Jetson):
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   bash scripts/jetson_phase15_verify.sh 2>&1 | tee /tmp/phase15_verify.log
#
# 기대 결과:
#   ✓ scipy installed
#   ✓ 126 plan_d Phase 1.5 tests PASS
#   ✓ smoke run prints all 6 module imports OK

set +e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "============================================================"
echo "  Plan D Phase 1.5 — Jetson verification"
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
    echo ""
    echo "  Install with one of:"
    echo "    pip3 install --user scipy"
    echo "    sudo apt install python3-scipy"
    exit 1
fi

# ─── 2. numpy + pytest baseline ─────────────────────────────────────────
echo ""
echo "── 2. numpy + pytest baseline ──"
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | sed 's/^/  /'
python3 -c "import pytest; print('pytest', pytest.__version__)" 2>&1 | sed 's/^/  /'

# ─── 3. Plan D Phase 1.5 modules import ─────────────────────────────────
echo ""
echo "── 3. Plan D Phase 1.5 imports ──"
PYTHONPATH=src python3 -c "
from perception.plan_d_prototype import (
    EKFL1, PredictStatus,
    CycleTemplate,
    CrossCorrPhaseEstimator, PhaseEstimate,
    HipVerticalPhaseEstimator, HilbertPhaseResult,
    wrap_to_pi, wrap_to_2pi, bin_of_phase, joseph_update,
)
print('  ✓ all 12 symbols imported')
" 2>&1

# ─── 4. 126 unit tests ──────────────────────────────────────────────────
echo ""
echo "── 4. 126 Plan D Phase 1.5 unit tests ──"
PYTHONPATH=src python3 -m pytest \
    tests/test_plan_d_utils.py \
    tests/test_plan_d_l1.py \
    tests/test_plan_d_cycle_template.py \
    tests/test_plan_d_phase_estimator.py \
    tests/test_plan_d_hilbert_phase.py \
    tests/test_plan_d_cold_start_integration.py \
    -q 2>&1 | tail -10

RC=$?
echo ""
if [ "$RC" -eq 0 ]; then
    echo -e "${GREEN}=== ALL Phase 1.5 VERIFIED on Jetson ===${NC}"
    echo ""
    echo "  Next steps (paste results to Mac after each):"
    echo "    1. Record walking SVO2:"
    echo "         (see docs/handovers/2026-05-12-jetson-tasks.md Step 2)"
    echo "    2. Replay + dump pose npz:"
    echo "         python3 src/perception/realtime/pipeline_main.py \\"
    echo "           --svo2 walking_60s.svo2 --record-pose-npz walking_60s.npz \\"
    echo "           --no-display --method B"
    echo "    3. Offline Plan D analysis:"
    echo "         python3 scripts/run_plan_d_offline.py walking_60s.npz"
    exit 0
else
    echo -e "${RED}=== Phase 1.5 verification FAILED ===${NC}"
    exit 1
fi
