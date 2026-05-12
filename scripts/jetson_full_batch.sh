#!/usr/bin/env bash
# 진정 single-batch Jetson test — sudo 의 root user 환경 손실 문제 fix.
#
# 사용자 의지 (2026-05-12):
#   "한번에 모두 진행하고 데이터 결과도 한번에 나올 수 있도록"
#
# 사용 (Jetson, ⚠ sudo 사용 X — user 모드):
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   bash scripts/jetson_full_batch.sh 2>&1 | tee /tmp/full_batch.log
#
# nvpmodel + jetson_clocks 는 의무 sudo — script 내부 sudo 명시.
# Python pipeline (tensorrt/torch/pyzed) 는 의무 user 모드 (root 의 PYTHONPATH 손실 방지).

set +e
set -o pipefail   # silent pass 방지

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}ERROR: 이 script 는 user 모드 의무 실행.${NC}"
    echo "  sudo 시 root 의 PYTHONPATH 에 user packages (tensorrt/torch/pyzed) 가 없음."
    echo "  Usage: bash scripts/jetson_full_batch.sh  (no sudo)"
    echo "  nvpmodel/jetson_clocks 는 script 내부 sudo 로 진행."
    exit 1
fi

echo "============================================================"
echo "  Jetson FULL BATCH test — 진정 single-run measurement"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "============================================================"
echo ""

# ─── Phase 1: Performance mode (sudo 의무) ──────────────────────────────
echo "── Phase 1: nvpmodel -m 0 + jetson_clocks (sudo) ──"
sudo nvpmodel -m 0 2>&1 | sed 's/^/  /' || true
sudo jetson_clocks 2>&1 | sed 's/^/  /' || true
echo ""

# ─── Phase 2: Plan D verify (215 tests, user mode) ──────────────────────
echo "── Phase 2: Plan D Phase 1.5 + Phase 2 verify (215 tests) ──"
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
    -q 2>&1 | tail -5
RC_PHASE2=$?
echo ""

# ─── Phase 3: Production 60s + trace logging (user mode!) ───────────────
echo "── Phase 3: Production 60s + RT trace logging (user mode) ──"
TS=$(date +%H%M%S)
TRACE_CSV=/tmp/production_trace_${TS}.csv
PROD_LOG=/tmp/production_full_${TS}.log

# Clean prior root-owned files (sudo run 의 *진정 *leftover*)
sudo rm -f /tmp/production_full.log /tmp/production_full_*.log 2>/dev/null || true

PYTHONPATH=src:src/perception/benchmarks timeout 60 \
    python3 src/perception/realtime/pipeline_main.py \
    --method B --no-display \
    --trace-csv "$TRACE_CSV" \
    2>&1 | tee "$PROD_LOG" | tail -30

echo ""
echo "  Trace CSV: $TRACE_CSV"
ls -la "$TRACE_CSV" 2>&1 | head -1
echo ""

# ─── Phase 4: Trace analysis ────────────────────────────────────────────
echo "── Phase 4: Trace analysis ──"
if [ -f "$TRACE_CSV" ]; then
    PYTHONPATH=src python3 scripts/analyze_trace.py "$TRACE_CSV" 2>&1
else
    echo -e "${RED}✗ Trace CSV not created${NC}"
fi
echo ""

# ─── Phase 5: cpp_ext detection (postprocess_accel build path) ──────────
echo "── Phase 5: cpp_ext build path detection (Python overhead 분해) ──"
echo "  현재 src/perception/benchmarks/cpp_ext/ 위치 확인..."
ls -la src/perception/benchmarks/cpp_ext/ 2>&1 | head -5 || echo "  → 위치 X"
echo "  Jetson 의 다른 location 의 cpp_ext find..."
find /home/chobb0 -type d -name "cpp_ext" 2>/dev/null | head -5
echo "  Jetson 의 *.so (pose_postprocess_cpp) find..."
find /home/chobb0 -name "pose_postprocess_cpp*" 2>/dev/null | head -5
echo ""

# ─── Phase 6: Summary ───────────────────────────────────────────────────
echo "============================================================"
echo "=== FULL BATCH SUMMARY ==="
echo "============================================================"
if [ "$RC_PHASE2" -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} Phase 2 verify: 215 tests PASS"
else
    echo -e "  ${RED}✗${NC} Phase 2 verify: failed (rc=$RC_PHASE2)"
fi

if [ -f "$TRACE_CSV" ]; then
    N_FRAMES=$(wc -l < "$TRACE_CSV")
    echo -e "  ${GREEN}✓${NC} Production trace: $N_FRAMES lines"
else
    echo -e "  ${RED}✗${NC} Production trace: not created"
fi

echo ""
echo "  Production log: /tmp/production_full.log"
echo "  Trace CSV:      $TRACE_CSV"
echo ""
echo "  진정 paste 의무:"
echo "    1. analyze_trace 의 *진정 *per-stage table*"
echo "    2. /tmp/production_full.log 의 *마지막 [PROFILE]*"
echo "    3. Phase 5 의 *cpp_ext find result*"
