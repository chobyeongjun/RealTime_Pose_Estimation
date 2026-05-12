#!/usr/bin/env bash
# 진정 single-batch Jetson test — *진정 *모든 측정 한번에*.
#
# 사용자 의지 (2026-05-12):
#   "한번에 모두 진행하고 데이터 결과도 한번에 나올 수 있도록"
#   "코드 모두 구현해두고 그 다음에는 빠르게 진행"
#
# 사용 (Jetson):
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   sudo bash scripts/jetson_full_batch.sh 2>&1 | tee /tmp/full_batch.log
#
# sudo 의무 (nvpmodel + jetson_clocks).
#
# 진행:
#   1. Phase 1.5/2 verify (215 Plan D tests)
#   2. Production 60s with --trace-csv
#   3. analyze_trace.py — per-stage p50/p95/p99/max
#   4. Summary table

set +e
set -o pipefail   # silent pass 방지

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}NOTE: sudo 권유 — nvpmodel/jetson_clocks 의무${NC}"
fi

echo "============================================================"
echo "  Jetson FULL BATCH test — 진정 single-run measurement"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "============================================================"
echo ""

# ─── Phase 1: Performance mode ──────────────────────────────────────────
echo "── Phase 1: nvpmodel -m 0 + jetson_clocks ──"
nvpmodel -m 0 2>&1 | sed 's/^/  /' || true
jetson_clocks 2>&1 | sed 's/^/  /' || true
echo ""

# ─── Phase 2: Plan D verify (215 tests) ─────────────────────────────────
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

# ─── Phase 3: Production 60s + trace logging ────────────────────────────
echo "── Phase 3: Production 60s + RT trace logging ──"
TRACE_CSV=/tmp/production_trace_$(date +%H%M%S).csv

PYTHONPATH=src:src/perception/benchmarks timeout 60 \
    python3 src/perception/realtime/pipeline_main.py \
    --method B --no-display \
    --trace-csv "$TRACE_CSV" \
    2>&1 | tee /tmp/production_full.log | tail -30

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

# ─── Phase 5: Summary ───────────────────────────────────────────────────
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
    echo -e "  ${GREEN}✓${NC} Production trace: $N_FRAMES lines (header + frames)"
else
    echo -e "  ${RED}✗${NC} Production trace: not created"
fi

echo ""
echo "  Production log: /tmp/production_full.log"
echo "  Trace CSV:      $TRACE_CSV"
echo ""
echo "  진정 paste 의무:"
echo "    1. analyze_trace 의 *최종 *per-stage table* + warmup analysis"
echo "    2. /tmp/production_full.log 의 *마지막 [PROFILE]*"
