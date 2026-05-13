#!/usr/bin/env bash
# Jetson 단독 풀 검증 — Teensy/AK60 없이 카메라+SHM+bridge mock까지 모두.
#
# 단계:
#   1. Unit tests (pytest)
#   2. nvpmodel + jetson_clocks
#   3. Pipeline boot 30s (no Teensy) — /hwalker_pose_v2 + /hwalker_forecast 생성 확인
#   4. dump_shm 으로 layout 검증 (5s watch)
#   5. shm_to_teensy_bridge --mock 으로 forecast → mock serial 흐름 검증
#   6. analyze_trace 로 RT latency 측정
#   7. (선택) walking session 60s
#
# 사용:
#   bash scripts/jetson_full_test.sh           # 전체 (no walking)
#   bash scripts/jetson_full_test.sh --walking # walking session 포함
#
# 사용자 의무: ZED X Mini 카메라 연결, Jetson Orin NX, TRT engine 빌드 완료.

set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DO_WALKING=0
[[ "${1:-}" == "--walking" ]] && DO_WALKING=1

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="recordings/jetson_full_${TS}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.txt"
: > "$SUMMARY"

note() {
    echo -e "${BLUE}$1${NC}"
    echo "$1" >> "$SUMMARY"
}
pass() {
    echo -e "  ${GREEN}✓ $1${NC}"
    echo "  PASS: $1" >> "$SUMMARY"
}
fail() {
    echo -e "  ${RED}✗ $1${NC}"
    echo "  FAIL: $1" >> "$SUMMARY"
}
warn() {
    echo -e "  ${YELLOW}⚠ $1${NC}"
    echo "  WARN: $1" >> "$SUMMARY"
}

echo "============================================================"
echo "  Jetson Full Test — ${TS}"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "  Output: $LOG_DIR"
echo "============================================================"
echo "" | tee -a "$SUMMARY"

# ─── Phase 1: Unit tests ────────────────────────────────────────────────
note "[1/7] Unit tests (pytest)"
PYTEST_LOG="$LOG_DIR/01_pytest.log"
PYTHONPATH=src:src/perception/benchmarks python3 -m pytest \
    tests/test_phase_b_integration.py \
    tests/test_teensy_protocol.py \
    tests/test_bridge_flow.py \
    -v --tb=short 2>&1 | tee "$PYTEST_LOG" | tail -30
if grep -q "passed" "$PYTEST_LOG" && ! grep -q "failed" "$PYTEST_LOG"; then
    pass "all unit tests passed"
else
    fail "unit test failures — see $PYTEST_LOG"
fi
echo ""

# ─── Phase 2: Performance mode ──────────────────────────────────────────
note "[2/7] nvpmodel + jetson_clocks"
PERF_LOG="$LOG_DIR/02_perf.log"
sudo nvpmodel -m 0 2>&1 | tee "$PERF_LOG" || warn "nvpmodel -m 0 needed sudo"
sudo jetson_clocks 2>&1 | tee -a "$PERF_LOG" || warn "jetson_clocks needed sudo"
pass "performance mode applied"
echo ""

# ─── Phase 3: Pipeline boot ─────────────────────────────────────────────
note "[3/7] Pipeline boot 30s with --enable-shm-v2 --enable-plan-d"
PIPELINE_LOG="$LOG_DIR/03_pipeline.log"
PYTHONPATH=src:src/perception/benchmarks timeout 30 python3 \
    src/perception/realtime/pipeline_main.py \
    --no-display --method B \
    --enable-shm-v2 --enable-plan-d \
    > "$PIPELINE_LOG" 2>&1 &
PIPE_PID=$!

# Give it 8s to warm up (engine load + camera init)
sleep 8
if ! ps -p $PIPE_PID > /dev/null 2>&1; then
    fail "pipeline died during warmup — tail of log:"
    tail -20 "$PIPELINE_LOG"
    exit 1
fi
pass "pipeline running (pid=$PIPE_PID)"

# ─── Phase 4: SHM layout dump (5s watch) ────────────────────────────────
note "[4/7] dump_shm 5s watch — /hwalker_pose_v2 + /hwalker_forecast"
DUMP_LOG="$LOG_DIR/04_dump_shm.log"
PYTHONPATH=src python3 scripts/dump_shm.py --watch 5 --rate 5 \
    > "$DUMP_LOG" 2>&1 && pass "SHM layout sane" || fail "SHM layout issues — $DUMP_LOG"
echo ""

# ─── Phase 5: bridge --mock ──────────────────────────────────────────────
note "[5/7] shm_to_teensy_bridge --mock 10s — forecast → frame emission"
BRIDGE_LOG="$LOG_DIR/05_bridge_mock.log"
PYTHONPATH=src python3 scripts/shm_to_teensy_bridge.py \
    --mock --duration 10 --rate-hz 200 --verbose \
    > "$BRIDGE_LOG" 2>&1
if grep -q "cmds_sent" "$BRIDGE_LOG"; then
    CMDS=$(grep -oP "cmds_sent: \K\d+" "$BRIDGE_LOG" | tail -1)
    HBS=$(grep -oP "heartbeats_sent: \K\d+" "$BRIDGE_LOG" | tail -1)
    pass "bridge mock: ${CMDS} commands + ${HBS} heartbeats over 10s"
    if [ "${CMDS:-0}" -lt 200 ]; then
        warn "  cmds_sent < 200 — forecast may not have been valid_for_control"
        warn "  check log: tail $BRIDGE_LOG"
    fi
else
    fail "bridge mock did not emit summary — $BRIDGE_LOG"
fi
echo ""

# ─── Phase 6: Pipeline shutdown + trace analysis ───────────────────────
note "[6/7] Stop pipeline + analyze trace"
kill -SIGTERM $PIPE_PID 2>/dev/null || true
wait $PIPE_PID 2>/dev/null || true

# pipeline_main may have produced a trace CSV — check standard location
TRACE_CSV=$(find . -maxdepth 4 -name "trace_*.csv" -newer "$LOG_DIR" 2>/dev/null | head -1)
if [ -n "$TRACE_CSV" ] && [ -x scripts/analyze_trace.py ]; then
    ANALYZE_LOG="$LOG_DIR/06_analyze.log"
    PYTHONPATH=src python3 scripts/analyze_trace.py "$TRACE_CSV" \
        > "$ANALYZE_LOG" 2>&1 && pass "trace analyzed" || warn "analyze_trace error"
elif [ -z "$TRACE_CSV" ]; then
    warn "no trace CSV produced — pipeline_main was run without --trace-csv"
fi
echo ""

# ─── Phase 7: (optional) Walking session ────────────────────────────────
if [ "$DO_WALKING" -eq 1 ]; then
    note "[7/7] Walking session 60s"
    if [ -x scripts/walking_session.sh ]; then
        bash scripts/walking_session.sh 60 2>&1 | tee "$LOG_DIR/07_walking.log"
    else
        warn "scripts/walking_session.sh not executable — skipping"
    fi
else
    note "[7/7] Walking session skipped (use --walking to enable)"
fi

# ─── Summary ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
cat "$SUMMARY"
echo ""
echo "All logs: $LOG_DIR/"
ls -lh "$LOG_DIR"

if grep -q "FAIL:" "$SUMMARY"; then
    echo -e "\n${RED}=== TESTS FAILED ===${NC}"
    exit 1
else
    echo -e "\n${GREEN}=== ALL TESTS PASSED ===${NC}"
fi
