#!/usr/bin/env bash
# 기존 walking SVO를 재처리 (다시 걸을 필요 X).
#
# 사용:
#   bash scripts/replay_walking_svo.sh recordings/walking_*/walking_*.svo2
#   bash scripts/replay_walking_svo.sh recordings/walking_*/walking_*.svo2 --diagnose
#
# 옵션:
#   --diagnose   pipeline_main 옵션을 하나씩 켜면서 어느 옵션이 1Hz fail 원인인지 격리
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SVO="${1:-}"
MODE="${2:-full}"

if [ -z "$SVO" ] || [ ! -f "$SVO" ]; then
    echo "Usage: $0 <svo2_path> [--diagnose]"
    echo ""
    echo "Available SVO files:"
    ls -lh recordings/walking_*/walking_*.svo2 2>/dev/null | tail -10
    exit 1
fi

SVO_DIR=$(dirname "$SVO")
TS=$(date +%Y%m%d_%H%M%S)
NPZ="${SVO_DIR}/replay_${TS}.npz"
TRACE="${SVO_DIR}/replay_${TS}_trace.csv"
LOG="${SVO_DIR}/replay_${TS}.log"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'

run_replay() {
    local label="$1"; shift
    local extra_args="$@"
    local log_file="${SVO_DIR}/diag_${TS}_${label}.log"
    echo -e "${YELLOW}── Trying: ${label} (extra: ${extra_args:-none}) ──${NC}"
    timeout 30 \
        env PYTHONPATH=src:src/perception/benchmarks python3 \
            src/perception/realtime/pipeline_main.py \
            --svo2 "$SVO" \
            --method B \
            --no-display \
            $extra_args 2>&1 | tee "$log_file" | tail -8
    local rc=$?
    local frames=$(grep -oP "frames_logged.*\K\d+" "$log_file" | tail -1)
    local fps_lines=$(grep -c "\[FPS\]" "$log_file")
    echo "  → rc=$rc  fps_msgs=$fps_lines  frames_logged=${frames:-?}"
    echo ""
}

echo "============================================================"
echo "  Replay walking SVO"
echo "  SVO:    $SVO ($(du -h "$SVO" | cut -f1))"
echo "  Output: $SVO_DIR/replay_${TS}.*"
echo "  Mode:   $MODE"
echo "============================================================"
echo ""

if [ "$MODE" = "--diagnose" ]; then
    echo "── Diagnostic: 옵션 격리 ──"
    run_replay "bare"                                              # 1
    run_replay "trace_only"      --trace-csv "${TRACE}.bare"       # 2
    run_replay "npz_only"        --record-pose-npz "${NPZ}.bare"   # 3
    run_replay "shmv2"           --enable-shm-v2                   # 4
    run_replay "plan_d"          --enable-plan-d                   # 5
    run_replay "shmv2_plan_d"    --enable-shm-v2 --enable-plan-d   # 6
    echo "════════ 진단 결과 ════════"
    echo "fps_msgs가 1~6인 경우 → 정상 종료 (SVO 짧음)"
    echo "fps_msgs가 30+ 인 경우 → 1Hz 로 stuck → 해당 옵션이 원인"
    echo ""
    echo "각 단계 로그:"
    ls -lh "${SVO_DIR}/diag_${TS}_"*.log
    exit 0
fi

# Full mode: walking_session.sh Phase 3 과 동일
echo "── Full replay (walking_session.sh Phase 3 와 동일 옵션) ──"
timeout 120 \
    env PYTHONPATH=src:src/perception/benchmarks python3 \
        src/perception/realtime/pipeline_main.py \
        --svo2 "$SVO" \
        --method B \
        --no-display \
        --record-pose-npz "$NPZ" \
        --trace-csv "$TRACE" \
        --enable-plan-d \
        --enable-shm-v2 \
        2>&1 | tee "$LOG" | tail -30

if [ -f "$NPZ" ] && [ -s "$TRACE" ]; then
    echo -e "\n${GREEN}✓ Replay 성공${NC}"
    echo "  NPZ:   $(ls -lh "$NPZ" | awk '{print $5}')"
    echo "  Trace: $(wc -l < "$TRACE") rows"
    echo ""
    echo "다음:"
    echo "  python3 scripts/analyze_walking_results.py $SVO_DIR"
else
    echo -e "\n${RED}✗ Replay 실패${NC} (npz=$([ -f "$NPZ" ] && echo Y || echo N), trace_lines=$([ -f "$TRACE" ] && wc -l < "$TRACE" || echo 0))"
    echo "원인 격리:"
    echo "  bash scripts/replay_walking_svo.sh $SVO --diagnose"
    exit 1
fi
