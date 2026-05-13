#!/usr/bin/env bash
# Walking session — 사용자 카메라 앞 walking 후 *진정 *Plan D real-data validation*.
#
# 사용 (Jetson, user mode):
#   bash scripts/walking_session.sh [duration_sec]
#
# Steps:
#   1. nvpmodel + jetson_clocks (sudo)
#   2. ZED record walking_NNNNNN.svo2 (사용자 카메라 앞 walking)
#   3. Pipeline replay --record-pose-npz + --enable-plan-d + --enable-shm-v2
#   4. analyze_trace + run_plan_d_offline (Plan D validation)
#
# 사용자 의무: walking 60s — camera 정면 ~2m, fitted pants, healthy gait.

set +e
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DURATION_S="${1:-60}"
TS=$(date +%Y%m%d_%H%M%S)
RECORDINGS_DIR="recordings/walking_${TS}"
SVO_PATH="${RECORDINGS_DIR}/walking_${TS}.svo2"
NPZ_PATH="${RECORDINGS_DIR}/walking_${TS}.npz"
TRACE_PATH="${RECORDINGS_DIR}/trace_${TS}.csv"
ANALYZE_PATH="${RECORDINGS_DIR}/analyze_${TS}.txt"
PLAN_D_PATH="${RECORDINGS_DIR}/plan_d_${TS}.log"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}ERROR: user 모드 실행. nvpmodel/jetson_clocks 만 sudo (script 내부).${NC}"
    exit 1
fi

mkdir -p "$RECORDINGS_DIR"

echo "============================================================"
echo "  Walking Session — ${DURATION_S}s"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "  Recordings: $RECORDINGS_DIR"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "============================================================"
echo ""

# ─── Phase 1: Performance mode ──────────────────────────────────────────
echo "── Phase 1: nvpmodel + jetson_clocks ──"
sudo nvpmodel -m 0 2>&1 | sed 's/^/  /' || true
sudo jetson_clocks 2>&1 | sed 's/^/  /' || true
echo ""

# ─── Phase 2: ZED SVO record ────────────────────────────────────────────
echo "── Phase 2: ZED SVO record (${DURATION_S}s walking) ──"
echo "  사용자 의무: 카메라 정면 ~2m, walking 시작 후 enter 누르세요"
echo "  녹화 path: $SVO_PATH"
echo ""
read -p "  walking 준비 됐으면 Enter 누르세요... " _

# Headless-safe: use Python ZED SDK directly. ZED_Explorer is a Qt GUI app
# and crashes over SSH (qt.qpa.xcb: could not connect to display).
if command -v ZED_Recorder >/dev/null 2>&1; then
    # Prefer ZED_Recorder CLI if available (some SDK installs include it).
    timeout "${DURATION_S}" ZED_Recorder "$SVO_PATH" --resolution SVGA --fps 120 \
        2>&1 | sed 's/^/  /' || true
elif python3 -c "import pyzed.sl" 2>/dev/null; then
    echo "  Using headless Python recorder (pyzed.sl)"
    python3 scripts/zed_svo_record.py "$SVO_PATH" \
        --duration "$DURATION_S" --resolution SVGA --fps 120 \
        2>&1 | sed 's/^/  /' || true
else
    echo -e "  ${RED}ERROR: ZED_Recorder CLI 도 없고 pyzed.sl 도 import 안 됨.${NC}"
    echo "  pyzed 설치: cd /usr/local/zed && python3 get_python_api.py"
    exit 1
fi

if [ ! -f "$SVO_PATH" ]; then
    echo -e "  ${RED}ERROR: SVO file 생성 X — $SVO_PATH${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ SVO recorded: $(ls -lh "$SVO_PATH" | awk '{print $5}')${NC}"
echo ""

# ─── Phase 3: Pipeline replay + dumps ───────────────────────────────────
echo "── Phase 3: Pipeline replay + pose npz dump + trace CSV + Plan D ──"
PYTHONPATH=src:src/perception/benchmarks timeout $((DURATION_S + 30)) \
    python3 src/perception/realtime/pipeline_main.py \
    --svo2 "$SVO_PATH" \
    --method B \
    --no-display \
    --record-pose-npz "$NPZ_PATH" \
    --trace-csv "$TRACE_PATH" \
    --enable-plan-d \
    --enable-shm-v2 \
    2>&1 | tee "$PLAN_D_PATH" | tail -20

if [ ! -f "$NPZ_PATH" ]; then
    echo -e "  ${RED}ERROR: pose npz 생성 X${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Pose npz: $(ls -lh "$NPZ_PATH" | awk '{print $5}')${NC}"
echo -e "  ${GREEN}✓ Trace CSV: $(ls -lh "$TRACE_PATH" | awk '{print $5}')${NC}"
echo ""

# ─── Phase 4: Trace analysis ────────────────────────────────────────────
echo "── Phase 4: RT trace analysis ──"
PYTHONPATH=src python3 scripts/analyze_trace.py "$TRACE_PATH" \
    2>&1 | tee "$ANALYZE_PATH" | tail -30
echo ""

# ─── Phase 5: Plan D offline validation ─────────────────────────────────
echo "── Phase 5: Plan D offline validation (real-data) ──"
PYTHONPATH=src python3 scripts/run_plan_d_offline.py "$NPZ_PATH" --plot \
    2>&1 | tee -a "$PLAN_D_PATH" | tail -20
echo ""

# ─── Summary ────────────────────────────────────────────────────────────
echo "============================================================"
echo "=== WALKING SESSION SUMMARY ==="
echo "============================================================"
echo "  Recordings: $RECORDINGS_DIR"
ls -lh "$RECORDINGS_DIR" | tail -10
echo ""
echo "  진정 *paste 의무:"
echo "    cat $ANALYZE_PATH | tail -40"
echo "    cat $PLAN_D_PATH | tail -30"
echo ""
echo -e "${GREEN}=== walking session 완료 ===${NC}"
