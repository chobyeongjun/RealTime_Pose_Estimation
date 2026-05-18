#!/usr/bin/env bash
# Sprint 1 Phase 1 A.1: Plan D async test — 5/18 SVO replay 으로 측정.
#
# 2 conditions:
#   inline (baseline, == 5/18 A2):  17.56 ms p50 reference
#   async (new):                    expected 16.0-16.5 ms p50
#
# Usage:
#   sudo bash scripts/run_plan_d_async_test.sh \
#       recordings/walking_20260518_115340/walking_20260518_115340.svo2 \
#       [frames=1000]
set +e
set -o pipefail

if [ -z "$1" ]; then
    echo "Usage: sudo bash $0 <SVO_PATH> [frames=1000]"
    exit 1
fi

SVO_PATH="$1"
FRAMES="${2:-1000}"
TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="recordings/plan_d_async_${TS}"

if [ ! -f "$SVO_PATH" ]; then
    echo "ERROR: SVO not found: $SVO_PATH"
    exit 2
fi

ORIGINAL_USER="${SUDO_USER:-chobb0}"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run with sudo (nvpmodel + jetson_clocks)"
    exit 1
fi

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

echo "============================================================"
echo "  Plan D async test — A.1 effect measurement"
echo "  SVO: $SVO_PATH"
echo "  Frames: $FRAMES per condition"
echo "  Output: $OUT_ROOT"
echo "============================================================"

sudo -u "$ORIGINAL_USER" -- mkdir -p "$OUT_ROOT/inline" "$OUT_ROOT/async"

# System state
nvpmodel -m 0 2>&1 | sed 's/^/  /' || true
jetson_clocks 2>&1 | sed 's/^/  /' || true
{
    date
    echo "=== nvpmodel -q ==="
    nvpmodel -q 2>&1 || true
    echo ""
    echo "=== thermal ==="
    for tz in /sys/class/thermal/thermal_zone*; do
        type=$(cat "$tz/type" 2>/dev/null)
        temp=$(cat "$tz/temp" 2>/dev/null)
        echo "  $tz ($type): $((temp/1000))°C"
    done
    echo ""
    sudo -u "$ORIGINAL_USER" -- bash -c "cd '$(pwd)' && git rev-parse --short HEAD && git branch --show-current"
} > "$OUT_ROOT/system_state.txt" 2>&1
chown "$ORIGINAL_USER:$ORIGINAL_USER" "$OUT_ROOT/system_state.txt"

cleanup_shm() {
    # Force clean ALL hwalker_* SHM between conditions (prevent state pollution)
    rm -f /dev/shm/hwalker_pose /dev/shm/hwalker_pose_v2 \
          /dev/shm/hwalker_pose_cuda /dev/shm/hwalker_forecast 2>/dev/null
    # Kill any zombie processes
    pkill -9 -f plan_d_feeder 2>/dev/null
    sleep 1
}

run_cond() {
    local LABEL="$1"
    local FLAGS="$2"
    local OUT_DIR="$OUT_ROOT/$LABEL"
    echo ""
    echo "── $LABEL ──"
    echo "  Flags: $FLAGS"

    # CRITICAL: clean SHM before each condition to avoid stale state
    cleanup_shm
    echo "  (cleaned /dev/shm/hwalker_*)"

    # Use sudo -u (preserves environment correctly, unlike `su - -c`)
    sudo -u "$ORIGINAL_USER" -- bash -c "
        cd '$(pwd)' && \
        PYTHONPATH=src:src/perception/benchmarks timeout $((FRAMES / 30 + 120)) \
            python3 src/perception/realtime/pipeline_main.py \
                --svo2 '$SVO_PATH' --method B --no-display \
                --trace-csv '$OUT_DIR/trace.csv' \
                --record-pose-npz '$OUT_DIR/pose.npz' \
                --enable-plan-d --enable-shm-v2 \
                $FLAGS \
                2>&1 | tee '$OUT_DIR/run.log' | tail -25
    "
    if [ -f "$OUT_DIR/trace.csv" ]; then
        local lines=$(wc -l < "$OUT_DIR/trace.csv")
        echo -e "  ${GREEN}✓ trace.csv: $lines lines${NC}"
    else
        echo -e "  ${RED}✗ trace.csv missing${NC}"
    fi
}

# inline (baseline)
run_cond "inline" "--plan-d-mode inline"

# async (new)
run_cond "async" "--plan-d-mode async"

chown -R "$ORIGINAL_USER:$ORIGINAL_USER" "$OUT_ROOT"

echo ""
echo "============================================================"
echo "=== Summary ==="
echo "============================================================"
{
    date
    echo "Output: $OUT_ROOT"
    echo ""
    for cond in inline async; do
        echo "─── $cond ───"
        ls -lh "$OUT_ROOT/$cond/" 2>/dev/null
        if [ -f "$OUT_ROOT/$cond/trace.csv" ]; then
            echo "  trace lines: $(wc -l < "$OUT_ROOT/$cond/trace.csv")"
        fi
        echo ""
    done
    if [ -f /tmp/plan_d_feeder.log ]; then
        echo "─── /tmp/plan_d_feeder.log (last 20 lines) ───"
        tail -20 /tmp/plan_d_feeder.log
    fi
} | tee "$OUT_ROOT/run_summary.txt"
chown "$ORIGINAL_USER:$ORIGINAL_USER" "$OUT_ROOT/run_summary.txt"

echo ""
echo "Mac 분석:"
echo "  scp -r chobb0@JETSON_IP:$(pwd)/$OUT_ROOT ~/realtime-vision-control/recordings/"
echo "  python3 scripts/analyze_track_comparison.py $OUT_ROOT/  # auto-detect inline/async"
echo ""
echo -e "${GREEN}=== plan_d_async test 완료 ===${NC}"
