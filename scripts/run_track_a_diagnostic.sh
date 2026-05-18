#!/usr/bin/env bash
# Track A diagnostic — 5/13 vs 5/18 regression cause 진단 + Plan D cost 측정.
#
# Track B 는 SVO replay 안 함 (benchmark_stream.py 가 live camera only).
# 그래서 Track A 만 측정 + Track B 는 historical 비교.
#
# Usage:
#   sudo bash scripts/run_track_a_diagnostic.sh <SVO_PATH> [frames=1000]
#
# 2 conditions:
#   A1: Track A minimal      (--no-display, --trace-csv only)
#   A2: Track A full         (+ --enable-plan-d --enable-shm-v2 --record-pose-npz)
#
# Output:
#   recordings/track_a_diag_TS/
#     system_state.txt
#     A1_minimal/   trace.csv + run.log
#     A2_full/      trace.csv + run.log + pose.npz + plan_d.log

set +e
set -o pipefail

if [ -z "$1" ]; then
    echo "Usage: sudo bash $0 <SVO_PATH> [frames=1000]"
    exit 1
fi

SVO_PATH="$1"
FRAMES="${2:-1000}"
TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="recordings/track_a_diag_${TS}"

if [ ! -f "$SVO_PATH" ]; then
    echo "ERROR: SVO not found: $SVO_PATH"
    exit 2
fi

ORIGINAL_USER="${SUDO_USER:-chobb0}"
ORIGINAL_HOME=$(eval echo "~$ORIGINAL_USER")

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run with sudo (need nvpmodel + jetson_clocks)"
    exit 1
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "============================================================"
echo "  Track A Diagnostic — same SVO, 2 conditions"
echo "  SVO: $SVO_PATH"
echo "  Frames: $FRAMES per condition"
echo "  Output: $OUT_ROOT"
echo "============================================================"
echo ""

# Create dirs as user (Issue 1 fix)
su - "$ORIGINAL_USER" -c "cd $(pwd) && mkdir -p '$OUT_ROOT/A1_minimal' '$OUT_ROOT/A2_full'"

# ─── System state probe ────────────────────────────────────────────────
echo "── System state probe ──"
nvpmodel -m 0 2>&1 | sed 's/^/  /'
jetson_clocks 2>&1 | sed 's/^/  /' || true
{
    echo "=== timestamp ==="
    date
    echo ""
    echo "=== nvpmodel -q ==="
    nvpmodel -q 2>&1 || echo "nvpmodel error"
    echo ""
    echo "=== jetson_clocks --show ==="
    jetson_clocks --show 2>&1 | head -40 || true
    echo ""
    echo "=== thermal zones ==="
    for tz in /sys/class/thermal/thermal_zone*; do
        type=$(cat "$tz/type" 2>/dev/null)
        temp=$(cat "$tz/temp" 2>/dev/null)
        echo "  $tz ($type): $((temp/1000))°C"
    done
    echo ""
    echo "=== nvidia-smi ==="
    nvidia-smi 2>&1 | head -25 || echo "nvidia-smi N/A"
    echo ""
    echo "=== free -h ==="
    free -h
    echo ""
    echo "=== git rev-parse ==="
    su - "$ORIGINAL_USER" -c "cd $(pwd) && git rev-parse --short HEAD && git branch --show-current"
} > "$OUT_ROOT/system_state.txt" 2>&1
chown "$ORIGINAL_USER:$ORIGINAL_USER" "$OUT_ROOT/system_state.txt"
echo -e "  ${GREEN}✓ system_state.txt saved${NC}"
echo ""

# ─── Helper to run Track A ──────────────────────────────────────────────
run_condition() {
    local LABEL="$1"
    local EXTRA_FLAGS="$2"
    local OUT_DIR="$OUT_ROOT/$LABEL"
    local TRACE="$OUT_DIR/trace.csv"
    local NPZ="$OUT_DIR/pose.npz"
    local LOG="$OUT_DIR/run.log"
    local PLAND_LOG="$OUT_DIR/plan_d.log"

    echo "── $LABEL ──"
    echo "  Flags: $EXTRA_FLAGS"

    # Run as user (not root) so file ownership is user
    su - "$ORIGINAL_USER" -c "
        cd $(pwd)
        PYTHONPATH=src:src/perception/benchmarks timeout $((FRAMES / 30 + 60)) \
            python3 src/perception/realtime/pipeline_main.py \
                --svo2 '$SVO_PATH' \
                --method B \
                --no-display \
                --trace-csv '$TRACE' \
                --record-pose-npz '$NPZ' \
                $EXTRA_FLAGS \
                2>&1 | tee '$LOG' | tail -15
    "

    if [ -f "$TRACE" ]; then
        lines=$(wc -l < "$TRACE")
        echo -e "  ${GREEN}✓ trace.csv: $lines lines${NC}"
    else
        echo -e "  ${RED}✗ trace.csv missing${NC}"
    fi
    echo ""
}

# ─── A1: minimal ────────────────────────────────────────────────────────
run_condition "A1_minimal" ""

# ─── A2: full ───────────────────────────────────────────────────────────
run_condition "A2_full" "--enable-plan-d --enable-shm-v2"

# ─── Final chown (in case anything got root-owned) ─────────────────────
chown -R "$ORIGINAL_USER:$ORIGINAL_USER" "$OUT_ROOT"

# ─── Summary ────────────────────────────────────────────────────────────
echo "============================================================"
echo "=== Summary ==="
echo "============================================================"
{
    echo "Run completed: $(date)"
    echo "SVO: $SVO_PATH"
    echo "Frames: $FRAMES"
    echo ""
    ls -la "$OUT_ROOT/"
    echo ""
    for cond in A1_minimal A2_full; do
        echo "─── $cond ───"
        ls -lh "$OUT_ROOT/$cond/" 2>/dev/null
        echo ""
        if [ -f "$OUT_ROOT/$cond/trace.csv" ]; then
            lines=$(wc -l < "$OUT_ROOT/$cond/trace.csv")
            echo "  trace.csv: $lines lines"
        fi
        echo ""
    done
} | tee "$OUT_ROOT/run_summary.txt"
chown "$ORIGINAL_USER:$ORIGINAL_USER" "$OUT_ROOT/run_summary.txt"

echo ""
echo "결과 paste:"
echo "  cat $OUT_ROOT/run_summary.txt"
echo "  cat $OUT_ROOT/system_state.txt | head -50"
echo ""
echo -e "${GREEN}=== Track A diagnostic 완료 ===${NC}"
