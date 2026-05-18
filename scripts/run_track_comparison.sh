#!/usr/bin/env bash
# Track A vs Track B 의 comprehensive comparison — same SVO, 4 conditions.
#
# Usage:
#   sudo bash scripts/run_track_comparison.sh \
#       recordings/walking_20260518_115340/walking_20260518_115340.svo2 \
#       [frames_per_condition=1000]
#
# 4 conditions × ~10분 = ~40분 total.
#
# Output:
#   recordings/track_comparison_TS/
#     system_state.txt              — nvpmodel, jetson_clocks, thermal at run start
#     A1_track_a_minimal/           — Track A, no Plan D
#     A2_track_a_full/              — Track A, Plan D + SHM v2
#     B1_track_b_minimal/           — Track B, no Plan D
#     B2_track_b_full/              — Track B, Plan D
#     run_summary.txt               — header + warnings
#
# Each subdir has:
#   trace_TS.csv  — T0-T4 latency
#   *.npz          — Plan D output (A2/B2)
#   *.log          — stdout

set +e
set -o pipefail

if [ -z "$1" ]; then
    echo "Usage: sudo bash $0 <SVO_PATH> [frames=1000]"
    exit 1
fi

SVO_PATH="$1"
FRAMES="${2:-1000}"
TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="recordings/track_comparison_${TS}"

if [ ! -f "$SVO_PATH" ]; then
    echo "ERROR: SVO not found: $SVO_PATH"
    exit 2
fi

ORIGINAL_USER="${SUDO_USER:-chobb0}"
ORIGINAL_HOME="/home/$ORIGINAL_USER"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "============================================================"
echo "  Track Comparison — same SVO, 4 conditions"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "  SVO: $SVO_PATH"
echo "  Frames: $FRAMES per condition"
echo "  Output: $OUT_ROOT"
echo "============================================================"
echo ""

mkdir -p "$OUT_ROOT"

# ─── System state probe (once, before any run) ──────────────────────────
echo "── System state probe ──"
{
    echo "=== timestamp ==="
    date
    echo ""
    echo "=== nvpmodel -q ==="
    nvpmodel -q 2>&1 || echo "nvpmodel error"
    echo ""
    echo "=== nvpmodel set mode 0 (MAXN) ==="
    nvpmodel -m 0 2>&1 || true
    echo ""
    echo "=== jetson_clocks ==="
    jetson_clocks 2>&1 || true
    echo ""
    echo "=== thermal zones ==="
    for tz in /sys/class/thermal/thermal_zone*; do
        type=$(cat "$tz/type" 2>/dev/null)
        temp=$(cat "$tz/temp" 2>/dev/null)
        echo "  $tz ($type): $((temp/1000))°C"
    done
    echo ""
    echo "=== nvidia-smi ==="
    nvidia-smi 2>&1 | head -20 || echo "nvidia-smi N/A"
    echo ""
    echo "=== free -h ==="
    free -h
    echo ""
    echo "=== git rev-parse ==="
    su - "$ORIGINAL_USER" -c "cd $(pwd) && git rev-parse --short HEAD && git branch --show-current"
} > "$OUT_ROOT/system_state.txt" 2>&1
echo -e "  ${GREEN}✓ system state saved${NC}"
echo ""

# ─── Helper: run pipeline_main.py (Track A) ────────────────────────────
run_track_a() {
    local LABEL="$1"
    local EXTRA_FLAGS="$2"
    local OUT_DIR="$OUT_ROOT/$LABEL"
    mkdir -p "$OUT_DIR"
    local TRACE="$OUT_DIR/trace.csv"
    local NPZ="$OUT_DIR/pose.npz"
    local LOG="$OUT_DIR/run.log"

    echo "── $LABEL — Track A ──"
    echo "  Flags: $EXTRA_FLAGS"
    su - "$ORIGINAL_USER" -c "
        cd $(pwd)
        PYTHONPATH=src:src/perception/benchmarks timeout $((FRAMES / 50 + 30)) \
            python3 src/perception/realtime/pipeline_main.py \
                --svo2 '$SVO_PATH' \
                --method B \
                --no-display \
                --trace-csv '$TRACE' \
                --record-pose-npz '$NPZ' \
                $EXTRA_FLAGS \
                2>&1 | tee '$LOG' | tail -10
    " || true
    if [ -f "$TRACE" ]; then
        lines=$(wc -l < "$TRACE")
        echo -e "  ${GREEN}✓ trace.csv: $lines lines${NC}"
    else
        echo -e "  ${RED}✗ trace.csv missing${NC}"
    fi
    echo ""
}

# ─── Helper: run Track B (CUDA_Stream) ─────────────────────────────────
run_track_b() {
    local LABEL="$1"
    local EXTRA_FLAGS="$2"
    local OUT_DIR="$OUT_ROOT/$LABEL"
    mkdir -p "$OUT_DIR"
    local LOG="$OUT_DIR/run.log"

    echo "── $LABEL — Track B (CUDA_Stream) ──"
    echo "  Flags: $EXTRA_FLAGS"

    # Track B uses launch_clean.sh which expects no SVO arg. Use benchmark_stream directly
    # if available, otherwise note.
    local BENCH="src/perception/CUDA_Stream/benchmark_stream.py"
    if [ ! -f "$BENCH" ]; then
        echo -e "  ${YELLOW}⚠ benchmark_stream.py not found, skipping Track B${NC}"
        echo "skip: benchmark_stream.py missing" > "$LOG"
        echo ""
        return
    fi

    su - "$ORIGINAL_USER" -c "
        cd $(pwd)
        PYTHONPATH=src timeout $((FRAMES / 50 + 30)) \
            python3 $BENCH \
                --svo2 '$SVO_PATH' \
                --frames $FRAMES \
                --output-dir '$OUT_DIR' \
                $EXTRA_FLAGS \
                2>&1 | tee '$LOG' | tail -10
    " || true

    if [ -f "$OUT_DIR/trace.csv" ] || ls "$OUT_DIR"/*trace*.csv 2>/dev/null | head -1; then
        echo -e "  ${GREEN}✓ Track B output created${NC}"
    else
        echo -e "  ${YELLOW}⚠ Track B output uncertain — check $LOG${NC}"
    fi
    echo ""
}

# ─── A1: Track A minimal ─────────────────────────────────────────────────
run_track_a "A1_track_a_minimal" ""

# ─── A2: Track A + Plan D + SHM v2 ───────────────────────────────────────
run_track_a "A2_track_a_full" "--enable-plan-d --enable-shm-v2"

# ─── B1: Track B minimal ─────────────────────────────────────────────────
run_track_b "B1_track_b_minimal" ""

# ─── B2: Track B + Plan D ────────────────────────────────────────────────
run_track_b "B2_track_b_full" "--enable-plan-d"

# ─── Summary ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "=== Track Comparison Summary ==="
echo "============================================================"
{
    echo "Run started: $(date)"
    echo "SVO: $SVO_PATH"
    echo "Frames per condition: $FRAMES"
    echo ""
    echo "Output:"
    ls -la "$OUT_ROOT"/
    echo ""
    for cond in A1_track_a_minimal A2_track_a_full B1_track_b_minimal B2_track_b_full; do
        echo "─── $cond ───"
        if [ -d "$OUT_ROOT/$cond" ]; then
            ls -lh "$OUT_ROOT/$cond/" 2>/dev/null | tail -10
        else
            echo "  (not created)"
        fi
        echo ""
    done
} | tee "$OUT_ROOT/run_summary.txt"

echo ""
echo "Result paste 의무 — Mac 에서 분석 위해:"
echo "  cat $OUT_ROOT/run_summary.txt"
echo "  cat $OUT_ROOT/system_state.txt"
echo "  scp -r $OUT_ROOT MAC_USER@MAC_IP:~/realtime-vision-control/recordings/"
echo ""
echo -e "${GREEN}=== Track comparison 완료 ===${NC}"
