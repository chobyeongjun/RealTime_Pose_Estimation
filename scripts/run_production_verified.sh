#!/usr/bin/env bash
# Production launcher with state verification (Sprint 1 Phase 2 close).
#
# Purpose: reproducible real-time measurement / production runs.
#   1. Verifies Jetson is at MAXN + jetson_clocks (auto-applies if not).
#   2. Records system state to log alongside measurement results.
#   3. Runs pipeline with VERIFIED-SAFE flags:
#        --plan-d-mode async        (Phase 1: 3 ms gain at low clock,
#                                    ~1.5 ms regression at high clock)
#        --use-cuda-preprocess      (Week 3: bit-equal cos=1.0, free GPU time)
#        --enable-plan-d --enable-shm-v2
#
# Why these flags:
#   - async:        sub-process Plan D EKF, forecast 1-frame lag absorbed
#                   by 50 ms forecast horizon. No accuracy loss.
#   - cuda-preproc: bit-equal vs torch (cos=1.000000, 12 cases). No accuracy loss.
#   - Tradeoff: at HIGH clock state, inline+torch may be ~1 ms faster e2e,
#               but loses CUDA accuracy headroom AND keeps Plan D EKF in
#               hot loop (blocks if EKF latency drifts in future iterations).
#               Chose async+cuda for: (a) no accuracy regression, (b) cleaner
#               architecture for Sprint 2 EKF refactor, (c) sub-20 ms safe.
#
# Usage:
#     bash scripts/run_production_verified.sh [pipeline_args...]
#
# Example (with SVO):
#     bash scripts/run_production_verified.sh \
#         --svo2 recordings/walking_20260518_115340/walking_20260518_115340.svo2 \
#         --method B --no-display \
#         --trace-csv /tmp/prod_trace.csv
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STATE_LOG="/tmp/prod_run_$(date +%Y%m%d_%H%M%S)_state.txt"
PIPELINE_LOG="/tmp/prod_run_$(date +%Y%m%d_%H%M%S)_pipeline.log"

echo "════════════════════════════════════════════════════════════"
echo " Production launcher — Sprint 1 Phase 2 verified config"
echo " State log:    $STATE_LOG"
echo " Pipeline log: $PIPELINE_LOG"
echo "════════════════════════════════════════════════════════════"

# ── Step 1: verify and apply Jetson state ────────────────────────────────
echo ""
echo "── Step 1: Jetson state ──"

# nvpmodel MAXN
CUR_MODE=$(sudo nvpmodel -q 2>/dev/null | grep -oE 'NV Power Mode.*' | head -1 || true)
if ! echo "$CUR_MODE" | grep -q "MAXN\|0$"; then
    echo "  ⚠ Not in MAXN. Applying:  sudo nvpmodel -m 0"
    sudo nvpmodel -m 0
fi
echo "  power: $(sudo nvpmodel -q 2>/dev/null | grep -oE 'NV Power Mode.*' | head -1)"

# jetson_clocks (best-effort detection via GPU freq)
GPU_CUR=$(cat /sys/class/devfreq/*/cur_freq 2>/dev/null | head -1)
GPU_MAX=$(cat /sys/class/devfreq/*/max_freq 2>/dev/null | head -1)
if [ -n "$GPU_CUR" ] && [ -n "$GPU_MAX" ] && [ "$GPU_CUR" != "$GPU_MAX" ]; then
    echo "  ⚠ GPU clock not at max ($GPU_CUR / $GPU_MAX). Applying:  sudo jetson_clocks"
    sudo jetson_clocks
fi
GPU_CUR=$(cat /sys/class/devfreq/*/cur_freq 2>/dev/null | head -1)
GPU_MAX=$(cat /sys/class/devfreq/*/max_freq 2>/dev/null | head -1)
echo "  GPU clock: $GPU_CUR / $GPU_MAX"

# Record full state for reproducibility
bash "$REPO/scripts/check_jetson_state.sh" > "$STATE_LOG" 2>&1 || true
echo "  state recorded → $STATE_LOG"

# ── Step 2: SHM cleanup ──────────────────────────────────────────────────
echo ""
echo "── Step 2: SHM cleanup ──"
sudo rm -f /dev/shm/hwalker_* 2>/dev/null || true
echo "  /dev/shm cleaned"

# ── Step 3: run pipeline ─────────────────────────────────────────────────
echo ""
echo "── Step 3: Pipeline (async + cuda preprocess) ──"
echo ""

# If user passed args, use them. Otherwise default test invocation.
if [ "$#" -gt 0 ]; then
    USER_ARGS="$@"
else
    # Default test (5/18 SVO)
    USER_ARGS="--svo2 recordings/walking_20260518_115340/walking_20260518_115340.svo2 \
        --method B --no-display \
        --trace-csv /tmp/prod_trace.csv"
fi

PYTHONPATH=src:src/perception/benchmarks \
    python3 -u src/perception/realtime/pipeline_main.py \
        --enable-plan-d --enable-shm-v2 \
        --plan-d-mode async \
        --use-cuda-preprocess \
        $USER_ARGS \
        2>&1 | tee "$PIPELINE_LOG"
rc=${PIPESTATUS[0]}

# ── Step 4: summary ──────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " Results"
echo "════════════════════════════════════════════════════════════"
echo "  exit code: $rc"
echo ""
grep "e2e lat\|FPS" "$PIPELINE_LOG" | tail -10 | sed 's/^/  /'
echo ""
echo "  Full log:   $PIPELINE_LOG"
echo "  State log:  $STATE_LOG"
