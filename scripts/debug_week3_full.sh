#!/usr/bin/env bash
# Sprint 1 Phase 2 — comprehensive debug session
# ═══════════════════════════════════════════════════════════════════════════
# Investigates 3 unresolved findings from ablation (commit 760131a):
#
#   F1) 03_cuda depth jump: 0.302 → 1.591 ms  (only in inline+CUDA)
#       Hypothesis: CUDA preprocess kernel on self._stream blocks ZED retrieve
#                   (which uses ZED's own CUDA stream).
#
#   F2) 04_both infer jump: 10.501 → 11.665 ms  (only in async+CUDA)
#       Hypothesis: async feeder process creates GPU/scheduling contention.
#
#   F3) Phase 1 (async) e2e regression: -1.58 ms claim doesn't replicate
#       (now +1.50 ms). Hypothesis: variance / different methodology.
#
# Plan (each scenario × REPEATS runs for statistics):
#   A) Repeatability of original 4 scenarios — variance / noise floor
#   B) Stream variant test (HWALKER_PREPROC_STREAM env var):
#         03_cuda + stream=trt     (current, control)
#         03_cuda + stream=null    (NULL stream — auto-syncs all)
#         03_cuda + stream=default (torch default + explicit wait_stream)
#   C) 04_both stream variants similarly
#
# Total: ~ (4 + 3 + 3) × REPEATS = 30 runs (5min each = 2.5 hours)
#
# Usage:
#     bash scripts/debug_week3_full.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVO="recordings/walking_20260518_115340/walking_20260518_115340.svo2"
OUTDIR=/tmp/debug_week3_full
TIMEOUT=90
REPEATS=3   # 3 runs per scenario for variance estimate

cd "$REPO"

if [ ! -f "$SVO" ]; then
    echo "ERROR: SVO not found: $SVO" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

# Pre-auth sudo (one-time)
echo "Sudo auth needed (one-time for SHM cleanup between runs):"
sudo -v || { echo "ERROR: sudo failed"; exit 1; }
( while true; do sleep 60; sudo -n true 2>/dev/null || exit; done ) &
KP=$!
trap "kill $KP 2>/dev/null || true" EXIT

# ──────────────────────────────────────────────────────────────────────────
# Scenario definition: label, env vars, pipeline args
# ──────────────────────────────────────────────────────────────────────────
SCENARIOS=(
    # F3 repeatability — baseline & async
    "baseline_inline_torch|--plan-d-mode inline|"
    "baseline_async_torch|--plan-d-mode async|"

    # F1 stream variants — inline+CUDA with different stream modes
    "cuda_inline_strm_trt|--plan-d-mode inline --use-cuda-preprocess|HWALKER_PREPROC_STREAM=trt"
    "cuda_inline_strm_null|--plan-d-mode inline --use-cuda-preprocess|HWALKER_PREPROC_STREAM=null"
    "cuda_inline_strm_default|--plan-d-mode inline --use-cuda-preprocess|HWALKER_PREPROC_STREAM=default"

    # F2 + F1 stream variants — async+CUDA
    "cuda_async_strm_trt|--plan-d-mode async --use-cuda-preprocess|HWALKER_PREPROC_STREAM=trt"
    "cuda_async_strm_null|--plan-d-mode async --use-cuda-preprocess|HWALKER_PREPROC_STREAM=null"
    "cuda_async_strm_default|--plan-d-mode async --use-cuda-preprocess|HWALKER_PREPROC_STREAM=default"
)

echo "════════════════════════════════════════════════════════════"
echo " Week 3 comprehensive debug — ${#SCENARIOS[@]} scenarios × $REPEATS runs"
echo " SVO:     $SVO"
echo " Out:     $OUTDIR"
echo "════════════════════════════════════════════════════════════"

for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r label args envvars <<< "$entry"
    echo ""
    echo "─── [$label] args='$args' env='$envvars' ───"

    for r in $(seq 1 $REPEATS); do
        sudo rm -f /dev/shm/hwalker_* 2>/dev/null || true

        log="$OUTDIR/${label}_run${r}.log"
        trace="$OUTDIR/${label}_run${r}_trace.csv"

        # Build env prefix
        env_prefix=""
        if [ -n "$envvars" ]; then
            env_prefix="$envvars "
        fi

        eval "${env_prefix}PYTHONPATH=src:src/perception/benchmarks timeout $TIMEOUT \
            python3 -u src/perception/realtime/pipeline_main.py \
                --svo2 \"$SVO\" \
                --method B --no-display \
                --enable-plan-d --enable-shm-v2 \
                --trace-csv \"$trace\" \
                $args \
                > \"$log\" 2>&1"
        rc=$?

        e2e=$(grep "e2e lat" "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+±[0-9]+\.[0-9]+ms' | head -1)
        if [ -z "$e2e" ]; then e2e="NO_E2E"; fi
        echo "  run${r}: rc=$rc  e2e=${e2e}"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo " All runs done. Analyzing..."
echo "════════════════════════════════════════════════════════════"
python3 scripts/analyze_week3_debug.py
