#!/usr/bin/env bash
# Sprint 1 Phase 2 — E2E gain ablation
# ──────────────────────────────────────────────────────────────────────────
# Runs 4 combinations on the SAME SVO to isolate each optimization's gain:
#   01_baseline:  inline Plan D + torch preprocess  (Sprint 1 START)
#   02_async:     async Plan D + torch preprocess   (+Phase 1)
#   03_cuda:      inline Plan D + CUDA preprocess   (+Week 3)
#   04_both:      async Plan D + CUDA preprocess    (+Phase 1 + Week 3)
#
# Expected (additive model):
#   Phase 1 gain = (01 − 02) e2e p50 difference
#   Week 3 gain  = (01 − 03) e2e p50 difference
#   Combined     = (01 − 04) — should equal Phase1 + Week3 if independent
#
# If 04 e2e does NOT equal 01 − Phase1 − Week3, the gains are NOT independent
# (e.g. one bottleneck moved elsewhere). The trace CSV stage breakdown will
# show where.
#
# Usage (from repo root):
#     bash scripts/run_sprint1_ablation.sh
#     python3 scripts/analyze_sprint1_ablation.py
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVO="recordings/walking_20260518_115340/walking_20260518_115340.svo2"
OUTDIR=/tmp/sprint1_ablation
TIMEOUT=120

cd "$REPO"

if [ ! -f "$SVO" ]; then
    echo "ERROR: SVO not found: $SVO" >&2
    echo "  Run from repo root: cd ~/realtime-vision-control" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

# 4 combinations
LABELS=(01_baseline 02_async 03_cuda 04_both)
ARGS=(
    "--plan-d-mode inline"
    "--plan-d-mode async"
    "--plan-d-mode inline --use-cuda-preprocess"
    "--plan-d-mode async --use-cuda-preprocess"
)

echo "════════════════════════════════════════════════════════════"
echo " Sprint 1 Phase 2 — E2E gain ablation"
echo " SVO:     $SVO"
echo " Out:     $OUTDIR"
echo " Timeout: ${TIMEOUT}s per run"
echo "════════════════════════════════════════════════════════════"

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    args="${ARGS[$i]}"
    echo ""
    echo "──── [${label}]  args: $args ────"

    # SHM cleanup between runs (single producer SHM v2 will reject re-create)
    sudo rm -f /dev/shm/hwalker_* 2>/dev/null || true

    # Run pipeline. timeout 124 exit code is OK (SVO finished early).
    PYTHONPATH=src:src/perception/benchmarks timeout "$TIMEOUT" \
        python3 -u src/perception/realtime/pipeline_main.py \
            --svo2 "$SVO" \
            --method B --no-display \
            --enable-plan-d --enable-shm-v2 \
            --trace-csv "$OUTDIR/${label}_trace.csv" \
            $args \
            > "$OUTDIR/${label}.log" 2>&1
    rc=$?

    # Summary (124 = timeout, OK)
    if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
        e2e_line=$(grep "e2e lat" "$OUTDIR/${label}.log" | tail -1 | xargs)
        fps_line=$(grep "FPS" "$OUTDIR/${label}.log" | tail -1 | xargs)
        echo "  ✓ ${e2e_line:-no e2e line}"
        echo "    ${fps_line:-no fps line}"
    else
        echo "  ✗ FAILED rc=$rc — check $OUTDIR/${label}.log"
        tail -10 "$OUTDIR/${label}.log" | sed 's/^/    /'
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo " All runs done. For full breakdown:"
echo "   python3 scripts/analyze_sprint1_ablation.py"
echo "════════════════════════════════════════════════════════════"
