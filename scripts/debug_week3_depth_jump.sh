#!/usr/bin/env bash
# Sprint 1 Phase 2 — debug 03_cuda 의 depth jump (0.3 → 1.59 ms)
#
# Step 1: reproducibility — run 03_cuda 5번 해서 depth p50 의 variance 확인.
#         만약 항상 ~1.5 ms 면 진짜 stream contention, ~0.3-1.5 ms 변동 이면 noise.
# Step 2: stream variant test (다음 commit) — CUDA preprocess 를 default stream 으로
#         돌려서 ZED retrieve 와 비경합 vs 현재 self._stream 경합 비교.
#
# Usage:
#     bash scripts/debug_week3_depth_jump.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVO="recordings/walking_20260518_115340/walking_20260518_115340.svo2"
OUTDIR=/tmp/debug_week3_depth
TIMEOUT=90
REPEATS=5

cd "$REPO"

mkdir -p "$OUTDIR"

# Pre-auth sudo (one-time)
echo "Sudo auth (one-time, for SHM cleanup):"
sudo -v
( while true; do sleep 60; sudo -n true 2>/dev/null || exit; done ) &
KP=$!
trap "kill $KP 2>/dev/null || true" EXIT

echo "════════════════════════════════════════════════════════════"
echo " Debug: 03_cuda depth jump reproducibility (${REPEATS} runs)"
echo "════════════════════════════════════════════════════════════"

for i in $(seq 1 $REPEATS); do
    echo ""
    echo "──── run ${i}/${REPEATS} ────"
    sudo rm -f /dev/shm/hwalker_* 2>/dev/null || true

    PYTHONPATH=src:src/perception/benchmarks timeout "$TIMEOUT" \
        python3 -u src/perception/realtime/pipeline_main.py \
            --svo2 "$SVO" \
            --method B --no-display \
            --enable-plan-d --enable-shm-v2 \
            --trace-csv "$OUTDIR/run${i}_trace.csv" \
            --plan-d-mode inline \
            --use-cuda-preprocess \
            > "$OUTDIR/run${i}.log" 2>&1
    rc=$?

    if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
        e2e=$(grep "e2e lat" "$OUTDIR/run${i}.log" | tail -1 | xargs)
        # PROFILE 마지막 block 에서 depth_3d 추출
        depth=$(grep -A 4 "PROFILE" "$OUTDIR/run${i}.log" | grep "depth_3d" | tail -1 | xargs)
        echo "  $e2e"
        echo "  $depth"
    else
        echo "  ✗ FAIL rc=$rc"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Stage breakdown across ${REPEATS} runs (trace CSV p50):"
echo "════════════════════════════════════════════════════════════"
python3 - <<'PY'
import csv, glob, numpy as np
from pathlib import Path
out = Path("/tmp/debug_week3_depth")
print(f"{'run':<8} {'fetch':>8} {'predict':>10} {'depth':>10} {'shm':>8}  ← p50 ms")
print("─" * 50)
for p in sorted(out.glob("run*_trace.csv")):
    rows = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t0 = float(row["t0_mono_ns"]) / 1e9
                t1 = float(row["t1_fetch_done_perf"])
                t2 = float(row["t2_predict_done"])
                t3 = float(row["t3_depth3d_done"])
                t4 = float(row["t4_publish_done_mono_ns"]) / 1e9  # monotonic, but ok for rel
                rows.append({
                    "fetch": (t1 - t0) * 1000,
                    "predict": (t2 - t1) * 1000,
                    "depth": (t3 - t2) * 1000,
                })
            except (ValueError, KeyError):
                continue
    if not rows:
        continue
    skip = min(100, len(rows) // 4)
    arr = rows[skip:]
    fetch_p = np.percentile([r["fetch"] for r in arr], 50)
    pred_p = np.percentile([r["predict"] for r in arr], 50)
    depth_p = np.percentile([r["depth"] for r in arr], 50)
    print(f"{p.stem:<8} {fetch_p:>7.3f}m {pred_p:>9.3f}m {depth_p:>9.3f}m  N={len(arr)}")
PY
echo ""
echo "Verdict:"
echo "  depth p50 spread < 0.3ms across runs  → NOISE (no real issue)"
echo "  depth p50 ≈ 1.5 ms in ALL runs        → consistent contention (real bug)"
echo "  depth p50 spread > 1.0 ms             → intermittent, depends on warmup"
