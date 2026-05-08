#!/usr/bin/env bash
# Plan v7 — zed_lag 측정 자동 sweep
#
# Round 1 (exposure) 또는 Round 2 (depth_mode) 의 여러 케이스 자동 실행 +
# 결과 한 번에 비교. 각 case 마다 launch_clean.sh 20s.
#
# 사용법:
#   sudo bash scripts/zedlag_sweep.sh fps           # Round 0 — 120/60/30 (★ ISP buffer 결정적 격리)
#   sudo bash scripts/zedlag_sweep.sh exposure      # Round 1 — AUTO/5/8/12ms
#   sudo bash scripts/zedlag_sweep.sh depth         # Round 2 — PERFORMANCE/QUALITY/ULTRA
#   sudo bash scripts/zedlag_sweep.sh sensing       # Round 3 — STANDARD/FILL
#   sudo bash scripts/zedlag_sweep.sh combinations  # Phase 4+5 — 8 조합 ablation (★)
#
# 주의: TDD discipline 상 단계별 측정 후 결정이 정도. 본 sweep 은
# *시간 절약* 용 — 각 round 의 4 case 를 손으로 안 돌리고 자동화.
# 단계별 가려면 launch_clean.sh 직접 호출.

set -e
# Codex review fix (2026-05-08): pipefail 활성 — `tee` 통과한 cmd 의 fail mask 방지.
# 기존 `bash launch_clean.sh ... | tee ... || true` 가 *모든 fail* 을 silent 통과.
# pipefail + run_case 안 의 `|| true` 는 *case 별 격리* 위해 유지.
set -o pipefail

ROUND="${1:-exposure}"
TS="$(date +%Y%m%d_%H%M%S)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="/tmp/zedlag_sweep_${ROUND}_${TS}"
mkdir -p "$OUTDIR"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: sudo 필요. Usage: sudo bash scripts/zedlag_sweep.sh {exposure|depth|sensing}"
    exit 1
fi

echo "=== Plan v7 sweep: ${ROUND} ==="
echo "  output dir : ${OUTDIR}"
echo ""

run_case() {
    local label="$1"
    shift
    local extra="$*"
    local logfile="${OUTDIR}/${label}.log"
    local rc=0
    echo "▶ Running case: ${label}  (extra: ${extra})"
    bash "$ROOT/src/perception/CUDA_Stream/launch_clean.sh" 20 $extra \
        2>&1 | tail -200 > "$logfile" || rc=$?

    # Codex γ review fix: || true 가 모든 fail 을 silent 통과 → ablation 측정
    # 가 fake-success. 변경: failure record + final metrics 검사 + summary 에 mark.
    # case isolation 위해 exit 안 함 (다음 case 계속), 단 log 에 명시.
    if [ "$rc" -ne 0 ] || ! grep -q "true_e2e" "$logfile"; then
        echo "  ⚠️ FAILED (rc=${rc}, no final metrics)"
        echo "FAILED rc=${rc}" >> "$logfile"
        FAILED_CASES+=("${label}")
    fi
    echo "  → ${logfile}"
    echo ""
}

# 실패 case 추적
FAILED_CASES=()

case "$ROUND" in
    fps)
        # Plan v7 Round 0 — ISP buffer 가설의 결정적 격리.
        # 가설: zed_lag 가 1/fps 에 비례 → frame N 개 buffered (Jetson ISP).
        # 결과:
        #   120fps zed_lag ≈ 21ms (현재) → 60fps 시 ≈ 42ms = ISP buffer ✓
        #                                → 60fps 시 변화 없음 = 다른 가설
        run_case "fps_120" --fps 120
        run_case "fps_60"  --fps 60
        run_case "fps_30"  --fps 30
        ;;
    exposure)
        run_case "auto"        ""
        run_case "manual_5ms"  --exposure-us 5000
        run_case "manual_8ms"  --exposure-us 8000
        run_case "manual_12ms" --exposure-us 12000
        ;;
    depth)
        run_case "performance" --depth-mode PERFORMANCE
        run_case "quality"     --depth-mode QUALITY
        run_case "ultra"       --depth-mode ULTRA
        ;;
    sensing)
        run_case "standard" --sensing-mode STANDARD
        run_case "fill"     --sensing-mode FILL
        ;;
    combinations)
        # Phase 4 D1 + Phase 5 D1 + γ (Codex R3+R4+R5) — 12 조합 ablation.
        # 어느 flag 조합이 진짜 -ms 효과 있는지 결정적 격리.
        # Expected (Codex R5):
        #   00_baseline ~65 / 03_overlap_async ~42-50 / 07_all_lever_old ~42-50 /
        #   08_interop_only ~60 (-5 bridge) / ★ 11_all_lever_new ~35-45 (best)
        # 측정 시간: 12 case × 25s ≈ 5분
        run_case "00_baseline"               ""
        run_case "01_overlap_only"           --frame-overlap
        run_case "02_async_only"             --post-async
        run_case "03_overlap_async"          --frame-overlap --post-async
        run_case "04_lpost_only"             --lpost-ablation
        run_case "05_overlap_lpost"          --frame-overlap --lpost-ablation
        run_case "06_async_lpost"            --post-async --lpost-ablation
        run_case "07_all_lever_old"          --frame-overlap --post-async --lpost-ablation
        # γ Phase (Codex R5) — ZED CUDA interop 추가 4 case
        run_case "08_interop_only"           --zed-cuda-interop
        run_case "09_interop_overlap"        --zed-cuda-interop --frame-overlap
        run_case "10_interop_async"          --zed-cuda-interop --post-async
        run_case "11_all_lever_new"          --zed-cuda-interop --frame-overlap --post-async --lpost-ablation
        ;;
    *)
        echo "ERROR: unknown round '${ROUND}'. Use {fps|exposure|depth|sensing|combinations}"
        exit 1
        ;;
esac

echo ""
# Codex γ review fix: 실패 case 명시.
if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "⚠️ ============================================================"
    echo "⚠️ FAILED CASES (${#FAILED_CASES[@]}/${#FAILED_CASES[@]}):"
    for c in "${FAILED_CASES[@]}"; do
        echo "⚠️   - ${c}"
    done
    echo "⚠️ Sweep result PARTIAL — check $logfile for each."
    echo "⚠️ ============================================================"
fi

echo ""
echo "=== Comparison table ==="
LABELS=()
FILES=()
for f in "$OUTDIR"/*.log; do
    LABELS+=("--label" "$(basename "$f" .log)")
    FILES+=("$f")
done

python3 "$ROOT/scripts/parse_zedlag_results.py" "${LABELS[@]}" "${FILES[@]}"

echo ""
echo "Logs: ${OUTDIR}/"
