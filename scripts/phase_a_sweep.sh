#!/usr/bin/env bash
# Phase A — 8-case ablation sweep (Codex consult 2026-05-10 권장).
#
# Codex Q2: 32 case 다 안 함. 8 critical combination 만 — measurement budget 보호.
#   1. baseline           — no γ, no async, no fusion, no graph, priority off
#   2. current_best       — γ + post-async (현재 production, p99=61.81ms)
#   3. fusion_only        — --post-fusion only (A.2 효과 단독 검증)
#   4. gamma_fusion       — γ + --post-fusion
#   5. target_A           — γ + post-async + post-fusion + priority infer-only
#   6. priority_off_ctrl  — target_A but priority=off (A.3 falsification)
#   7. graph_no_gamma     — post-async + fusion + graph + priority (γ 부재 시 graph 효과)
#   8. full_A             — γ + post-async + fusion + graph + priority infer-only (★ best)
#
# Mutually exclusive (Codex Q2):
#   --graph-extended without --post-fusion   (current post 가 graph-hostile)
#   --frame-overlap                          (DEPRECATED, p99 +10-15ms regression)
#
# 사용법:
#   sudo BRIDGE_CORES="6,7" BRIDGE_RT_PRIO=80 bash scripts/phase_a_sweep.sh
#   sudo PHASE_A_DURATION=120 bash scripts/phase_a_sweep.sh   # duration 변경
#
# Acceptance (Codex Q5):
#   target_A or full_A: true_e2e p99 < 55 ms AND actual_publish p99 < 57 ms → freeze.

set -e
set -o pipefail

TS="$(date +%Y%m%d_%H%M%S)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="/tmp/phase_a_sweep_${TS}"
mkdir -p "$OUTDIR"
DURATION="${PHASE_A_DURATION:-90}"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: sudo 필요"
    exit 1
fi

echo "=== Phase A — 8-case sweep (duration=${DURATION}s) ==="
echo "  output dir : ${OUTDIR}"
echo "  bridge env : BRIDGE_CORES=${BRIDGE_CORES:-unset}, BRIDGE_RT_PRIO=${BRIDGE_RT_PRIO:-unset}"
echo ""

FAILED_CASES=()

run_case() {
    local label="$1"
    shift
    local extra="$*"
    local logfile="${OUTDIR}/${label}.log"
    local rc=0
    echo "▶ ${label}  (extra: ${extra})"
    bash "$ROOT/src/perception/CUDA_Stream/launch_clean.sh" "$DURATION" $extra \
        2>&1 | tail -300 > "$logfile" || rc=$?
    if [ "$rc" -ne 0 ] || ! grep -q "true_e2e" "$logfile"; then
        echo "  ⚠️ FAILED (rc=${rc}, no final metrics)"
        FAILED_CASES+=("${label}")
    fi
    echo "  → ${logfile}"
}

COMMON="--no-constraints --strict-correctness"

run_case "1_baseline"           $COMMON --gpu-stream-priority off
run_case "2_current_best"       $COMMON --zed-cuda-interop --post-async
run_case "3_fusion_only"        $COMMON --post-fusion
run_case "4_gamma_fusion"       $COMMON --zed-cuda-interop --post-fusion
run_case "5_target_A"           $COMMON --zed-cuda-interop --post-async --post-fusion --gpu-stream-priority infer-only
run_case "6_priority_off_ctrl"  $COMMON --zed-cuda-interop --post-async --post-fusion --gpu-stream-priority off
run_case "7_graph_no_gamma"     $COMMON --post-async --post-fusion --graph-extended --gpu-stream-priority infer-only
run_case "8_full_A"             $COMMON --zed-cuda-interop --post-async --post-fusion --graph-extended --gpu-stream-priority infer-only

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo ""
    echo "⚠️ ============================================================"
    echo "⚠️ FAILED CASES (${#FAILED_CASES[@]}/8):"
    for c in "${FAILED_CASES[@]}"; do
        echo "⚠️   - ${c}"
    done
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
echo ""
echo "Acceptance criteria (Codex Q5):"
echo "  target_A or full_A: true_e2e p99 < 55ms AND actual_publish p99 < 57ms → freeze"
