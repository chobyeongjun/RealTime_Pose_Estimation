#!/usr/bin/env bash
# A.3 priority + bridge resource 의 진짜 효과 측정 — 4 case 자동 sweep.
#
# 측정 (각 60s, sleep 15s 사이):
#   case 1  — priority OFF  + bridge OFF  (모든 lever 끄기, baseline 의 baseline)
#   case 2  — priority infer-only + bridge ON (current best, 이전 ~61.81ms)
#   case 2b — priority all-high  + bridge ON (Codex falsification 검증)
#   case 3  — priority infer-only + bridge OFF (bridge resource 효과 격리)
#
# 사용 (Jetson SSH):
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   PYTHONPATH=src python3 scripts/jetson_smoke_test.py        # smoke 먼저
#   sudo bash scripts/measurement_priority_ablation.sh
#
# 끝나면 summary 가 자동 출력 — 그것 paste 부탁.

set -e
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="/tmp/priority_ablation_${TS}"
mkdir -p "$OUTDIR"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: sudo 필요. Usage: sudo bash scripts/measurement_priority_ablation.sh"
    exit 1
fi

echo ""
echo "============================================================"
echo "  A.3 priority + bridge resource ablation"
echo "  output dir  : ${OUTDIR}"
echo "  4 cases, each 60s, sleep 15s between"
echo "  total time  : ~5 minutes"
echo "============================================================"
echo ""

# Helper — run one case with optional env vars + args.
# tee 로 화면 + log 모두 출력 (사용자 진행 보임).
run_case() {
    local label="$1"
    local env_vars="$2"
    shift 2
    local args="$*"
    local logfile="${OUTDIR}/${label}.log"

    echo ""
    echo "============================================================"
    echo "▶ Case: ${label}"
    echo "  env  : ${env_vars:-(none)}"
    echo "  args : ${args}"
    echo "  → ${logfile}"
    echo "============================================================"

    if [ -n "${env_vars}" ]; then
        eval "${env_vars} bash ${ROOT}/src/perception/CUDA_Stream/launch_clean.sh 60 ${args}" 2>&1 | tee "${logfile}"
    else
        bash "${ROOT}/src/perception/CUDA_Stream/launch_clean.sh" 60 ${args} 2>&1 | tee "${logfile}"
    fi

    echo ""
    echo "  ✓ ${label} done"
    echo "  sleep 15s for process/SHM/Argus cleanup..."
    sleep 15
}

# ----- case 1: priority OFF + bridge OFF (baseline 의 baseline) -----
run_case "1_off_nobr" "" \
    --no-constraints --strict-correctness \
    --gpu-stream-priority off

# ----- case 2: priority infer-only + bridge ON (current best) -----
run_case "2_infer_br" 'BRIDGE_CORES="6,7" BRIDGE_RT_PRIO=80' \
    --no-constraints --strict-correctness \
    --zed-cuda-interop --post-async --gpu-stream-priority infer-only

# ----- case 2b: priority all-high + bridge ON (Codex falsification) -----
run_case "2b_all_br" 'BRIDGE_CORES="6,7" BRIDGE_RT_PRIO=80' \
    --no-constraints --strict-correctness \
    --zed-cuda-interop --post-async --gpu-stream-priority all-high

# ----- case 3: priority infer-only + bridge OFF (bridge 효과 격리) -----
run_case "3_infer_nobr" "" \
    --no-constraints --strict-correctness \
    --zed-cuda-interop --post-async --gpu-stream-priority infer-only

# =====================================================================
# Comparison + per-case metrics (사용자 paste 용)
# =====================================================================
echo ""
echo "============================================================"
echo "=== Comparison table (parse_zedlag_results.py) ==="
echo "============================================================"
LABELS=()
FILES=()
for f in "${OUTDIR}"/*.log; do
    LABELS+=("--label" "$(basename "$f" .log)")
    FILES+=("$f")
done
python3 "${ROOT}/scripts/parse_zedlag_results.py" "${LABELS[@]}" "${FILES[@]}" || true

echo ""
echo "============================================================"
echo "=== Per-case key metrics ==="
echo "============================================================"
for f in "${OUTDIR}"/*.log; do
    echo ""
    echo "--- $(basename "$f" .log) ---"
    grep -E "StreamManager priority|bridge thread CPU|SCHED_FIFO|true_e2e \(cam|actual_publish \(cam|decomposition p50/p99|HARD LIMIT 20 ms .true_e2e|→ [0-9]+ frames" "$f" | head -15 || true
done

echo ""
echo "============================================================"
echo "Logs : ${OUTDIR}/"
echo "Done. Paste 'Comparison table' + 'Per-case key metrics' 부탁."
echo "============================================================"
