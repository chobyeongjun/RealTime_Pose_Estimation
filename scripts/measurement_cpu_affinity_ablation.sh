#!/usr/bin/env bash
# CPU affinity + RT priority 6-case ablation — bridge thread 의 정직한 위치 결정.
#
# Background: CLAUDE.md 의 Python cores 2-5 / C++ cores 6-7 spec 와
# commit 69f918d 의 BRIDGE_CORES="6,7" 가 *같은 cores* 사용 → C++ 실행 시 충돌.
# 환자 실험 시 latency 회귀 risk. 측정으로 진짜 정답 결정.
#
# 6 cases (각 60s, sleep 15s 사이):
#   A) no_env       — BRIDGE 환경 변수 미설정 (kernel inherit)
#   B) br67_rt80    — BRIDGE 6,7 RT 80 (현재 commit 69f918d, C++ cores 와 동일)
#   C) br45_rt80    — BRIDGE 4,5 RT 80 (Python cores 의 일부, C++ 와 분리)
#   D) br4_rt80     — BRIDGE 4 single RT 80 (deterministic)
#   E) br01_rt80    — BRIDGE 0,1 RT 80 (system cores 와 충돌 risk)
#   F) br67_rt99    — BRIDGE 6,7 RT 99 (C++ 의 90 보다 high — bridge 가 C++ 우선)
#
# 모든 case 는 production default flags 사용:
#   --no-constraints --strict-correctness --zed-cuda-interop --post-async
#   (--gpu-stream-priority all-high 가 default, commit f551dba)
#
# 사용:
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   sudo bash scripts/measurement_cpu_affinity_ablation.sh
#
# 끝나면 자동 summary + per-case metrics. paste 부탁.

set -e
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="/tmp/cpu_affinity_${TS}"
mkdir -p "$OUTDIR"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: sudo 필요. Usage: sudo bash scripts/measurement_cpu_affinity_ablation.sh"
    exit 1
fi

echo ""
echo "============================================================"
echo "  CPU affinity + RT priority ablation (6 cases)"
echo "  output dir : ${OUTDIR}"
echo "  commit     : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  60s × 6 + sleep × 5 = ~6.5 min"
echo "============================================================"
echo ""

COMMON_ARGS="--no-constraints --strict-correctness --zed-cuda-interop --post-async"

run_case() {
    local label="$1"
    local env_vars="$2"
    local logfile="${OUTDIR}/${label}.log"

    echo ""
    echo "============================================================"
    echo "▶ Case: ${label}"
    echo "  env  : ${env_vars:-(none, kernel inherit)}"
    echo "  args : ${COMMON_ARGS}"
    echo "  → ${logfile}"
    echo "============================================================"

    if [ -n "${env_vars}" ]; then
        eval "${env_vars} bash ${ROOT}/src/perception/CUDA_Stream/launch_clean.sh 60 ${COMMON_ARGS}" 2>&1 | tee "${logfile}"
    else
        bash "${ROOT}/src/perception/CUDA_Stream/launch_clean.sh" 60 ${COMMON_ARGS} 2>&1 | tee "${logfile}"
    fi

    echo ""
    echo "  ✓ ${label} done"
    echo "  sleep 15s for cleanup..."
    sleep 15
}

# A — no env (kernel inherit)
run_case "A_no_env"     ""

# B — current commit 69f918d (C++ cores)
run_case "B_br67_rt80"  'BRIDGE_CORES="6,7" BRIDGE_RT_PRIO=80'

# C — bridge separate (Python cores 의 일부, C++ 분리)
run_case "C_br45_rt80"  'BRIDGE_CORES="4,5" BRIDGE_RT_PRIO=80'

# D — single core (deterministic)
run_case "D_br4_rt80"   'BRIDGE_CORES="4"   BRIDGE_RT_PRIO=80'

# E — system cores
run_case "E_br01_rt80"  'BRIDGE_CORES="0,1" BRIDGE_RT_PRIO=80'

# F — high RT priority (C++ 보다 high)
run_case "F_br67_rt99"  'BRIDGE_CORES="6,7" BRIDGE_RT_PRIO=99'

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "============================================================"
echo "=== Comparison table ==="
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
    grep -E "StreamManager priority|bridge thread CPU|SCHED_FIFO|true_e2e \(cam→gpu_done\) p50|actual_publish \(cam→shm\) p50|decomposition p50/p99" "$f" | tail -8 || true
done

echo ""
echo "============================================================"
echo "Logs : ${OUTDIR}/"
echo "Done. Comparison table + Per-case metrics paste 부탁."
echo "============================================================"
