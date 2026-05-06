#!/usr/bin/env bash
# Plan v7 (2026-05-07) — zed_lag 21ms timeline 시각화
#
# nsys profile launch_clean — ZED capture pipeline + TRT inf 의 GPU 동시
# 활동 timeline 을 캡처. Round 5 분석 input.
#
# 사용법:
#   sudo bash scripts/nsys_zedlag.sh [duration_sec=20] [extra_args...]
# 예시:
#   sudo bash scripts/nsys_zedlag.sh 20 --diag-zed-lag
#   sudo bash scripts/nsys_zedlag.sh 20 --exposure-us 8000

set -e

DURATION="${1:-20}"
shift || true
EXTRA="$*"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT="/tmp/nsys_zedlag_${TS}"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: 이 스크립트는 sudo 로 실행. nsys + launch_clean 둘 다 root 필요."
    echo "Usage:  sudo bash scripts/nsys_zedlag.sh [duration_sec] [extra_args]"
    exit 1
fi

echo "=== Plan v7 nsys profile ==="
echo "  duration : ${DURATION}s"
echo "  output   : ${OUTPUT}.nsys-rep"
echo "  extra    : ${EXTRA}"
echo ""

# Force-overwrite 안전 — 이전 sample 이 lock 해 있을 수 있음
rm -f "${OUTPUT}.nsys-rep" "${OUTPUT}.qdrep" 2>/dev/null || true

# nsys profile + launch_clean.sh 직접 호출. Bash 안에서 sudo 안 함 (이미 root).
nsys profile \
    -t cuda,nvtx,osrt \
    --sample=none \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "$OUTPUT" \
    timeout "$DURATION" \
    bash "$ROOT/src/perception/CUDA_Stream/launch_clean.sh" \
    "$((DURATION - 2))" $EXTRA

echo ""
echo "=== nsys stats summary ==="
nsys stats "${OUTPUT}.nsys-rep" 2>&1 | tee "${OUTPUT}_stats.txt" | head -100

echo ""
echo "=== 분석 ==="
echo "  Full report : ${OUTPUT}.nsys-rep"
echo "  Stats text  : ${OUTPUT}_stats.txt"
echo ""
echo "  GUI 분석:    nsys-ui ${OUTPUT}.nsys-rep"
echo "  CLI top 10:  nsys stats --report cuda_kern_exec_sum ${OUTPUT}.nsys-rep | head"
