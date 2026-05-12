#!/usr/bin/env bash
# Jetson 의 모든 test 의 single command — Level 1 (verify) + Level 2 (V4L2 smoke).
#
# 사용자 의지 (정확 + 시간 무관) — 한번에 모든 결과.
#
# Level 1 (~1분):
#   1.1 verify_shm_v2.py
#   1.2 verify_quality_dataset.py
#   1.3 batch_pose_compute.py --self-test
#
# Level 2 (~3-5분):
#   2.1 check_v4l2_capability.sh (V4L2 format 검증)
#   2.2 v4l2_capture.py main (30 frames bayer smoke)
#   2.3 vpi_pipeline.py self-test (numpy build_rectify_maps)
#   2.4 sparse_stereo_kernel.py self-test (synthetic stereo pair)
#
# 사용:
#   cd ~/realtime-vision-control
#   git pull origin local_backup
#   bash scripts/run_all_jetson_tests.sh 2>&1 | tee /tmp/all_tests.log
#   echo "exit=${PIPESTATUS[0]}"
#
# Output: 각 sub-test 의 PASS/FAIL + final summary.

set +e   # not exit on first fail (모든 sub-test 결과 수집)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Color codes (optional)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' NC=''
fi

declare -a RESULTS_NAMES
declare -a RESULTS_STATUS

run_test() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "============================================================"
    echo "  ▶ $name"
    echo "============================================================"
    # ★ Fix: pipefail + sub-shell — pipe 안의 *어느* command fail 시 *전체 fail*.
    # 기존 eval "cmd | tail -5" 의 exit code 가 tail (0) 만 보임 → silent pass bug.
    bash -c "set -o pipefail; $cmd"
    local rc=$?
    RESULTS_NAMES+=("$name")
    if [ "$rc" -eq 0 ]; then
        RESULTS_STATUS+=("PASS")
        echo -e "  ${GREEN}✓ $name PASS${NC}"
    else
        RESULTS_STATUS+=("FAIL (exit=$rc)")
        echo -e "  ${RED}✗ $name FAIL (exit=$rc)${NC}"
    fi
}

echo "============================================================"
echo "  Jetson 통합 test — Level 1 + Level 2"
echo "  $(date +%Y-%m-%d_%H:%M:%S)"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo "============================================================"

# ── Level 1: Existing single-command verify ─────────────────────────────

run_test "1.1 verify_shm_v2.py (SHM v2 publisher + reader)" \
    "python3 scripts/verify_shm_v2.py 2>&1 | tail -5"

run_test "1.2 verify_quality_dataset.py (Quality dataset I/O)" \
    "python3 scripts/verify_quality_dataset.py 2>&1 | tail -5"

run_test "1.3 batch_pose_compute --self-test (synthetic round-trip)" \
    "PYTHONPATH=src python3 scripts/batch_pose_compute.py --self-test 2>&1 | tail -3"

# ── Level 2: V4L2 + VPI + sparse stereo ─────────────────────────────────

run_test "2.1 check_v4l2_capability.sh (V4L2 format detect)" \
    "bash scripts/check_v4l2_capability.sh 2>&1 | grep -E 'Pixel Format|disto\[|libnvargus|PASS' | head -10"

run_test "2.2 v4l2_capture.py (30-frame bayer smoke)" \
    "PYTHONPATH=src python3 -m perception.CUDA_Stream.v4l2_capture --device /dev/video0 --n-frames 30 2>&1 | tail -5"

run_test "2.3 vpi_pipeline.py self-test (build_rectify_maps numpy)" \
    "PYTHONPATH=src python3 -m perception.CUDA_Stream.vpi_pipeline 2>&1 | tail -5"

run_test "2.4 sparse_stereo_kernel.py self-test (synthetic stereo, torch 의무)" \
    "PYTHONPATH=src python3 -m perception.CUDA_Stream.sparse_stereo_kernel 2>&1 | tail -5"

# ── Summary ─────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "=== FINAL SUMMARY ==="
echo "============================================================"
pass_count=0
fail_count=0
for i in "${!RESULTS_NAMES[@]}"; do
    name="${RESULTS_NAMES[$i]}"
    status="${RESULTS_STATUS[$i]}"
    if [ "$status" = "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $name"
        pass_count=$((pass_count + 1))
    else
        echo -e "  ${RED}✗${NC} $name [$status]"
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "  Total: $pass_count PASS, $fail_count FAIL"
echo ""
if [ "$fail_count" -eq 0 ]; then
    echo -e "${GREEN}=== ALL Jetson TESTS PASSED ===${NC}"
    echo "  → Week 1 Day 1 의 모든 Jetson test infra 정상 작동"
    echo "  → 다음 phase: V4L2 bypass full integration + Plan D EKF"
    exit 0
else
    echo -e "${RED}=== $fail_count test(s) FAILED ===${NC}"
    echo "  → log paste 필수: cat /tmp/all_tests.log"
    exit 1
fi
