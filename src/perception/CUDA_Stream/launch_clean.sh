#!/bin/bash
# launch_clean.sh — 재부팅 없이 clean 상태로 publisher 시작
#
# Jetson Orin NX에서 CUDA_Stream publisher를 여러 번 재시작하면
# Argus SDK 내부 IPC 파일 (root 소유)이 누적되어 HARD LIMIT 위반율이
# 0.05% → 60%+로 악화됨. 이 스크립트는:
#   1. Argus daemon 재시작 (내부 상태 리셋)
#   2. stale IPC 파일 제거 (sudo 필요)
#   3. Clock 고정 (jetson_clocks)
#   4. Publisher 실행
#
# 사용법:
#   sudo ./launch_clean.sh [기간초수=60]
#
# sudo 암호 한 번만 요구. 이후 자동.

set -e

DURATION="${1:-60}"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$(cd "$DIR/../../.." && pwd)"
ENGINE="$DIR/yolo26s-lower6-v2.engine"

echo "=== CUDA_Stream clean launcher ==="
echo "  Repo: $ROOT"
echo "  Engine: $ENGINE"
echo "  Duration: ${DURATION}s"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: 이 스크립트는 sudo로 실행해야 합니다."
    echo "Usage:  sudo $0 [duration_sec]"
    exit 1
fi

ORIGINAL_USER="${SUDO_USER:-chobb0}"
ORIGINAL_HOME="/home/$ORIGINAL_USER"

echo "[1/5] 이전 publisher/viewer 정리..."
pkill -9 -f run_stream_demo 2>/dev/null || true
pkill -9 -f view_sagittal   2>/dev/null || true
sleep 1
# Verify old processes are actually dead (root can kill any user's process;
# if still alive after SIGKILL, something is stuck in D-state — wait longer).
for i in 1 2 3; do
    ALIVE=$(pgrep -f run_stream_demo 2>/dev/null | grep -v "^$$" || true)
    if [ -z "$ALIVE" ]; then break; fi
    echo "  [wait $i] pids still alive: $ALIVE — waiting 2s..."
    sleep 2
done
ALIVE=$(pgrep -f run_stream_demo 2>/dev/null | grep -v "^$$" || true)
if [ -n "$ALIVE" ]; then
    echo "  WARNING: could not kill pids $ALIVE (D-state?). GPU contention possible."
fi

echo "[2/5] Stale SHM / Argus IPC 파일 제거..."
rm -f /dev/shm/hwalker_pose_cuda 2>/dev/null || true
rm -f /dev/shm/sem.hwalker_pose_cuda 2>/dev/null || true
rm -f /dev/shm/sem.ipc_test_* 2>/dev/null || true
rm -f /dev/shm/sem.itc_test_* 2>/dev/null || true

echo "[3/5] nvargus-daemon 재시작..."
systemctl restart nvargus-daemon
sleep 3

echo "[4/5] jetson_clocks + MAXN..."
nvpmodel -m 0 > /dev/null 2>&1 || true
jetson_clocks

echo "[5/6] 메모리 lock unlimited (RT 전용)..."
# RT process가 page fault로 느려지지 않도록 memlock 해제
ulimit -l unlimited 2>/dev/null || true

echo "[6/6] Publisher 실행 (RT priority 90, cores 2-5, user=$ORIGINAL_USER)..."
echo ""

cd "$ROOT"
# RT priority는 root 권한으로 설정 → sudo -u가 user로 전환해도 상속됨.
# chrt -r 90: SCHED_FIFO priority 90 (OS scheduler가 다른 프로세스로
# 인한 선점 방지 → spike 큰 폭 감소).
# taskset -c 2,3,4,5: cores 2-5 격리 (system: 0-1, C++: 6-7 여유)
# sudo -u: user로 전환 (user의 torch 접근 가능)
exec chrt -r 90 sudo -u "$ORIGINAL_USER" -H env \
    PYTHONPATH="$ROOT/src" \
    PATH="$ORIGINAL_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    HOME="$ORIGINAL_HOME" \
    taskset -c 2,3,4,5 \
    python3 -u -m perception.CUDA_Stream.run_stream_demo \
        --engine "$ENGINE" \
        --schema lowlimb6 --resolution SVGA \
        --duration "$DURATION" \
        --bone-constraint \
        --velocity-bound-mps 8.0
