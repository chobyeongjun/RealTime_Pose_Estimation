# realtime-vision-control

**Vault**: `~/research-vault/realtime-vision-control/` — 실험/회의/논문 노트는 여기에도 동시 기록

## 프로젝트 개요
ZED X Mini + YOLO26s-lower6 TRT 기반 실시간 하체 포즈 추정 → POSIX SHM → C++ 임피던스 제어 → Teensy → AK60 케이블.

## 두 트랙 (브랜치 가이드)

| 브랜치 | 용도 | 코드 위치 | 검증 수치 |
|---|---|---|---|
| **`feature/track-a-onepipeline`** | mainline (Python single pipeline) | `src/perception/realtime/` + `src/perception/benchmarks/` | Python only **13.7ms / 73Hz**, Python+C++ 동시 14.6ms / 67Hz |
| **`feature/track-b-cuda-stream`** | 4-stage CUDA pipeline (실시간성 보장) | `src/perception/CUDA_Stream/` | p99 **19.8ms**, HARD LIMIT 위반 **0.031%** (300s 측정) |

**언제 어느 걸 쓰는가**:
- 일반 실험·디버깅: Track A (단순)
- 실제 환자 실험·논문 측정: Track B (20ms HARD LIMIT 보장)

## 핵심 제약 (수정됨 — 2026-04-20)

```
E2E latency : < 20ms HARD LIMIT
              Track B: e2e>20ms → valid=False → C++ control loop가 자동 skip
              Track A: soft warning만, hard guarantee 없음

Throughput  : ≥ 60Hz 권장, 67-73Hz 달성

Camera      : ZED X Mini SVGA 960×600 @ 120fps + PERFORMANCE depth
              NEURAL depth는 TRT YOLO와 GPU SM 경합으로 ×2.4 감속 → 영구 기각

Model       : yolo26s-lower6-v2 (TRT FP16, 6 keypoints: L/R hip/knee/ankle)
              imgsz=640 (480은 사용자 영구 거부)

Method      : B (IMU World Frame, static R, skip_imu=True)

Depth       : np.isfinite(z) and z > 0 가드 필수, copy=True 강제

CPU/RT      : Python cores 2-5, C++ cores 6-7, system 0-1
              SCHED_FIFO 90 (chrt -r 90 via launch_clean.sh)
              gc.disable() 필수

Safety      : C++ watchdog 0.2s → pretension 5N fallback
              5중 force clamp (max 70N AK60)
              POSIX SHM seqlock (half-write 방지)
```

## 절대 금지 (skiro-learnings 영구 기각)

| 금지 | 이유 |
|---|---|
| One Euro Filter (모든 variant) | Joints 0/6 — 2D keypoint 이동 시 depth NaN |
| 2D keypoint smoothing | 동일 |
| SegmentLengthConstraint on 2D | 피드백 루프, 한쪽 keypoint 고착 |
| GDM(X server) 끄기 | GMSL/CSI = EGL=X 필수, segfault + 리부팅 강제 |
| NEURAL/NEURAL_LIGHT depth | TRT와 GPU SM 경합 |
| imgsz 480 | 사용자 거부 |
| zero-copy depth (copy=False) | ZED 내부 버퍼 race |
| C++ loop rate < 100Hz | CPU 3% 이하, 이득 없음 |
| Python에서 Teensy 직접 송신 | C++ RT 보장·watchdog·force clamp 우회 |
| sagittal display + pipeline 한 프로세스 | FPS 반토막 (74→42Hz) |
| jetson_clocks 미적용 실행 | GPU 306MHz로 fall-back |
| `trt_pose_engine_zerocopy.py` (v1) | 단일 stream — Track B의 4-stream으로 대체됨 |
| TRT INT8 quantization (YOLO26s-lower6) | YOLO26s에서 INT8이 제대로 동작하지 않음 + keypoint 정확도 손실 허용 불가 |
| Depth decimation / depth skip | 모든 프레임에 3D pose 필요. depth 빠지면 그 프레임 무효 → 제어 루프 stale data |

## 구조

```
realtime-vision-control/
├── src/perception/
│   ├── realtime/         ← Track A 메인 (pipeline_main.py)
│   ├── benchmarks/       ← 공통 모듈 (zed_camera, trt_pose_engine, postprocess_accel)
│   ├── CUDA_Stream/      ← Track B (해당 브랜치에만)
│   └── models/           ← .pt, .onnx (Git LFS), .engine은 gitignore
├── docs/
│   ├── evolution/        ← 마스터 정리 (perception-evolution, why-it-got-faster, cuda-stream-architecture)
│   ├── experiments/      ← 실험별 (meta.yaml, benchmark-results.json)
│   ├── handovers/        ← 날짜별 세션 인수인계
│   ├── meetings/         ← 격주 교수님 미팅 자료
│   ├── paper/            ← 논문 outline
│   ├── hardware/         ← 하드웨어 스펙
│   └── lessons/          ← 설계 결정 근거
└── scripts/
    └── sync-from-vault.sh   research-vault → docs 동기화
```

## 실행 (Track A)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
cd src/perception/realtime
python3 pipeline_main.py --no-display --method B
```

## 실행 (Track B)

```bash
sudo src/perception/CUDA_Stream/launch_clean.sh 60
# 다른 터미널 (선택, 실험 시 끄기)
python3 -m perception.CUDA_Stream.view_sagittal
```

## 모델 다운로드
```bash
git lfs pull   # *.pt, *.onnx
# *.engine은 Jetson에서 trtexec로 빌드 (15~20분)
# ⚠️ TRT 10.3+ syntax: --workspace=4096 deprecated → --memPoolSize=workspace:4096
/usr/src/tensorrt/bin/trtexec \
    --onnx=src/perception/models/yolo26s-lower6-v2.onnx \
    --saveEngine=src/perception/CUDA_Stream/yolo26s-lower6-v2.engine \
    --fp16 --memPoolSize=workspace:4096
```

## 첫 세션 시작 시 확인 (다음 세션 누구든)
1. 이 문서의 "두 트랙" 표를 먼저 보고 어느 브랜치인지 확인
2. `docs/evolution/perception-evolution.md` 읽기 (전체 여정)
3. `docs/evolution/why-it-got-faster.md` 읽기 (최적화 원리)
4. `docs/evolution/cuda-stream-architecture.md` 읽기 (Track B 상세)
5. `docs/experiments/benchmark-results.json` (수치 raw)
