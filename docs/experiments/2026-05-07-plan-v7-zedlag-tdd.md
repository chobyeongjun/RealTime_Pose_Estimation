# Plan v7 — zed_lag 21ms 격파 (TDD loop, ultrathink)

작성: 2026-05-06 PM
실행: 2026-05-07 PM (내일 밤 4-6시간 예상)
이전: `2026-05-06-checkpoint-zedlag-pivot.md` (PM 체크포인트)

---

## 0. 결정적 단서

`zed_gpu_bridge.py:442-444` 주석:
> ts_ns = sensor exposure time (ZED hardware), bridge_start_ns = right after
> grab() returned. Their difference is ZED SDK's internal latency from
> exposure to grab completion (**typically 1-3 ms on Orin NX**).

우리 측정: **21ms** (16:23 + 17:18 양 run 재현).
괴리: **7-20배**. 주석이 *보편적 가정* 인지 *실제 우리 시스템 측정* 인지 검증 필요.

→ Plan v7 의 *진짜 출발점*: "주석의 1-3ms가 진짜인가? 21ms가 진짜인가? 진짜 21ms 라면 어디?"

---

## 1. zed_lag 21ms 의 물리적 분해 (이론 모델)

ZED X Mini (1MP, GMSL2, dual sensor, SVGA@120fps + PERFORMANCE depth):

| 단계 | 추정 (ms) | 제어 가능 | 근거 |
|---|---|---|---|
| Sensor exposure (AUTO) | 1-15 | △ MANUAL fixed | 조도 의존, AUTO 시 가변 |
| MIPI/CSI/GMSL transfer | 1-2 | ✗ | GMSL2 12Gbps 한계 |
| ISP (debayer, WB, color) | 2-4 | ✗ | NVIDIA Argus pipeline |
| Stereo rectification | 1-2 | ✗ | ZED SDK 내부 |
| Disparity (PERFORMANCE) | 3-6 | △ depth_mode 변경 | 영구 기각 NEURAL 외 비교 안 함 |
| Depth filter / refine | 1-2 | △ ZED conf | sensing_mode |
| Buffer DMA (host or GPU) | 1-2 | △ MEM::GPU | CUDA interop |
| **합 (이론)** | **10-33ms** | — | 우리 21ms = 중간값 |

**제어 가능 lever** (이론):
1. **Exposure MANUAL 짧게** → -3~10ms 가능 (밤 어두운 환경 큰 영향)
2. **depth_mode 비교** → PERFORMANCE 가 진짜 최저 latency 인가
3. **CUDA interop (MEM::GPU)** → DMA 단축 + bridge_proc 도 절감
4. **sensing_mode**: STANDARD vs FILL — FILL이 더 무거울 수 있음
5. **resolution sensor mode**: SVGA(960×600) 가 진짜 fast-path인가

**제어 불가**:
- ISP, rectification, GMSL transfer 자체 시간

---

## 2. 가설 트리 (격리 가능 7개)

| ID | 가설 | 격리 protocol | 작업 시간 |
|---|---|---|---|
| **G1** | ts_ns 가 진짜 *exposure* 시점이 아니라 *acquisition* (grab 시점)일 수 있다 → 21ms 측정 자체가 의미 다름 | ZED SDK doc 확인 + `sl.TIME_REFERENCE.IMAGE` 의 정확한 의미 | 30분 |
| **G2** | AUTO exposure가 어두운 환경에서 길게 (10-15ms) | exposure mode = MANUAL fixed (5/8/12ms) → zed_lag 변화 | 1시간 |
| **G3** | PERFORMANCE depth 자체가 무거움 (5-6ms) | depth_mode 비교 (PERFORMANCE vs QUALITY) | 1시간 |
| **G4** | sensing_mode = FILL 사용 중 (default STANDARD?) | sensing_mode 명시적 STANDARD 강제 | 30분 |
| **G5** | ZED SDK 버전 / JetPack 차이 | sdk_version 출력, release notes 확인 | 30분 |
| **G6** | nvargus-daemon 다른 process와 경합 | nvargus-daemon GST_DEBUG, 단일 process 환경 | 1시간 |
| **G7** | retrieve_image / retrieve_measure 자체가 zed_lag 안에 안 들어가야 하는데 들어가고 있다 | bridge code 의 timestamp 측정 위치 재검증 | 30분 |

**최우선**: G1 (측정 의미 자체) > G2 (AUTO exposure) > G3 (depth mode) > G7 (측정 버그)

---

## 3. 내일 밤 TDD loop (4-6시간) — *코드 도구는 이미 작성 완료* (commit 2026-05-06)

### Round 0 — zed_lag 21ms 원인 진단 (5 가설 격리, 90분) ★ 우선순위 1

**Red**: zed_lag 21ms 의 *진짜 원인* 모름. 우리 가설 (Jetson ISP buffer) 은 forum 검증된 *추정*. 직접 측정 안 됨.

**5 가설 + 격리 protocol**:

| ID | 가설 | 격리 방법 | 결정적 신호 |
|---|---|---|---|
| **H_zed1** | Jetson ISP frame buffer (2.5 frame buffered) | **fps sweep**: 120/60/30 fps 시 zed_lag 변화 | 비례 = ✓ ISP buffer<br>무관 = ✗ |
| **H_zed2** | ZED SDK 자체 buffering (Stereolabs internal) | SDK 4.x async option, depth_mode 변경 | options 영향 = ✓ |
| **H_zed3** | nvargus-daemon buffer count | `nvargus-daemon` config, GST_DEBUG | buffer count 변경 시 zed_lag 변화 |
| **H_zed4** | GMSL2 driver buffer | `dmesg \| grep -i gmsl`, `lspci -vvv` link | driver 진단 출력 |
| **H_zed5** | timestamp epoch mismatch (측정 자체 buggy) | `--diag-zed-lag` epoch sanity warning | EPOCH MISMATCH warn 검출 |

**도구 (이미 만듦)**:
- `--diag-zed-lag` flag → warmup 5 frames 다층 timestamp + epoch sanity check
- `scripts/zed_info_dump.py` → SDK version, firmware, RuntimeParameters defaults
- `scripts/zedlag_sweep.sh exposure/depth/sensing` → 자동 sweep + 비교 표

**명령** (순서):
```bash
# 0) ZED + system 정보 (한 번만)
python3 scripts/zed_info_dump.py 2>&1 | tee /tmp/r0_zed_info.log
sudo dmesg | grep -iE "gmsl|argus|csi" | tail -50 > /tmp/r0_dmesg.log
sudo cat /etc/argus/argus_camera.conf 2>/dev/null | head -50

# 1) baseline + diag (epoch sanity)
sudo bash src/perception/CUDA_Stream/launch_clean.sh 20 --diag-zed-lag 2>&1 | tee /tmp/r0_baseline.log
grep "\[zed_ts\]" /tmp/r0_baseline.log
# 검증: image_to_bridge 가 일관되게 ~21ms 면 H_zed5 배제 (epoch OK)

# 2) ★ fps sweep — H_zed1 결정적 격리
# SVGA@120fps (현재) — 1 frame = 8.3ms, 2.5 frame buffered = 21ms (가설)
# SVGA@60fps  — 1 frame = 16.7ms, 만약 ISP buffer 가설 ✓ 면 zed_lag ~42ms
# SVGA@30fps  — 1 frame = 33.3ms, 만약 ✓ 면 ~83ms
sudo bash src/perception/CUDA_Stream/launch_clean.sh 20 --fps 120 2>&1 | tee /tmp/r0_fps120.log
sudo bash src/perception/CUDA_Stream/launch_clean.sh 20 --fps 60  2>&1 | tee /tmp/r0_fps60.log
sudo bash src/perception/CUDA_Stream/launch_clean.sh 20 --fps 30  2>&1 | tee /tmp/r0_fps30.log
python3 scripts/parse_zedlag_results.py \
    --label "120fps" --label "60fps" --label "30fps" \
    /tmp/r0_fps120.log /tmp/r0_fps60.log /tmp/r0_fps30.log

# 3) tegrastats parallel — buffer / thermal 동시
tegrastats --interval 100 > /tmp/r0_tegra.log &
TEGRA_PID=$!
sudo bash src/perception/CUDA_Stream/launch_clean.sh 20 --diag-zed-lag
kill $TEGRA_PID
```

**Pass 조건 (각 가설별)**:
- H_zed1 ✓ 또는 ✗ 확정 (fps sweep)
- H_zed5 배제 (epoch sanity)
- H_zed4 진단 출력 확인
- H_zed2/H_zed3 는 후속 Round 에서 (option 변경)

**예상 결과**:
- H_zed1 ✓ 가 *가장 likely* (forum 검증)
- 만약 fps 와 zed_lag 비례 — ISP buffer 확정 → MAN exposure / depth_mode 외에 *큰 lever 없음*
- 만약 무관 — 가설 재검토 (H_zed2/3 경로)

---

### Round 1 — Exposure mode 격리 (1시간)
**Red**: `RuntimeParameters` exposure 가 AUTO. 어두운 환경 → exposure 길어짐 → zed_lag 늘어남 (가설).

**Green**: 4 케이스 실험 (각 5분 launch_clean.sh 20)
1. AUTO (baseline) → zed_lag p50/p99
2. MANUAL exposure 5ms (밝은 환경 가정)
3. MANUAL exposure 8ms (frame interval 8.3ms 같은)
4. MANUAL exposure 12ms (어두운 환경)

각 환경에서 같은 조명 조건 유지. `set_camera_settings(VIDEO_SETTINGS.EXPOSURE, ...)` API 사용.

**Refactor**: 결과 표 → 최적 exposure 선택. CLAUDE.md 에 "exposure MANUAL Xms" 명시 추가.

**Pass 조건**: zed_lag 가 *exposure 짧을수록 짧아짐* 확인. 줄어든 ms 수 정량화.

**예상 gain**: AUTO 시 10-15ms vs MANUAL 5ms 시 5-7ms → **-5~10ms**

---

### Round 2 — Depth mode 격리 (1시간)
**Red**: PERFORMANCE 만 측정. 다른 depth mode 의 latency 미상.

**도구 (이미 있음)**: `--depth-mode {PERFORMANCE,QUALITY,ULTRA}`, `scripts/zedlag_sweep.sh depth`

**명령**:
```bash
sudo bash scripts/zedlag_sweep.sh depth
```

**Pass 조건**: PERFORMANCE 가 진짜 최저 latency 확인 *또는* QUALITY 가 더 빠른 path 발견.
**예상 gain**: 0~3ms

---

### Round 3 — sensing_mode (45분)
**Red**: sensing_mode default 가 SDK 버전별 다름. FILL 이면 hole filling 로 무거움.

**도구 (이미 있음)**: `--sensing-mode {STANDARD,FILL}`, `scripts/zedlag_sweep.sh sensing`

**명령**:
```bash
sudo bash scripts/zedlag_sweep.sh sensing
```

**Pass 조건**: STANDARD vs FILL 차이 정량.
**예상 gain**: 0~3ms

---

### Round 4 — ZED CUDA interop prototype (1.5-2시간)
**Red**: bridge_proc 14.5ms 의 큰 부분이 host-side copy + H2D. ZED MEM::GPU 사용 시 zero-copy.

**Green**: prototype (별도 branch `feat/zed-cuda-interop-poc`)
1. `Mat(MEM.GPU)` 으로 retrieve_image / retrieve_measure
2. `Mat.get_pointer(MEM.GPU)` → cudaIpcMemHandle 또는 직접 device pointer 추출
3. PyTorch tensor: `torch.from_dlpack` 또는 `cuda.IpcMemoryHandle`
4. preprocessor 입력으로 직결, host pinned 우회

**위험**:
- ZED SDK 의 GPU buffer 가 *우리 CUDA context* 와 같은가
- ZED 다음 grab() 시 race 가능 (deep_copy 못 함, 동기화 필요)

**Refactor**: 실패 시 prototype 폐기. 성공 시 Phase 1 spec 으로.

**Pass 조건**: prototype run 가능 + bridge_proc 절감 확인 (실패해도 measurement 성공이면 OK).

**예상 gain**: -5~8ms on bridge_proc, zed_lag 자체 영향은 *적을* 가능성 (zed_lag 은 SDK internal)

---

### Round 5 — nsys profile + 결과 종합 (45분)
**Red**: 위 Round 들의 *시각적 검증* 안 됨. nsys 로 timeline 한 번 검증.

**도구 (이미 만듦)**: `scripts/nsys_zedlag.sh`

**명령**:
```bash
# 기본 (baseline 환경)
sudo bash scripts/nsys_zedlag.sh 20

# Round 1 결과 적용 (예: MAN 5ms 가 best 였다면)
sudo bash scripts/nsys_zedlag.sh 20 --exposure-us 5000 --diag-zed-lag
```

**확인 포인트**:
- ZED SDK thread 의 capture pipeline timeline
- grab() 호출 → return 사이의 GPU work
- ISP/depth 가 GPU compute 인지 다른 unit 인지

**Refactor**: 결과를 plan v8 (있을 시) 입력으로.

**Pass 조건**: zed_lag 21ms 의 *어떤 단계*가 dominant 인지 timeline 으로 확인.

---

## 4. 추가 다양한 + 정확한 탐색 path (10개)

이번 plan v7 직접 검증 후보 외에, *더 깊은 lever*:

| # | Path | 설명 | 가능 gain | 작업 시간 | 우선순위 |
|---|---|---|---|---|---|
| **A** | ZED SDK 4.2 → 4.3+ upgrade | release note 의 latency fix | 0-5ms | 하루 (검증 + 회귀 위험) | 중 |
| **B** | JetPack 6.x 검증 | L4T / ISP firmware 차이 | 0-3ms | 하루 (큰 변경) | 하 |
| **C** | GMSL link rate 확인 | `nvgstcapture` GST_DEBUG | 0-2ms | 1시간 | 중 |
| **D** | nvargus-daemon profile | `GST_DEBUG=nvarguscamerasrc:5` | 진단 only | 1시간 | 중 |
| **E** | sensor mode 다른 해상도 비교 | HD / FHD vs SVGA latency | -2~+5ms | 2시간 | 중 |
| **F** | IRQ affinity for CSI | `/proc/irq/<n>/smp_affinity` | 0-2ms | 1시간 | 중 |
| **G** | tegrastats 동시 모니터 | thermal/EMC 상관 | 진단 only | 30분 | 중 |
| **H** | ZED Forum 검색 — `latency Orin NX` | 사례 / fix | 정보 only | 1시간 | 상 |
| **I** | ZED X Mini firmware update | Stereolabs ZED tools | 0-5ms | 하루 (회귀 위험) | 중 |
| **J** | Argus API 직접 사용 | ZED SDK bypass + 자체 stereo | -10ms 가능, **stereo depth 자체 구현 필요** | 2-4주 | 영구 보류 |

**추가 우선순위 매핑**:
- 내일 밤: H (Forum 검색) 30분 → Round 0 전에 가설 좁힘
- Round 5 후: G (tegrastats) 동시 검증
- Plan v8 후보: A (SDK upgrade), C+D+F (system level)

---

## 5. Codex 라운드 — zed_lag 진단 (8개 질문)

`/tmp/codex_zedlag_q.txt` 별도 파일.

핵심 질문:
1. ZED SDK `TIME_REFERENCE.IMAGE` 의 *정확한 정의* — exposure 시점인가 acquisition 시점인가
2. ZED X Mini SVGA@120fps + PERFORMANCE depth 의 *normal latency* range (Orin NX 16GB)
3. zed_lag 21ms 가 *비정상 큰 값* 인가, 아니면 SVGA@120fps + depth 환경의 *natural floor* 인가
4. AUTO exposure → MANUAL 변경의 *실제 gain* 정량
5. depth_mode 변경의 latency 영향 (PERFORMANCE / QUALITY / ULTRA 비교)
6. ZED CUDA interop (MEM::GPU) prototype 시 *위험 요소* (context, race, sync)
7. nsys profile 에서 ZED capture pipeline 의 *어떤 row* 가 zed_lag 을 보여주는가
8. 만약 zed_lag 이 *unavoidable floor* 라면 환자 실험 가능 라인 도달 위한 *대안 architecture* — frame prediction (Kalman), Vision + IMU fusion 등

---

## 5b. Plan v8 outline — Frame overlap (Codex iterative)

사용자 제안 (2026-05-06 22시): "inf 끝 시점에 latest frame 으로 다음 inf"
= Frame-level overlap. plan v7 후 *최우선*.

진행 방식: **Codex 라운드 iterative** — 한 라운드 응답 받고 *후속 질문* 작성. 한 번에 spec 확정 안 함.

**Codex 라운드 1** (`/tmp/codex_overlap_q.txt`, 121 line, 작성 완료):
- Q1-2: 사용자 제안 vs Codex R6 P1 / 4 옵션 trade-off
- Q3-5: Graph ping-pong / PoseResult contract / 위험
- Q6: TDD 단계 분해 (Phase 1-6)
- Q7-8: 효과 정량 / 환자 실험 line 도달 확신도

**라운드 1 응답 후 결정 사항**:
- Phase 1-6 의 Pass 조건 + 작업 시간
- Output ring 의 정확한 구현 path (graph 2개 vs eager)
- PoseResult 1-frame-buffered contract

**라운드 2** (응답 후 작성):
- Phase 1 (재현성 + frame_id 추적) 의 정확한 코드 spec
- Phase 2 (TRT output 2-buffer) 의 변경 line 단위 spec
- 위험 mitigation 의 구체화

**라운드 3** (Phase 2 구현 후):
- 측정 결과 검토
- Phase 3-4 진행 결정

→ **iterative 진행 — Codex 응답 사이에 측정 + 검증 + 후속 질문**

## 6. 성공 기준

내일 밤 종료 시:
- ✓ zed_lag 21ms 의 *어디 어떻게 발생*인지 정량적 분해
- ✓ 줄일 수 있는 lever 1개 이상 확정 (-3ms 이상)
- ✓ 줄일 수 없는 floor 가 얼마인지 측정
- ✓ 환자 실험 가능 line (`true_e2e p99 < 20ms`) 도달 가능성 정량 평가
- ✓ Plan v8 / 환자 실험 일정 결정 가능

**실패 시나리오**:
- zed_lag 21ms 가 *물리적 floor* (모든 lever 시도 후 변화 없음) → architecture pivot:
  - Vision 38-50Hz + Foot IMU fusion (Kalman) — Vision delayed anchor + IMU integration
  - 또는 frame prediction (forward 20ms 예측)
  - 또는 ZED SDK 우회 (Argus, 큰 작업)

---

## 7. 작업 분배 — TDD discipline

각 Round 종료 시:
1. 측정 결과 → checkpoint 문서 update
2. 가설 검증 ✓/✗ 명시
3. Round N+1 *Red* 정의 update
4. 시간 초과 시 (해당 Round 에 +50% 이상) 다음 Round 로 이동, 미해결은 plan v8 후보로

**Stop 조건**:
- Round 0~3 중 하나라도 -10ms 이상 lever 발견 → 거기서 멈추고 측정 + commit
- Round 4 prototype 실패 → Round 5 (nsys) 직행
- 4시간 경과 → 측정 종료, 결과 정리

---

## 8. 작업 전 체크리스트 (내일 밤 시작 시)

```bash
# 1) 시스템 상태 확인
sudo nvpmodel -m 0 && sudo jetson_clocks
sudo jetson_clocks --show | grep -E "GPU|EMC"

# 2) 전체 코드 최신 — Plan v7 도구 모두 받음
git pull origin main
git log --oneline -5
ls scripts/zedlag_sweep.sh scripts/nsys_zedlag.sh scripts/zed_info_dump.py scripts/parse_zedlag_results.py

# 3) ZED SDK 정보 dump (Round 0 전, 한 번만)
python3 scripts/zed_info_dump.py 2>&1 | tee /tmp/zed_info.log

# 4) baseline 재측정 + diag (zed_lag 21ms 재현 확인 + timestamp 의미 격리)
sudo bash src/perception/CUDA_Stream/launch_clean.sh 20 --diag-zed-lag 2>&1 | tee /tmp/r0_baseline.log
grep "\[zed_ts\]" /tmp/r0_baseline.log

# 5) Round 1 (exposure sweep, 자동 4 case + 비교 표)
sudo bash scripts/zedlag_sweep.sh exposure

# 6) Round 2/3 도 동일 패턴
sudo bash scripts/zedlag_sweep.sh depth
sudo bash scripts/zedlag_sweep.sh sensing

# 7) 이 plan 재읽기
cat docs/experiments/2026-05-07-plan-v7-zedlag-tdd.md | head -50
```

## 8b. 코드 도구 인벤토리 (오늘 밤 미리 작성됨)

```
src/perception/CUDA_Stream/zed_gpu_bridge.py
  + exposure_us, sensing_mode, diag_zed_lag 옵션 (3 levers)
  + warmup 5 frames 다층 timestamp print

src/perception/CUDA_Stream/run_stream_demo.py
  + --exposure-us, --sensing-mode, --diag-zed-lag flag

scripts/zed_info_dump.py        — SDK 버전 / firmware / API 가용성
scripts/nsys_zedlag.sh          — nsys profile wrapper
scripts/parse_zedlag_results.py — launch output → 비교 표
scripts/zedlag_sweep.sh         — round 자동 sweep (exposure/depth/sensing)
```

---

## 9. 메모

- 5시간 작업 + zed_lag 발견 = *진단 우선* 의 가치 증명
- "lever 작업 전 측정 격리" — 이번 plan 의 모든 Round 가 이 원칙
- Codex 라운드는 *Round 0 시작 전* 진행해서 가설 트리 좁히기

---

## 10. 다음 문서

- 내일 작업 종료 시: `docs/experiments/2026-05-07-zedlag-results.md` 생성
- vault mirror: `~/research-vault/realtime-vision-control/handovers/2026-05-07-zedlag-results.md`
