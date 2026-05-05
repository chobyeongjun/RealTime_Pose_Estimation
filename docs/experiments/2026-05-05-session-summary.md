# 2026-05-05 Session Summary

## 한 줄 요약

ROS2 출처를 raw SDK fact로 잘못 단정한 발언들 정정 + Codex 4 round 검증으로 **진짜 발견 2개** 확보 (`4 stream 사실상 직렬`, `post .cpu() host block`) + 다음 행동 명확화.

---

## 오늘 작업 timeline

| 시간순 | 작업 | 결과물 |
|---|---|---|
| 1 | Plan v6 + 측정 자동화 | `scripts/compare_runs.py` self-test 9/9 PASS (commit `6b1f602`) |
| 2 | ROS2 17-25ms 잘못된 단정 정정 | 7개 위치 정리 (commit `69e4b24`) |
| 3 | ZED X Mini 공식 spec 검증 | Global Shutter 확정 (commit `346fc37`) + vault `~/research-vault/10_Wiki/perception/zed-x-mini.md` 업데이트 |
| 4 | 데이터 흐름 설명 | Bridge / Pipeline thread / deque / H2D / queue / Global Shutter |
| 5 | maxlen=2 토론 | maxlen=1과 *기능적 동등* 결론 (사용자 catch) |
| 6 | Codex 4 round consultation (R1-R7) | 결정적 발견 2개 |
| 7 | torch/CUDA audit | 7개 의심 spot 분류 |
| 8 | C++ age gate 우선순위 결정 | 현재 단계 우선순위 낮음 (환자 실험 직전 재검토) |

---

## Codex 4 round 발견 핵심

**Codex 세션 ID**: `019df0ff-4719-72c0-bf88-841535caafc6` (이어가기 가능)

| Round | 질문 | 핵심 발견 |
|---|---|---|
| R1-R2 | latency lower bound, 20ms HARD LIMIT 가능성 | ⚠️ ROS2 17-25ms 인용은 raw SDK에 부적용 — 정정됨 |
| R3 | C++ age gate 위험도 | A=19ms일 때 C++ stale read 확률 90% (Codex 추정) |
| **R4** | **4 stream true overlap?** | **★ 우리 4 stream은 사실상 직렬 실행** (confidence 0.9) |
| R5 | ZED CUDA context 정체 | PyTorch primary와 다를 가능성 0.6 — ctypes로 검증 필요 |
| R6 | empty_cache() 정책 | hot path 호출 비권장 (1-10ms spike) |
| **R7-d** | **post `.cpu()` 영향** | **★ host thread block → 다음 frame enqueue 막음 = 직렬화 진짜 원인** |

---

## 진짜 약점 (재정렬)

이전: "latency 절대값" 줄이는 것이 목표  
이번 세션 정정: **측정 정의 모호성 + frame-level overlap 미구현**이 본질

### 약점 우선순위

1. **frame-level overlap 미구현** (R4): 4 stream이 직렬 실행 → 우리가 자랑한 "Track B 4 stream 구조"의 효과는 *bridge ↔ pipeline 분리*뿐, preproc/infer/post 사이는 직렬
2. **post host block** (R7-d): post 안 `.cpu()` 한 줄이 다음 frame enqueue 막는 핵심 병목
3. **측정 정의 모호성**: ts_ns 의미가 SHM에 *주석 없음*, valid_reason 없음, publish_done timestamp 없음
4. **ZED CUDA interop 미구현** (skiro 메모 2026-04-14): GPU 왕복 두 번, 미적용
5. **C++ age gate** (R3 발견): 환자 안전 구멍 — 단 *현재 단계*에선 20ms HARD LIMIT으로 대부분 커버

### 진짜 수정 lever (우선순위)

| Lever | 영향 추정 | 작업 시간 |
|---|---|---|
| **L1**: post `.cpu()` 제거 + packed D2H | frame-level overlap 가능 | ~3-4 hours |
| **L2**: P1 — SHM에 publish_done_ns + valid_reason 박기 | 측정 정의 anchoring | ~1.5-2 hours |
| **L3**: ZED CUDA interop (C++ extension) | bridge_proc -10~15ms 추정 | ~1-2 weeks (큰 변경) |
| ~~L4: C++ age gate~~ | 환자 안전 추가 방어선 | **환자 실험 직전 재검토** |

---

## 이미 적용된 것 (audit, vault + docs/evolution 검증)

- ✅ Head 17→6kpt (44→18ms)
- ✅ jetson_clocks + MAXN power
- ✅ CPU isolation (Python 2-5, C++ 6-7) + SCHED_FIFO 90
- ✅ gc.disable()
- ✅ 20ms HARD LIMIT frame-skip (300s 0.031% 위반, motor 도달 0 검증됨)
- ✅ POSIX SHM seqlock
- ✅ launch_clean.sh (Argus IPC 정리)
- ✅ --no-display
- ✅ skip_imu=True (Method B static R)
- ✅ PERFORMANCE depth (NEURAL 영구 기각)
- ✅ depth `copy=True` 강제
- ✅ 4 CUDA stream 분리 (단 R4: 실제 overlap은 안 일어남)
- ✅ DirectTRT (Ultralytics 우회)
- ✅ IMU quaternion N=20 평균
- ✅ Phase 0: true_e2e_ms decomposition (8fcfbcc)
- ✅ Phase A: consume-once + actual_publish + publish budget gate (d87957c)

## 영구 기각 (다시 검토 금지)

- One Euro Filter 모든 variant
- 2D keypoint smoothing
- SegmentLengthConstraint on 2D
- GDM(X server) 끄기
- NEURAL/NEURAL_LIGHT depth
- imgsz 480
- zero-copy depth (`copy=False`)
- C++ loop rate < 100Hz
- Python에서 Teensy 직접 송신
- sagittal display + pipeline 한 프로세스
- jetson_clocks 미적용
- TRT INT8 quantization (YOLO26s)
- Depth decimation / depth skip
- ROS2 wrapper docs의 17-25ms를 raw SDK fact로 인용 (이번 세션 추가)

---

## 사용자 액션 아이템 (do this)

### Phase 0 — 사실 확정 (35-40분, ★ 첫 번째)

| # | 작업 | 어디서 | 시간 | 산출물 |
|---|---|---|---|---|
| **A1** | ZED + PyTorch CUDA context check (ctypes script) | Jetson | 5분 | hex 두 개 — 같은가 다른가 |
| **A2** | nsys profile 1회 (frame-level overlap 검증) | Jetson | 30분 | `/tmp/phaseA_streams.nsys-rep` |

→ 이 두 결과로 *Codex R4, R5 진실* 확정. diagnose.py 설계 refine.

### Phase 1 — 측정 도구 + baseline (사용자: Jetson 측정만)

| # | 작업 | 어디서 | 시간 |
|---|---|---|---|
| **B1** | diagnose.py v1 작성 (Q1-Q4 + Q1'/Q2'/Q3') | Mac (제가) | 3-4 hours |
| **B2** | Jetson에서 baseline 측정 60s × 5회 | Jetson (사용자) | 30분 |
| **B3** | report.md 5개 paste 또는 scp | Mac으로 회수 | 5분 |

### Phase 2 — P1 (SHM timestamp + valid_reason)

| # | 작업 | 어디서 | 시간 |
|---|---|---|---|
| **C1** | SHM publisher 변경 (publish_done_ns + valid_reason 박기) | Mac (제가) | 1-1.5h |
| **C2** | self-test (synthetic SHM read) | Mac (제가) | 30분 |
| **C3** | Jetson에서 재측정 60s × 5 | Jetson (사용자) | 30분 |
| **C4** | before vs after 비교 (compare_runs.py) | Mac (제가, 자동) | 5분 |

### Phase 3 — L1 (post .cpu() 제거 + packed D2H)

| # | 작업 | 어디서 | 시간 |
|---|---|---|---|
| **D1** | gpu_postprocess.py 수정 — `.cpu()` 제거, GPU valid mask 유지 | Mac (제가) | 2-3h |
| **D2** | run_stream_demo.py — publish 직전 packed D2H pinned + non_blocking | Mac (제가) | 1h |
| **D3** | self-test | Mac (제가) | 30분 |
| **D4** | Jetson 재측정 + diagnose 비교 | Jetson + Mac | 1h |

### Phase 4 — L2 (ZED CUDA interop, *결과에 따라*)

A1 결과 (context 같은가)에 의존:
- **같은 context**: DLPack/PyTorch wrap 가능 — 1주 작업
- **다른 context**: C++ extension 필요 — 2주 작업

→ Phase 3까지 *명확한 latency 감소* 확인 후 진입 결정.

---

## 사용자가 해야 할 일 list (요약)

1. **A1**: 제가 ctypes script 짜드림 → Jetson에서 한 번 실행 → hex 두 개 paste
2. **A2**: 제가 nsys 명령 짜드림 → Jetson에서 30분 실행 → `.nsys-rep` 파일 paste 또는 분석 결과
3. **B2**: diagnose.py 완성되면 Jetson에서 60s × 5회 측정 → log paste
4. **C3**: P1 적용 후 같은 측정 60s × 5회
5. **D4**: L1 적용 후 같은 측정

저는 코드 작성/분석. 사용자는 *Jetson 실행 + 결과 paste*만.

---

## 오늘 commit history

```
346fc37  docs(hardware): ZED X Mini 공식 spec 검증 — Global Shutter 확정
69e4b24  fix(plan-v6): ROS2 출처를 raw SDK fact로 단정한 발언 모두 제거
6b1f602  feat(plan-v6): TDD-loop 비교 도구 + Codex 5문 답변 반영
```

origin/main 동기화 완료.

---

## Codex 응답 archive

| Round | 응답 파일 | 길이 |
|---|---|---|
| R1-Q1~Q5 | `/tmp/codexresp.iMQV8os` | 16 lines JSON (~14KB) |
| R1-R3 메타 | `/tmp/codex_resp2_path.txt` 가리키는 파일 | — |
| R4-R7 CUDA | `/tmp/codex_resp3_path.txt` 가리키는 파일 | — |

세션 ID로 다음 round 이어가기 가능.

---

## 다음 세션 시작 명령 (compact 후)

```
docs/experiments/2026-05-05-session-summary.md 읽고 Phase 0 (A1, A2) 부터 시작하자
```

또는 빠른 시작:
```
A1 ZED context check 스크립트 만들어줘
```
