# 2026-05-08 Checkpoint — 진짜 baseline 확정 + Codex R1 + plan v8 outline

이전: `2026-05-06-checkpoint-zedlag-pivot.md`, `2026-05-07-plan-v7-zedlag-tdd.md`
다음: Phase 1 작업 (PipelineTick contract refactor) — 2026-05-09 (토) 시작

---

## 한 줄 요약

NoMachine 가 GPU 경합 원인. SSH 로 전환 후 e2e p99 19.27ms (HARD 20ms 안). 진짜 baseline = 65ms. Codex R1 검증 — frame overlap 은 *minimal patch 아님*, *full pipeline contract refactor*. HARD 20ms 도달 *물리적 불가능*, 환자 실험 gate 50ms 재정의 필요.

---

## 1. NoMachine 발견 (오전 측정 악화의 진짜 원인)

### 진단
오전 13:34 측정 e2e p99 28.5ms (어제 22.5ms 보다 +6ms 악화), HARD (e2e basis) 위반 71% (어제 4%).

### tegrastats + ps 분석
```
nxnode    6.4% CPU   /usr/NX/bin/nxnode.bin       ← NoMachine 원격 데스크톱
nxcodec   5.0% CPU   /usr/NX/bin/nxcodec.bin      ← H.264 codec (NVENC GPU encode)
Xorg      4.0% CPU   gnome-shell render
GR3D_FREQ idle 0~37% spike  ← GUI render 가 GPU SM 점유
```

### 가설 검증 (NoMachine OFF + SSH 전환)
| 지표 | NoMachine ON | **NoMachine OFF** | 어제 16:23 |
|---|---|---|---|
| e2e p99 (GPU only) | 26.4 | **19.27** | 22.5 |
| true_e2e p99 | 86.3 | **65.0** | 67.3 |
| HARD (e2e) 위반 | 55% | **0.22%** | 3.92% |
| Hz | 40.8 | **51.6** | 50.3 |

→ **NoMachine 가 진짜 원인 확정**. 어제 67ms 도 NoMachine 영향 받았을 가능성. 진짜 floor = **65ms**.

### 영구 protocol
실험/측정 시 **반드시 SSH** 사용:
```bash
sudo systemctl stop nxserver       # NoMachine 종료
ssh chobb0@<jetson-ip>             # SSH 전환
```

vault skiro-learnings 에 CRITICAL lesson 추가됨.

---

## 2. Plan v7 Round 0-3 측정 결과 (이미 완료, 결과 정리)

### Round 0 — fps sweep (★ 결정적)

| fps | zed_lag | 1 frame interval |
|---|---|---|
| 120 | 21.9 ms | 8.3 ms |
| 60 | 28.5 ms | 16.7 ms |
| 30 | 39.6 ms | 33.3 ms |

**모델 (3점 fit)**: `zed_lag(fps) ≈ 14 ms (fixed) + 700/fps`
- 14ms = ISP/GMSL/disparity (fixed cost)
- 700/fps = 1 frame buffered (= 1/fps × ~840Hz)

**이전 가설 폐기**: "2.5 frame buffered" 폐기. 진짜는 *fixed + 1 frame*.

**함의**: zed_lag floor @ 120fps = 22ms (현재 측정 일치). fps 더 못 올림 (max). MANUAL exposure 효과 0. **zed_lag 자체 절감 불가능**.

### Round 1 — Exposure (효과 0)

| 모드 | zed_lag | true_e2e p99 |
|---|---|---|
| AUTO | 22.3 | 67.83 |
| MAN 5ms | 21.7 | 66.47 |
| MAN 8ms | 21.9 | 68.31 |
| MAN 12ms | 21.7 | 66.00 |

차이 0.6ms 이내 noise. 환경 밝음 → AUTO 가 이미 짧은 exposure. **exposure lever 폐기**.

### Round 2 — Depth mode

| 모드 | zed_lag | Hz |
|---|---|---|
| **PERFORMANCE** | **21.2** | **50.9** ✓ floor |
| QUALITY | 36.7 | 22.0 (+15ms) |
| ULTRA | failure | — |

PERFORMANCE 가 진짜 최저. ZED SDK 5.2.1 가 NEURAL 권장 (deprecated 경고) — 무시 (TRT 경합 영구 기각).

### Round 3 — Sensing mode

| 모드 | zed_lag | bridge_p50 | sum |
|---|---|---|---|
| STANDARD | 21.4 | 14.8 | 36.2 |
| FILL | 29.1 | 7.5 | 36.6 |

**Anchor 이동만, 효과 0**. STANDARD 유지.

---

## 3. Variance baseline 5회 (Codex #21 부분 수렴)

NoMachine OFF + jetson_clocks lock + SSH 환경. 30초 cooldown 사이에 5회 측정.

| run | Hz | e2e p99 | true_e2e p99 | zed_lag | bridge p50 | pipeline p50 |
|---|---|---|---|---|---|---|
| r1 | 51.9 | 19.38 | 64.64 | 21.0 | 14.5 | 16.7 |
| r2 | 51.2 | 19.37 | 64.31 | 21.3 | 14.4 | 17.0 |
| r3 | 52.8 | 18.63 | 67.02 | 21.0 | 14.5 | 16.3 |
| r4 | 51.6 | 19.07 | 64.43 | 21.2 | 14.3 | 16.8 |
| r5 | 50.4 | 19.90 | 64.72 | 21.7 | 14.3 | 17.1 |
| **mean** | **51.6** | **19.27** | **65.02** | **21.24** | **14.40** | **16.78** |
| **range** | 2.4 | 1.27 | 2.71 | 0.70 | 0.20 | 0.80 |

→ **variance 매우 작음 (p99 ±0.6ms)**. 측정 신뢰 가능. 진짜 baseline 확정.

```
오늘 baseline (NoMachine OFF, lock, 5회 평균):
  zed_lag       21.2 ms (architecture floor)
  bridge_proc   14.4 ms (별도 thread, 매우 안정)
  queue_wait     7.0 ms (pipeline-bridge 속도차)
  pipeline_proc 16.8 ms (= e2e GPU only p50)
  ──────────────
  true_e2e p50  57.0 ms
  true_e2e p99  65.0 ms

★ e2e (GPU only) p99 = 19.27 ms < HARD 20ms
★ HARD (e2e basis) 위반 = 0.22% — 우리 코드만 보면 거의 환자 실험 가능
```

---

## 4. Codex 라운드 1 검증 (frame overlap minimal patch)

`/tmp/codex_overlap_q_v2.txt` (171 line, 8 깊은 질문) → 응답 715k tokens, xhigh.

### 차단급 발견 (4개)

1. **HARD 20ms 도달 *물리적 불가능*** — `time.time_ns() - tick.ts_ns` gate. zed_lag 22 만으로 초과. true_e2e 100% 위반이 *정상*.
2. **`true_e2e = e2e + zed_lag` 잘못** — 실제 분해 `zed_lag + bridge + queue + pipeline`. clean 시 inf 전 floor 36.5ms.
3. **HARD 25ms 도 부족** — zed_lag 22 + bridge 14.5 = 36.5ms. **환자 실험 gate 50-70ms** 봐야.
4. **사용자 제안 latest-only 이미 구현됨** — `latest()` 가 deque[-1] (newest). 진짜 작업 = in-flight buffer lifetime.

### 구현 가정 문제 (5개)

5. **Preproc 도 single buffer** — `gpu_preprocess.py:47`. 다음 preproc 이 이전 inference 입력 덮음.
6. **TRT context/binding 1개 구조** — graph 2개 ping-pong 시 *동일 context 에 다른 address* capture 안전 보장 없음.
7. **`get_output()` 단일 tensor** — output slot API, lifetime ownership 없음.
8. **`PoseResult.kpt_conf` raw TRT output view** (`gpu_postprocess.py:248`) — publish D2H 전에 output slot 재사용되면 *조용히 깨짐*.
9. **Postprocess background 어려움** — `.cpu()` host sync + sticky/EMA state 갱신.

### 옵션 비교 (4 → 5, Codex 신규 추천)

| 옵션 | 평가 |
|---|---|
| A. Graph 2개 ping-pong | "minimal patch 아님" — graph 2개 + slot API + context 2개 + per-token events + scheduler + publish contract |
| B. Eager fallback | 진단용. graph 버리면 inf 이득 사라짐 |
| C. Output single + 즉시 D2H | host sync — 이득 죽임 |
| D. Post 만 background | TRT race 안 해결 + post host-sync |
| **E. D2D snapshot ring (★ Codex 추천)** | inf 직후 raw_output → snapshot[i] D2D copy. post 가 snapshot 읽음. **TRT graph 1개 + context 1개 유지**. A 의 절반 작업량. |

### 단계 누락 (5개)

10. **PipelineTick contract 먼저 변경** — frame metadata, pickup_ns, ready_ns, trace, valid gate 가 frame-token 에 묶여야. `pipeline._pending` 선언만 있고 미사용.
11. **stream `done_event` bundle 당 1개** — multi-inflight 시 re-record/wait 순서 버그. **per-token Event 필요**.
12. **R6 Phase 1 spec 미제시** — "사용자 제안과 동일" 단정 근거 없음.
13. **baseline 5회만으로 nvargus 가설 못 잡음** — `_cleanup_stale_resources()` 가 nvargus restart 안 함. 분리 측정 필요.
14. **Pass 조건에 correctness 누락** — frame_id 단조성, result/source ts 일치, valid=True over-budget 금지, graph fallback 없음, actual_publish p99.

---

## 5. Plan v8 outline (Phase 0-6, 17-22일)

사용자 결정: **정확하게 끝까지 + 모든 lever 시도 후 fusion 결정**.

### Phase 0 — 측정 인프라 (✓ 완료)
- 5회 baseline variance (Codex #21 부분 수렴)
- NoMachine OFF protocol 확정
- Plan v7 Round 0-3 모두 측정 (효과 정리)

### Phase 1 — Correctness pre-work (2-3일, **5/9 토 시작**)
- **PipelineTick contract refactor** — frame metadata 묶기 (Codex #10)
- **per-token event** — StreamBundle 변경 (Codex #11)
- **pre-input ring** — GpuPreprocessor.out single → ring (Codex #5)
- **Pass 조건 검증 도구** — frame_id 단조성, ts 일치 (Codex #14)
- 회귀 0 검증 (true_e2e p99 65ms 유지)

### Phase 2 — ZED CUDA interop (5-7일, *다음 주*)
- ZED MEM::GPU + DLPack
- bridge_proc 14.4 → 8-10ms (-5)
- frame overlap prerequisite (GPU contention 감소)

### Phase 3 — Output snapshot ring (1-2일, **5/10 일**)
- D2D copy: `inf.stream` 에서 raw_output → snapshot[i]
- post 가 snapshot 읽기 (raw_output race 무관)
- TRT graph 1개 + context 1개 유지

### Phase 4 — Frame overlap 옵션 E (3-5일, **5/10 일 시작 + 다음 주**)
- pipeline cycle 재구조: inf 끝 → background post + 다음 cycle
- per-token event + PipelineTick token 사용
- queue_wait 0~2 + pipeline_proc 8-10
- true_e2e p99 65 → **40-46ms** 추정

### Phase 5 — Post fusion (3일, *다음 주*)
- unlet+lift+EMA 단일 커널
- post 가 critical path 에서 빠진 후 추가 절감
- true_e2e p99 추가 -3ms → **37-43ms**

### Phase 6 — 종합 측정 + 결정 (1-2일, *다음 주*)
- 모든 lever 합산 후 true_e2e p99
- 50ms 도달 시 → fusion 미룸 (사용자 결정)
- 50ms 미달 시 → fusion 검토

### 일요일 (5/10) 종료 milestone
- Phase 1 완료
- Phase 3 완료
- Phase 4 시작

### 다음 주말 (5/17-19) 종료 milestone
- 모든 Phase 완료
- true_e2e p99 **35-40ms** 도달 목표
- 환자 실험 gate 50ms 충분 도달

---

## 6. Codex 라운드 2 spec (line 단위 implementation)

`/tmp/codex_overlap_q_round2.txt` (152 line, 8 질문) — 던질 준비 완료.

### 핵심 질문
- Q1: PipelineTick contract refactor schema (`_pending` dict 정확한 fields)
- Q2: per-token event API (StreamBundle 변경 또는 PipelineToken 안에 Event 보관?)
- Q3: pre-input ring 의 graph capture 호환성
- Q4: D2D snapshot ring 의 stream/event 정확한 위치
- Q5: PoseResult lifetime contract (어느 field 가 raw view?)
- Q6: Phase 1 day-by-day 단계 분해 + Pass 조건
- Q7: Phase 4 cycle 의 stream/event pseudocode
- Q8: 시스템 회복성 (1시간 degradation) + nvargus 가설

→ 응답 후 Phase 1 *첫 코드 변경 위치* (file + line range) 결정.

---

## 7. 환자 실험 gate 재정의 (사용자 결정 보류)

Codex R1 결론 — HARD 20ms / 25ms 모두 *물리적 불가능*. 사용자 결정 보류 항목:

| Gate | 도달 가능성 | 작업 |
|---|---|---|
| 20ms | ✗ 물리적 불가능 | — |
| 25ms | ✗ floor 36.5ms | — |
| 30ms | △ frame overlap + interop + post fusion + IMU fusion | Phase 1-6 + IMU fusion |
| **50ms** | ✓ Phase 1-5 만으로 도달 (35-40ms 실현) | 가장 현실적 |
| 70ms | ✓ 현재 baseline 65ms 안에 들어감 (frame overlap 안 해도) | gait control gain 큰 변경 |

**사용자 의사**: "정확하게 끝까지 해보고 모든 거 다 해보고 fusion 어떻게 할지 고민". 즉 **Phase 1-6 모두 진행 후 결정**.

---

## 8. 상태 점검 — 일정

```
2026-05-08 (금, 오늘) 14:30 ~ : Phase 0 정리 (이 문서)
                       16:00 ~ : Codex 라운드 2 던지기 + plan v8 update
                       18:00 ~ : Phase 1 spec 확정 (Codex 응답 기반)
                       
2026-05-09 (토, 14h focus) : Phase 1 본격 (PipelineTick + per-token event + pre-input ring)
2026-05-10 (일, 14h focus) : Phase 1 완료 + Phase 3 (output snapshot ring) + Phase 4 시작

다음 주 (5/11~) : Phase 2 (ZED interop) + Phase 4 완료 + Phase 5
다음 주말 (5/17-19) : Phase 6 종합 측정 + 환자 실험 line 도달 확인
```

---

## 9. 결정적 인용 (Codex 라운드 1)

> "현재 가장 likely root cause는 TRT engine 자체가 느린 게 아니라 full pipeline에서
> 사용자가 기대한 '단순 frame overlap' 이 실제로는 PipelineTick contract refactor +
> per-token event + pre-input ring + output snapshot ring + PoseResult lifetime
> 모두 변경. 'minimal patch' 아님."

> "옵션 E (D2D snapshot ring) 가 A (graph 2개) 보다 simple + safe. TRT graph 1개
> + context 1개 유지. inf 직후 raw_output → snapshot[i] D2D copy. post 가
> snapshot 읽음."

> "HARD 20ms 도달 *원천 불가능*. 환자 실험 gate 50-70ms 봐야."

---

## 10. 다음 행동 (3가지, 순서대로)

1. **Codex 라운드 2 던지기** (지금) — `/tmp/codex_overlap_q_round2.txt`
2. **plan v7 → plan v8 rename + update** (15분) — Phase 분배 + 일정 명시
3. **commit + push** — 이 checkpoint + plan v8 + vault skiro-learnings 11 lessons
