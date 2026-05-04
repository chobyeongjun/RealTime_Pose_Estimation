---
date: 2026-05-04
project: realtime-vision-control
topic: Phase A 측정 결과 + zed_lag 발견 + Plan v5 로드맵
result: phase_a_partial_success
session: post-meeting (5/4 오후)
---

## TL;DR

- **Phase A 적용 완료** (commit `d87957c`): consume-once + actual_publish 측정 + safety fix.
- **Phase A 평가**: stale 재처리는 제거됨 (queue_wait dominance 0.4%). 그러나 true_e2e_ms p99 90+ms 그대로.
- **새로운 발견**: `zed_lag` (ZED→bridge 도달 전 묵는 시간) p99=43.4ms. 이게 진짜 큰 적.
- **다음 단계**: zed_lag와 bridge_proc 동시 공격. Phase B/C 진입.

---

## 오늘까지 한 작업 (5/3 ~ 5/4)

### 디버깅 인프라 구축 (Phase 0, commit `8fcfbcc`)
- `ZEDFrame`에 `bridge_start_ns`, `ready_ns` 추가 (epoch ns)
- `pipeline.run_overlapped_step`에서 `pickup_ns` 기록
- `latency_ms` 분해 4개 키:
  - `zed_lag_ms` = bridge_start_ns - ts_ns
  - `bridge_proc_ms` = ready_ns - bridge_start_ns
  - `queue_wait_ms` = pickup_ns - ready_ns
  - `pipeline_proc_ms` = t_gpu_done_ns - pickup_ns
  - 합 ≈ true_e2e_ms
- [STATS]/[SLOW] 출력 확장
- `scripts/analyze_run_log.py` 분석 도구 추가

### Phase A 구현 (commit `d87957c`)
- **A1**: `bridge.latest()` consume-once via `_last_returned_id`
  - 같은 frame_id를 두 번 반환 안 함
  - 새 frame 없으면 None → run_stream_demo가 publish skip
- **A2**: `actual_publish_e2e_ms` = `time.time_ns() - tick.ts_ns` 측정 (publisher.publish() 직후)
  - 이게 진짜 cam→shm 안전 지표
- **A3**: `frame_skip_count` 카운트 (None 반환 회수)
- **SAFETY FIX (Codex 발견)**: 기존에 `frame_exceeds_budget` 계산만 하고 publish에 반영 안 됨 → 위반 frame이 valid=True로 발행되던 버그 수정. 이제:
  ```python
  pre_publish_e2e_ms = (time.time_ns() - tick.ts_ns) / 1e6
  publish_valid = result.valid AND not in_warmup AND not pre_publish_exceeds_budget
  ```

### Codex 협업 흐름
- Codex 세션 ID: `019df0ff-4719-72c0-bf88-841535caafc6` (`.context/codex-session-id` 저장)
- 첫 분석 (xhigh): 7개 root cause + Fix Plan 제시. **bridge.latest() consume 안 함** + **_pending 미사용** 발견.
- Phase A 검증 (high): Phase A APPROVED. safety bug 1개 잡음 + 패치 제공 → 그대로 적용.

---

## Phase A 측정 결과 (Jetson 60초, post-reboot, 1인 피사체)

```
Decomposition (sum ≈ true_e2e):
  zed_lag         min= 14.8  p50= 23.5  p95= 38.3  p99= 43.4  max= 49.6 ms
  bridge_proc     min=  9.2  p50= 19.6  p95= 24.7  p99= 26.6  max= 30.0 ms
  queue_wait      min=  0.1  p50=  7.6  p95= 13.5  p99= 15.9  max= 27.9 ms
  pipeline_proc   min= 14.1  p50= 19.2  p95= 23.6  p99= 24.7  max= 30.9 ms

Bridge thread breakdown:
  grab            p50=  3.6  p99= 19.7
  ret_rgb         p50=  2.6  p99= 15.8
  getdata_rgb     p50=  5.1  p99= 19.0
  pinned_rgb      p50=  0.5  p99= 11.0
  ret_depth       p50=  2.1  p99= 13.9
  getdata_depth   p50=  0.9  p99=  1.7

Pipeline thread breakdown:
  pre             p50=  1.4  p99=  3.5
  inf             p50=  9.0  p99= 16.5  ← graph 잡혀도 SM 경합으로 4ms 못 됨
  post            p50=  7.1  p99= 12.0  ← .cpu() D2H sync 때문
  constraint      p50=  3.2  p99=  8.6  ← Python .item() 호출

Top 5 slow (true_e2e>20):
  frame  815  true_e2e=98ms = bridge 10 + queue 28 + pipeline 25 (zed_lag 35)
  frame  256  true_e2e=93ms (zed_lag 34)
  frame 1301  true_e2e=92ms (zed_lag 37)
  frame 1302  true_e2e=90ms (zed_lag 41)
  frame 1425  true_e2e=89ms (zed_lag 30)

Dominant (>20ms slow frames, 2250 total):
  bridge_proc:    1230 (54.7%)
  queue_wait:        9 ( 0.4%)  ← Phase A 효과
  pipeline_proc:  1011 (44.9%)
```

### Phase A 합격선 vs 실측

| 기준 | 목표 | 실측 | 결과 |
|---|---|---|---|
| queue_wait p99 | < 5ms | 15.9ms | ❌ 단 dominance 0.4% → 영향 미미 |
| true_e2e p99 | 30-40ms | ~90ms | ❌ |
| frame_id monotonic | yes | yes | ✅ |
| HARD LIMIT 위반 → valid=False | 적용됨 | 적용됨 | ✅ |

**부분 성공.** stale 재처리 제거 + 안전 버그 수정 = 정확도 보장. 하지만 절대 latency는 zed_lag 때문에 못 줄였음.

---

## zed_lag = 43ms p99의 정체

### 정의
```
zed_lag = bridge_start_ns - frame.ts_ns
        = (bridge thread가 grab() 후 처리 시작 시각) - (센서 노출 시각)
```

### 두 원인 분해

**원인 ① — ZED SDK 내부 파이프라인 (~10-15ms 고정)**
- 센서 readout → ISP (demosaic, gain, denoise) → GMSL 전송 → depth 계산 (PERFORMANCE GPU) → 큐 ready
- 못 줄임 (depth 필수, FPS 고정, depth_mode 변경 영구 기각)

**원인 ② — SDK 큐 적체 (~10-30ms 가변)**
- ZED 생산: 120fps = 8.3ms cycle
- Bridge 소비: 38fps = 26ms cycle (3.1배 느림)
- ZED 큐에 frame 쌓임. grab()이 묵은 것 반환.
- p50=23.5ms ≈ ZED 내부 latency 12 + 큐 1 frame
- p99=43.4ms ≈ ZED 내부 12 + 큐 4 frame

### 악순환
```
TRT inference SM 경합 → ZED depth GPU work도 느려짐
↓
bridge_proc 늘어남 → 큐 더 적체
↓
zed_lag 더 커짐
```

---

## 합의된 영구 제약 (CLAUDE.md에 박힘)

| 항목 | 값 | 변경 불가 이유 |
|---|---|---|
| Depth | 매 프레임 retrieve | 3D pose가 매 프레임 제어에 필수 |
| Depth mode | PERFORMANCE | NEURAL은 SM 경합 ×2.4 |
| imgsz | 640 | 사용자 거부 |
| Model precision | FP16 | INT8 영구 기각 (YOLO26s 호환성 + 정확도) |
| Camera | ZED X Mini SVGA@120fps | 하드웨어 |

### 영구 기각 (반복 금지)
- TRT INT8 quantization
- `torch.cuda.stream()` → `set_stream()` 교체 (SM 경합 노출)
- depth decimation (sync fence 누적)
- max-capture-fps throttle (rhythm 깨짐)
- 코드 원복으로 FPS 복구 (GPU 비결정성)

---

## 다음 단계 — Plan v5

### 핵심 통찰 (재정리)
- **true_e2e < 20ms 달성하려면 zed_lag 줄이는 게 1순위**
- zed_lag ② (큐 적체)를 줄이려면 → bridge가 ZED 속도 따라가야 함 → bridge_proc < 8.3ms
- 또는 ZED 큐를 매 cycle 드레인 → 큐 항상 1 frame
- 둘 다 결국 **bridge 빠르게** 만드는 작업

### Phase B (제안 순서, Codex 검증 필요)

**B0: ZED 큐 드레인 (가성비 최고 후보)**
- bridge `_grab_one()` 내부에서 grab() 여러 번 호출하여 큐 비움
- `grab()` non-blocking 모드 또는 timeout=0 활용
- 효과 추정: zed_lag 23ms → 12ms (ZED 내부 latency만 남음)
- 작업: 30분-1시간
- 리스크: ZED API에 비블로킹 grab 있는지 확인 필요

**B1: gpu_postprocess.py:266 `.cpu()` 제거**
- 3개 scalar (box_conf 등) GPU 유지
- post_ms 7→2ms 예상
- 영향 받는 코드: tracer.set_result_meta, constraint
- 작업: 1-2시간

**B2: constraint 비동기화 또는 C++ 이식**
- Python `.item()` 호출 제거
- 즉시 publish, constraint는 별도 thread에서 다음 frame 결정 영향
- 작업: 2-3시간 (간단 비동기) ~ 1일 (C++ pybind11)

**B3: RGB/depth ready_event 분리**
- `ZEDFrame.rgb_ready_event` + `depth_ready_event` 분리
- pre/inf는 RGB 만 wait, post는 둘 다 wait
- inf SM 경합 일부 완화 가능
- 작업: 2-3시간

### Phase C (구조 개선)

**C1: GPU/pinned ring buffer 사전 할당**
- `.to(device)` 매 frame 할당 제거
- GPU allocator jitter 감소
- 작업: 2-3시간

**C2: `_pending` 실제 사용 또는 제거**
- 현재 `_pending: Deque[dict] = deque(maxlen=3)` 선언만 되고 안 씀
- 실제 frame-level 파이프라이닝 구현 또는 정직하게 single-frame 명명
- 작업: 4-8시간 (실제 overlap 구현 시)

### Phase D (깊은 재설계)

**D1: 후처리 + constraint 통합 C++ kernel**
- pybind11로 단일 fused op
- 모든 D2H sync 제거
- 작업: 2-3일

**D2: ZED CUDA interop**
- `sl.Mat` GPU에서 직접 읽기 (현재 host memcpy 방식)
- getdata_rgb/depth 5ms 제거
- `init.sdk_cuda_ctx` 또는 cuda-python 통합
- 작업: 3-5일 (SDK 깊이 파야)

---

## 미해결 질문들

1. ZED SDK의 비블로킹 grab() API 존재 여부 — Codex 또는 ZED 문서로 확인 필요
2. ZED 내부 latency가 정확히 얼마인가 (10ms? 15ms?) — 큐 비웠을 때 zed_lag 측정으로 확인
3. inf p99=16ms 중 graph replay vs SM 경합 분리 측정 — nsys profiling 필요
4. true_e2e_ms와 actual_publish_ms 차이 (publish 오버헤드) — 다음 측정 시 확인

---

## Codex 협업 인계 메모

세션 ID: `019df0ff-4719-72c0-bf88-841535caafc6` (`.context/codex-session-id`)

**Codex가 잡은 것 (오늘까지):**
1. `bridge.latest()` consume 안 함 → 우리 fix
2. `_pending` 미사용 (4-stage overlapped 거짓말) → 미해결, Phase C2
3. RGB/depth 합쳐진 ready_event → 미해결, Phase B3
4. `.to(device)` 매 frame 할당 → 미해결, Phase C1
5. `gpu_postprocess.py:266` `.cpu()` D2H sync → 미해결, Phase B1
6. `constraints.py:145, 148, 216` `.item()` 호출 → 미해결, Phase B2
7. `true_e2e_ms`가 SHM publish 전 측정 → 우리 fix (actual_publish_e2e_ms)
8. `frame_exceeds_budget` 무시되던 safety bug → 우리 fix (Phase A 검증 단계)

**Codex가 못 잡은 것 (Phase A 측정 후 새로 발견):**
- zed_lag 자체. ZED SDK 큐 적체 + 내부 파이프라인. Phase B0 또는 D2 대상.

---

## 다음 세션 시작 시 할 것

1. 이 문서 읽기 (Phase A 결과 + zed_lag 분석 + Plan v5)
2. 사용자에게 Phase B 어디부터 시작할지 결정 요청
   - **추천: B0 (ZED 큐 드레인) 먼저** — 30분-1시간 작업, 효과 측정 빠름, 다른 fix와 독립
   - 효과 있으면 zed_lag 절반 절감 가능
   - 효과 없으면 즉시 B1/B2로 이동
3. Codex 세션 resume해서 B0 설계 검증
4. 구현 → push → Jetson 측정 → 결정

---

## 핵심 commit 목록

```
8fcfbcc  feat(diag): true_e2e_ms decomposition for latency root-cause analysis
d87957c  feat(safety+diag): Phase A — consume-once + true SHM latency + HARD_LIMIT fix
e9c1844  perf(metric): re-apply accurate measurement on d1c325f base (이전 세션)
abdf97b  restore(track-b): revert to d1c325f baseline + fix metrics (이전 세션)
2d83476  docs: TRT INT8 영구 기각 (이전 세션)
b554d47  docs: depth decimation 영구 기각 (이전 세션)
```

## 핵심 파일 (다음 작업 대상)

- `src/perception/CUDA_Stream/zed_gpu_bridge.py` — Phase B0 대상
- `src/perception/CUDA_Stream/gpu_postprocess.py` — Phase B1 대상 (`.cpu()` line 266)
- `src/perception/CUDA_Stream/constraints.py` — Phase B2 대상
- `src/perception/CUDA_Stream/pipeline.py` — Phase B3, C2 대상
- `scripts/analyze_run_log.py` — 측정 분석 도구

## 측정 명령

```bash
# Jetson에서:
git pull
sudo ~/realtime-vision-control/src/perception/CUDA_Stream/launch_clean.sh 60 2>&1 | tee phase_X.log
python3 ~/realtime-vision-control/scripts/analyze_run_log.py phase_X.log
```
