---
date: 2026-05-03
project: realtime-vision-control
topic: Track B latency debugging — e2e 측정 오류 발견 + set_stream 회귀
result: partial
---

## 목표

Track B CUDA_Stream 파이프라인의 `inf=7-15ms` / `getdata_rgb=9-14ms` 성능 저하 원인 파악
및 원래 벤치마크 (p99=19.8ms, ~79Hz) 수준 복원.

---

## 실험 결과 타임라인

| 시각(UTC) | 커밋 / 시도 | FPS | p50(e2e) | SOFT WARN | 비고 |
|---|---|---|---|---|---|
| 08:57 | 29d39a5 — per-stage timing 추가 | ~73Hz | 10.19ms | 0.81% | HARD LIMIT inf로 비활성화 |
| 09:03 | fa859be — tracer nan 수정 | ~76Hz | 11.59ms | 0.21% | |
| 09:07 | d1c325f — CUDA graph inf.stream 수정 | **~79Hz** | **10.31ms** | **0.49%** | 오늘 최고 수치 |
| 09:16 | **2c5ae6c — set_stream 교체** | ~55Hz | 19.53ms | **64.61%** | ← 여기서 회귀 |
| 10:24 | 8b90f9d — ring buffer + decimation=2 | 46.7Hz | 19.15ms | 60.10% | |
| 10:29 | decimation=2만 적용 결과 | 38.8Hz | 19.81ms | 77.19% | 오히려 악화 |
| 10:33 | 81db3f0 — depth-before-RGB (old order) | 38.8Hz | 22.86ms | **92.41%** | 최악 |
| 10:40 | 81db3f0 — depth-before-RGB (new order) | **46.3Hz** | **18.05ms** | **50.50%** | 복구 시도 최선 |
| 10:51 | cc87940 — max-capture-fps 75 | 45.3Hz | 18.55ms | 56.01% | 악화 → 즉시 revert |

---

## 회귀 원인 분석

### 2c5ae6c (set_stream 교체) 가 회귀를 일으킨 메커니즘

**원래 코드:**
```python
with torch.cuda.stream(inf.stream):
    self.runner.infer_async(...)
```
`torch.cuda.stream()` context manager는 종료 시 `stream_0.wait_stream(inf.stream)` 을 주입한다.
이로 인해 ZED SDK가 사용하는 stream_0 (PERFORMANCE depth 계산)이
inference가 끝날 때까지 자동으로 대기했다.

→ 부수 효과: **ZED depth와 TRT inference가 절대 동시에 돌지 않음** → SM 경합 없음.

**수정 후 코드 (set_stream):**
```python
torch.cuda.set_stream(inf.stream)
self.runner.infer_async(...)
torch.cuda.set_stream(_prev_stream)
```
stream_0에 대한 wait 없음 → ZED PERFORMANCE depth가 TRT inference와 동시 실행
→ Orin NX 1024 SM 경합 → inf 4ms → 7-15ms.

**결론:** context manager의 `stream_0.wait_stream(inf.stream)` 이 "의도치 않은 SM 직렬화"로
오히려 성능을 보호하고 있었다.

---

## 오늘 발견한 측정 오류 (중요)

### 1. e2e 측정 범위 오류

| | 기존 | 수정 |
|---|---|---|
| `t_start` | `bridge.latest()` 반환 직후 | 동일 |
| `t_end` | constraint 계산 **이후** | `po.stream.synchronize()` **직후** |
| `constraint_ms` | e2e에 포함 (2-5ms 부풀림) | 별도 키로 분리 |

### 2. HARD LIMIT이 잘못된 지표를 보고 있었음

```python
# 기존 (잘못됨)
frame_exceeds_budget = e2e_ms > LATENCY_HARD_LIMIT_MS
# e2e_ms = GPU 처리만 (~10-14ms). capture 시간 제외.

# 수정
frame_exceeds_budget = true_e2e_ms > LATENCY_HARD_LIMIT_MS
# true_e2e_ms = frame.ts_ns → GPU 완료. capture 포함.
```

### 3. 기존 "0.031% 위반율" 수치의 의미 재해석

기존 벤치마크 (이전 세션)의 "HARD LIMIT 위반 0.031%"는 `e2e_ms > 20ms` 기준.
`e2e_ms`는 GPU-only (~10-14ms)이므로 **capture 시간이 빠진 수치**.

`true_e2e_ms > 20ms` 기준의 실제 위반율은 **아직 미측정** — 다음 세션에서 측정 필요.

### 4. Latency vs Throughput 개념 정리

| 개념 | 값 | 의미 |
|---|---|---|
| Throughput | ~70-79Hz | 초당 몇 프레임 publish |
| true_e2e latency | ~18-22ms | 각 데이터가 얼마나 오래됐나 |

`true_e2e_ms ≈ buffer_wait(~7ms) + GPU_pipeline(~14ms)`

buffer_wait이 존재하는 이유: ZED 120fps, pipeline ~70Hz → 프레임이 버퍼에서 평균 7ms 대기.

---

## 오늘 세션 종료 상태

### 복원된 코드 (d1c325f 기준)

- `pipeline.py`: d1c325f 상태 + t_end/constraint_ms 측정 수정
- `gpu_preprocess.py`: d1c325f 상태 (torch.cuda.stream() context 복원)
- `gpu_postprocess.py`: d1c325f 상태 (torch.cuda.stream() context 복원)
- `zed_gpu_bridge.py`: d1c325f 상태 (ring buffer / decimation / depth-before-RGB 모두 제거)
- `run_stream_demo.py`: d1c325f 상태 + HARD_LIMIT 20ms 재활성화 + true_e2e_ms 기반 체크

### 유지된 수정

| 커밋 | 내용 | 이유 |
|---|---|---|
| 6bd6280 | verify_world_frame + launch_clean 개선 | 정확성 버그 수정 |
| 29d39a5 | per-stage timing 추가 | 디버깅 가시성 |
| fa859be | tracer nan 수정 | 버그 수정 |
| d1c325f | CUDA graph inf.stream 수정 | 진짜 버그 수정 (correctness) |

---

## 다음 세션 — Latency 개선 계획

### 현재 true_e2e_ms 구성 (추정)

```
buffer_wait:  ~7ms   (ZED 120fps vs pipeline 70Hz → 프레임 대기)
H2D:          ~2ms   (bridge 내부, t_start 이전)
pre:          ~1ms
inf:          ~4ms   (CUDA graph)
post:         ~2ms
constraint:   ~2ms
-----------------------
true_e2e_ms: ~18ms  (이상적 케이스)
             ~22ms  (평균)
```

### 개선 방향 (우선순위 순)

#### Phase 1 — 정확한 baseline 측정 (Jetson에서 즉시)
- `git pull` 후 `sudo launch_clean.sh 300` 실행
- `true_e2e_ms` 분포 확인 (p50, p99, 20ms 초과율)
- 이게 실제 시스템 상태의 첫 정직한 측정값

#### Phase 2 — Constraint를 C++로 이식 (~1.5ms 절약)
```
현재: Python tensor .item() 호출 → D2H sync → ~2ms
목표: C++ pybind11로 이식 → ~0.5ms
효과: GPU pipeline total ~16ms → 120Hz 근접 → buffer_wait 감소
```

#### Phase 3 — 120Hz 달성 시 buffer_wait → 0
```
현재: pipeline 70Hz → buffer_wait ~7ms
목표: pipeline ≥ 120Hz → buffer_wait ~0ms (ZED 속도와 동기화)
조건: GPU pipeline total < 8.3ms (1/120fps)
현재 gap: 16ms(Phase2 후) - 8.3ms = 7.7ms 더 필요
```

#### Phase 4 — TRT INT8 quantization (~2ms 절약)
```
현재: FP16 inference ~4ms
목표: INT8 ~2ms
리스크: keypoint 정확도 하락 → 재검증 필요
```

#### 달성 가능한 목표
- Phase 1+2: true_e2e_ms p99 ≈ 18ms (현재 ~22ms에서 개선)
- Phase 1+2+3: true_e2e_ms p99 ≈ 12-14ms (buffer_wait 제거)
- Phase 1+2+3+4: true_e2e_ms p99 ≈ 10-12ms

---

## 오픈 질문

1. `true_e2e_ms > 20ms` 기준 실제 위반율은? (첫 측정 필요)
2. stream context manager의 "accidental serialization"을 명시적으로 활용할 방법?
3. Constraint를 C++로 이식 후 pipeline Hz가 실제로 120Hz에 도달하는가?
