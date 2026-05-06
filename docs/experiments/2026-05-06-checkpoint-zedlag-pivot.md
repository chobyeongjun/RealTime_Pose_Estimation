# 2026-05-06 PM Checkpoint — R11 진단 + zed_lag pivot

오전: `2026-05-06-full-diagnosis.md` (Codex R6/R7/R10/R11, L1/L_post/L2a 시도, trtexec 측정)
오후: counter R11 적용 + 측정 → 진단 방향 *완전 전환*

---

## 한 줄 요약

inf 6ms overhead 가설은 *깨졌다*. inf 는 이미 trtexec floor 근처(7.5ms).
**진짜 bottleneck 은 `zed_lag 21ms` (ZED SDK 내부)** — 5시간 동안 한 번도 본 적 없는 곳.

---

## 확정 사실 (측정 수치)

### A. trtexec engine floor (Orin NX 16GB MAXN, GPU 918 MHz lock)

```
mean   6.95 ms
median 6.87 ms
p99    9.98 ms
CoV    6.25%
```

명령: `--useCudaGraph --noDataTransfers --useSpinWait --duration=30 --avgRuns=200`

GPU lock 검증: `MinFreq=MaxFreq=CurrentFreq=918000000`, `EMC=2133MHz`. 둘 다 max로 잠김.
trtexec CoV 6%는 *thermal/SM scheduling jitter*; clock floating 아님 (검증됨).

### B. R11 counter dump (commit `e2272db`)

```
[diag] inf_graph captured=True replay=990 eager=0 set_address=3 frames=1019
```

| 카운터 | 값 | 가설 검증 |
|---|---|---|
| `captured=True` | ✓ | graph 잡힘 |
| `replay=990 / frames=1019` | ✓ | warmup 30 빼면 모든 frame replay |
| `eager=0` | ✓ | silent fallback 없음 → **H1 배제** |
| `set_address=3` | ✓ | n_io(1 in + 2 out)만 1회씩, cache hit → **H2 배제** |

### C. 실측 inf — 자체적으로 13 → 7.5ms 변화

```
frame 1235: inf=7.0   1236: inf=7.6   1237: inf=9.6
frame 1239: inf=8.8   1240: inf=7.1   1241: inf=7.0   1242: inf=7.1
                                      평균 ≈ 7.5 ms
```

**문제**: 이전 측정의 `inf p50=13ms`가 어떤 조건이었는지 *모름*. 재현성 미검증.
가능한 차이: thermal cool/hot, background load, GC cycle, ZED depth retrieve 동시성.

### D. system 전체 변화 (이전 → 이번)

| 지표 | 이전 (Full base) | 이번 (post-R11) | 변화 |
|---|---|---|---|
| system Hz | 38 | **50.3** | +32% |
| pipeline_proc p50 | ~26ms | **17.0ms** | -35% |
| inf p50 | 13ms | **7.5ms** | -42% |
| bridge_proc p50 | 17.5ms | 14.5ms | -17% |
| true_e2e p99 | 79ms | 67.25ms | -15% |
| HARD LIMIT 위반 | 100% | **100%** | 그대로 |

→ 자연 향상 +12 Hz. 그러나 환자 실험 *여전히 불가* (HARD LIMIT 919/919 violation).

### E. 진짜 bottleneck — zed_lag

decomposition mean sum check:

```
bridge_proc  12.9 ms
queue_wait    6.1 ms
pipeline_proc 17.2 ms
─────────────────────
합           36.2 ms

vs true_e2e   57.8 ms
─────────────────────
gap         +21.6 ms  ← zed_lag (camera capture → grab() return 사이 ZED SDK 내부)
```

**`zed_lag 21ms`는 우리 코드 외부**. ZED SDK 내부 buffering/pipeline 지연.
4-5시간 lever 만지작거리는 동안 한 번도 진단 표적이 아니었음.

---

## 5시간 작업 lever 결과 정리

| Lever | 의도 | 결과 | 상태 |
|---|---|---|---|
| **A11** (idle pipeline) | bridge cycle 격리 | bridge 11→6.5ms, isolation 확인 | 진단 도구 (메인 X) |
| **A6** (mock pipeline) | proc cost 격리 | Hz 64, bridge 12ms | 진단 도구 |
| **L1** (BGRA→RGB flip 제거) | bridge -4.4ms | A11에서만 효과, Full에서 0 | 머지됨, 무영향 |
| **L_post** (post packing) | post host block 제거 | bridge 17.5→20.5ms (**negative**) | flag로 default OFF |
| **L2a** (event pool) | per-frame allocation 제거 | 모든 metric -5~6ms (**negative**) | revert (commit `e1001b1`) |
| **R11** (diag counters) | H1/H2/H3 격리 | H1/H2 배제, H3도 무관 (inf already floor) | 머지됨 (commit `e2272db`) |

**적중**: 자연 향상 +12 Hz (코드와 무관, system state 변화)
**진짜 발견**: `inf 13ms`는 *misleading number*였다. 현재 7.5ms = floor 도달.

---

## 우선순위 재정렬

이전 (오전 기준, 이제 폐기):
1. ~~inf overhead 6ms 잡기~~ → 폐기 (이미 floor)
2. ~~bind cache fix~~ → 폐기 (cache 정상)
3. ~~graph eager fallback fix~~ → 폐기 (graph healthy)

신규 (오후 기준):

| 우선순위 | Lever | 현재 → 가능 | 작업 estimate | 비고 |
|---|---|---|---|---|
| **1** | **zed_lag 분해** | 21ms → ?? | 2-3일 진단 | ZED SDK API: triple buffering, capture pipeline, DMA timing |
| **2** | **bridge_proc** | 14.5 → 8-10ms | 1주 | ZED CUDA interop (Codex R3에서 검토했던 path) |
| **3** | **post 통합** | 6.8 → 3-4ms | 3일 | unlet+lift+EMA 커널 fusion |
| **4** | **frame overlap** (Codex R6 P1) | Hz 50 → 80+ | 5일 | output ring buffer + post N // pre/inf N+1 |
| ~~**5**~~ | ~~모델 축소~~ | inf 7.5 → 4-5ms | 1-2주 | 영구기각 항목, 다른 lever 소진 후 재고려 |

---

## 미해결 의문

1. **inf 13ms 의 정체** — 이전 baseline 측정에서 정말 13ms가 나왔는가? 재현 시 7.5ms 일관 나오는지 검증 필요. (다음 launch_clean 한 번 더 돌려서 inf 7.5ms 재현 확인)

2. **zed_lag 21ms 의 분해** — 어디서 발생?
   - camera sensor → DMA 도착: ?? ms
   - DMA 도착 → ZED SDK depth pipeline 종료: ?? ms
   - depth pipeline 종료 → grab() return: ?? ms
   - 검증 protocol: ZED SVO recording timestamp + grab() ts 차이, ZED SDK API timestamp 다층 확인

3. **CoV 6.25%의 정체** — clock 잠겼는데 trtexec jitter. thermal? background? TRT scheduler?

4. **Hz 50 자연 향상의 정체** — 코드 변경 없이 38→50. 무엇이 변했나?
   - 가능: jetson_clocks reapply, GC 누적 해제, ZED daemon restart 효과
   - 검증: 5회 재현 (변동성 baseline task #3)

---

## 다음 라운드 (Codex 질문 후보)

이번 발견 기반으로 Codex 다음 질문:

```
trtexec 6.95ms = inf floor 도달 확인. counter는 graph healthy.
실측 inf 7.5ms (이전 13ms 재현 안 됨, 문제일 수 있음).

진짜 bottleneck: true_e2e 57.8 - decomposition_sum 36.2 = zed_lag 21.6ms
이건 ZED SDK 내부 — camera → grab() return 사이.

질문:
Q1. zed_lag 21ms 의 *물리적 분해* — sensor→ISP→depth→grab return 어디 어떻게?
Q2. ZED X Mini SVGA@120fps 의 진짜 frame interval 8.3ms 인데 zed_lag 21ms
   = 2.5 frame buffering. ZED SDK depth pipeline (PERFORMANCE)이
   triple-buffer로 잡혀있는가? 줄일 수 있는가?
Q3. 현재 bridge_proc은 grab() 후 시간 — 즉 zed_lag 다음 단계.
   bridge_proc(14.5) + zed_lag(21) = 36ms 가 ZED SDK 종속.
   ZED CUDA interop 으로 zero-copy 하면 zed_lag 자체도 줄어드는가?
Q4. Frame-level overlap 구현 시 effect — pipeline 17ms 줄어도
   zed_lag 21ms 이 critical path. true_e2e 의미 있는 절감 가능한가?
Q5. 현실적 다음 행동: zed_lag 진단 vs ZED CUDA interop vs frame overlap —
   가장 cost-effective?
```

---

## 재현 검증 (17:18 — 추가 측정)

```
pipeline_proc p50  17.2 ms  (16:23 측정 17.0 ms 재현 ✓)
bridge_proc   p50  14.5 ms  (동일 ✓)
zed_lag           21.8 ms   (16:23 측정 21.6 ms 재현 ✓)
true_e2e      p99 68.79 ms  (16:23 67.25 ms 재현 ✓)
e2e (GPU only) p99 21.98 ms ← HARD LIMIT 20ms 거의 임박
HARD LIMIT (GPU only) 위반 3.81%
```

→ inf 7.5ms / zed_lag 21ms / pipeline_proc 17ms 모두 *재현*. 진단 확정.
→ **결정적 함의**: 우리 코드만 보면 e2e p99 22ms (HARD LIMIT 거의 도달).
   `zed_lag 21ms 만 잡으면 환자 실험 가능 line`.

## Commit 로그 (오늘 PM)

- `e2272db` diag(R11): replay/eager/set_address counters for inf overhead
  → graph healthy 확인, H1/H2 배제

(추가 commit 없음 — 측정 결과로 진단 방향 전환만 결정)

---

## 메모: `inf 13ms 가설 폐기`의 의미

5시간 동안 우리는 *틀린 가설*에 매달렸다.
- "inf 13ms 가 trtexec 6.95ms 와 6ms gap 이다 → 우리 코드 overhead"
- 이걸 풀려고 L_post, L2a 만들고 둘 다 negative
- counter 추가 후 측정해 보니 *현재 inf 가 7.5ms* — gap 자체가 거의 없다

교훈 (다음 round 적용):
- *의심 가설* 측정 전에 lever 작업 금지. 가설은 측정으로 확정 후 작업.
- 단일 측정값 (예: inf 13ms p50) 을 *불변 사실*로 가정 금지. 시간 경과 후 재측정.
- counter 같은 *cheap diagnostic* 를 lever 작업보다 먼저.

이 패턴은 `~/research-vault/Research/10_Wiki/skiro-learnings.md` 후보.
