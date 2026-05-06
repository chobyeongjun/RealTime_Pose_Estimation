# 2026-05-06 Baseline 5 runs 측정 (현재 main)

commit: `c2ac670` (system-state RT kernel 적용 안 함 결정 후)

## 결과

| Run | timestamp | Hz | true_e2e p50 | true_e2e p99 | actual_publish p99 | bridge_proc p99 | pipeline_proc p99 | HARD 위반 |
|---|---|---|---|---|---|---|---|---|
| 1 | 11:08 | 37.6 | 66.1 ms | 80.2 ms | 81.9 ms | **21.2** ⚠ | 25.3 | 100% |
| 2 | 11:10 | 37.5 | 66.6 ms | 80.9 ms | 82.7 ms | 14.2 | 25.5 | 100% |
| 3 | 11:11 | 37.9 | 65.5 ms | 79.6 ms | 81.4 ms | 14.8 | 25.4 | 100% |
| 4 | 11:13 | 39.1 | 64.5 ms | 75.8 ms | 77.7 ms | 18.6 | 24.7 | 100% |
| 5 | 11:15 | 39.0 | 64.0 ms | 76.8 ms | 78.7 ms | 14.4 | 24.9 | 100% |

Run 1 bridge_proc p99=21.2 다른 runs (14-19)보다 큼 — *warmup 영향* 추정 (첫 run, ZED daemon 정렬 안 됨).

**Mean / spread**:
- Hz mean = 38.4, spread (max-min)/mean = 4.2% ✓ Inv2 PASS
- true_e2e p99 mean = 78.3 ms, spread 6.5% ✓ Inv2 PASS

## ★ 결정적 발견

1. **Hz = 37-39** (이전 메모 docs/evolution `67Hz` 와 차이 큼 — 회귀 시점 의문)
2. **HARD 위반율 100%** — 모든 frame이 valid=False publish. C++ 측 *모든 frame skip*. 사실상 control loop watchdog fallback 의존
3. **stage 우세도**:
   - zed_lag ~33ms p50 (★ dominant — SDK 큐 적체 또는 bridge cycle 26ms > ZED 주기 8.3ms)
   - pipeline_proc ~21ms p50 (post .cpu() block + inf 9-16ms 변동)
   - bridge_proc ~11ms p50 (host copy 경로)
   - queue_wait ~0.5ms p50 (정상)

## 측정 안정성 (Inv2 검증)

```
Inv1 decomp 등식:  decomp sum (~32ms) < true_e2e (~65ms)  ← Gap ~33ms = zed_lag (정상, decomp에서 빠짐)
Inv2 안정성:       Hz spread 4.2%, p99 spread 6.5%, 모두 ≤30% ✓
Inv3 (skip=0):     Phase A consume-once 동작 ✓
Inv4 frame_skip > 0:  미적용 — 사실상 한 번도 skip 안 함 (모든 새 frame 처리)
```

## raw [SLOW] 발견

- zed_lag 분포: 25-46ms (range 21ms — 큰 변동)
- inf 분포: 9-16ms (★ inf 자체 변동 큼 — SM 경합 의심)
- post 분포: 5-12ms (★ Codex R7-d post `.cpu()` block 영향)
- 가끔 `bridge_proc=23-25ms, ret_rgb=14-16ms` 거대 frame — ZED retrieve 자체 spike

## 행동 결정 (이 baseline 위에)

| Lever | 효과 | 작업 시간 |
|---|---|---|
| P1: SHM publish_done_ns + valid_reason | 측정 정의 anchoring | 1.5h |
| L1: post .cpu() 제거 + packed D2H | pipeline_proc -3-5ms | 3-4h |
| L2: ZED interop DLPack | bridge_proc + zed_lag 감소 | ~1주 |

→ **P1 → L1 → L2** 순서.  
→ diagnose.py 도구는 baseline [STATS]가 이미 충분 → 미루기.

## 의문 사항 (추후 분석)

- `why-it-got-faster.md` 의 67Hz가 어느 commit 시점인가? Phase A 후 어떤 변경이 38Hz로 회귀시킴?
- `inf` 9-16ms 변동의 진짜 원인 — TRT graph capture 동작 중인가? CUDA Graph 안 잡힌 frame 가능성
- ZED retrieve_rgb 가끔 14-16ms spike — SDK 내부 jitter

## raw log files (Jetson)

- `/tmp/baseline_run{1..5}.log`
