# Measurements Log — realtime-vision-control

모든 Jetson 측정 결과의 영구 기록. 각 entry = (date, commit, config, result, conclusion).

회귀 추적 + 의사 결정 근거 + 환자 실험 reference.

---

## 2026-05-10 18:24 — A.3 priority ablation (4-case) — commit `f0884e5..a29e799`

### Setup
- Jetson Orin NX 16GB + ZED X Mini SVGA@120fps + YOLO26s-lower6 TRT FP16
- jetson_clocks applied, SCHED_FIFO 90 (chrt)
- 60s per case, sleep 15s between

### Cases + Results

| case | priority | bridge env | true_e2e p99 | hz | bridge_p50 | bridge_p99 | pipeline_p50 | zed_lag |
|---|---|---|---|---|---|---|---|---|
| 1_off_nobr | OFF | OFF | 68.10 | 51.4 | 14.30 | 15.30 | 17.20 | 22.00 |
| 2_infer_br | infer-only | 6,7 RT 80 | 68.94 | 53.5 | 5.00 | 13.80 | 16.60 | **27.10** |
| **2b_all_br** ★ | **all-high** | 6,7 RT 80 | **61.20** | 54.5 | 12.70 | 13.60 | **15.90** | 22.70 |
| 3_infer_nobr | infer-only | OFF | 61.98 | 52.8 | 4.60 | 13.80 | 17.10 | 22.60 |

### Conclusions

1. **all-high 가 진짜 lever** — production best (61.20ms p99). Codex 가설 ("priority 효과 0") falsified.
2. **bridge_p50 5ms 는 misleading** — infer-only 모드의 *median 빠름, p99 동일* 비대칭. all-high 가 *전체 일관* (12.70 / 13.60).
3. **case 2 의 zed_lag 27.10 outlier** — bridge resource (cores 6,7) + C++ control 충돌 sneak preview 가능성. **CPU affinity 측정 필요**.
4. **production default = all-high** — commit `f551dba`.

### Action items
- [x] commit f551dba: --gpu-stream-priority default `all-high`
- [ ] CPU affinity ablation 측정 (다음)
- [ ] Validation 측정 (default = all-high 검증)

---

## 2026-05-08 — 12-case combination ablation — commit `4c28cba` 영역

### Setup
- 동일 hw/sw setup
- 8 lpost variants + 4 γ variants = 12 cases
- 각 case 25s

### Results (요약)

| case | flags | true_e2e p99 |
|---|---|---|
| 00_baseline | (no flags) | ~73 |
| 02_async_only | --post-async | ~70 |
| **08_interop_only** | --zed-cuda-interop | ~67 |
| 09_interop_overlap | γ + --frame-overlap | ~78 (REGRESSION) |
| **10_interop_async** ★ | γ + --post-async | **61.81** ← previous best |
| 11_all_lever_new | γ + --frame-overlap + --post-async + --lpost-ablation | ~75 (REGRESSION) |

### Conclusions

1. **case 10 (γ + --post-async) = previous best 61.81ms p99**.
2. **--frame-overlap 영구 deprecated** — case 09, 11 에서 +10-15ms 회귀 (commit `e782805`).
3. ZED CUDA interop (γ) -4.6ms p99 (vs case 02).

---

## 2026-05-10 17:33 — invalid 측정 (3 case 동시 복붙)

### Setup
- 3 cases 한 줄에 복붙 → 동시 실행
- ZED 카메라 충돌 가능성

### Result
- p99 = 65.21ms, hz = 51.6 (production best 보다 회귀)
- queue_wait p99 14.5ms (이전 ~7-8ms 보다 +7ms)

### Conclusion
**INVALID** — process 충돌. 측정 protocol 위반 (각 case 60s 후 다음 시작).

### Lesson
launch_clean.sh 60 은 *60s 동안 카메라 점유*. 한 줄 복붙 시 1번째만 정상, 나머지 충돌. **각 case 60s 끝까지 기다린 후 다음**.

---

## CPU affinity Notes (2026-05-10)

### 현재 conflict
- CLAUDE.md: C++ cores 6-7
- commit 69f918d: BRIDGE_CORES="6,7"
- → **같은 cores**. C++ 실행 시 bridge starvation risk.

### Architecture
- Jetson Orin NX = 8x ARM Cortex-A78AE (homogeneous)
- 모든 cores 동일 (P/E core 구분 없음)
- L3 2MB shared
- *core 그룹 = NUMA/cache locality + RT priority 충돌 회피* 의미만

### 측정 예정 — 6 cases
- A) no_env
- B) BRIDGE 6,7 RT 80 (현재)
- C) BRIDGE 4,5 RT 80 (분리, 정답 후보)
- D) BRIDGE 4 RT 80 (single, deterministic)
- E) BRIDGE 0,1 RT 80 (system 충돌 risk)
- F) BRIDGE 6,7 RT 99 (C++ 보다 high)

### 검증 가설
- C 또는 D 가 best (C++ 와 분리)
- B 와 F 는 C++ 실행 시 latency 회귀
- E 는 system service 와 충돌

---

## Templates

### 새 측정 entry format

```markdown
## YYYY-MM-DD HH:MM — <name> — commit `<short-sha>`

### Setup
- ...

### Cases + Results
| case | config | metric_1 | metric_2 |
|---|---|---|---|

### Conclusions
1. ...

### Action items
- [ ] ...
```

---

## References (Codex consults)

| Date | Topic | Tokens | Key finding |
|---|---|---|---|
| 2026-05-10 | Phase A/B/C plan review | 1.5M | Plan D EKF predictor = main path |
| 2026-05-10 | Predictive + 추가 lever | 0.34M | A.3 priority "이미 active, 효과 0" — falsified by 측정 |
| 2026-05-10 | ZED bypass + zero-copy + EKF validity | 0.08M | Full bypass abandon, RawBuffer = 2-3주, vision-only EKF = research path |

---

## TODO — 다음 측정

1. **CPU affinity ablation** (이번 turn 의 script) — 6 case
2. **Validation 측정** (production default = all-high 1 case, statistical 검증)
3. **bridge resource 효과 재측정** (case 2 의 zed_lag 27 outlier 재현 여부)
4. (Week 2-3) **ZED RawBuffer + sparse stereo prototype** 측정
5. (Week 2-3, 사용자) **C++ EKF predictor** 의 phase residual 검증

---

*Last updated: 2026-05-10*
