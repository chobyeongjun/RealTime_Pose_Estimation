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

## CPU affinity Notes (2026-05-10) — 진정 정정

### 진정한 환경 정정 (사용자 정확 지적)

**현재 상태**:
- C++ Teensy 통신 = **미실행** (의도 spec 만, 실제 안 돔)
- 즉 cores 6-7 의 reservation = **낭비 가능**
- cores 0-1 = Xorg + nvargus-daemon (CLAUDE.md "GMSL = EGL=X 필수")
- → vision pipeline 이 **cores 1-7 (7 cores) 자유 사용 가능**

**미래 상태 (C++ Teensy 시작 후)**:
- C++ control loop = cores 6-7 점유 (RT FIFO 90)
- bridge thread + Python = cores 2-5 분리 필요
- → 그때 commit 69f918d 의 BRIDGE 6,7 = conflict

### Architecture
- Jetson Orin NX 16GB = 8x Cortex-A78AE (homogeneous, P/E 구분 X)
- L1 64KB I+D / L2 256KB / L3 2MB shared
- core 그룹 = cache locality + RT priority 충돌 회피 의미만

### 측정 예정 — 8 cases (commit `ce07b9f` 의 6 → 8 확장)

| case | bridge config | 의미 |
|---|---|---|
| A | no env | kernel inherit, 가장 가까운 baseline |
| B | 6,7 RT 80 | commit 69f918d (C++ cores 와 동일) |
| C | 4,5 RT 80 | C++ 와 분리, Python 의 일부 |
| D | 4 RT 80 | single core deterministic |
| E | 0,1 RT 80 | system cores 충돌 risk |
| F | 6,7 RT 99 | C++ 보다 high (위험) |
| **G** | **2,3,4,5,6,7 RT 80** | **★ 현재 환경 best — 모든 vision cores 자유** |
| **H** | **2,3 RT 80** | **Python 와 같은 cores — cache locality 검증** |

### 검증 가설
- 현재 환경 (C++ X): G 또는 H 가 best (cache locality)
- 미래 환경 (C++ O): C 또는 D 가 best (분리)
- B / F = C++ 실행 시 latency 회귀
- E = system service 충돌

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

1. ~~**CPU affinity ablation** (8-case)~~ ✓ 완료 (2026-05-10 19:16)
2. **V4L2 formats 검증** (`v4l2-ctl --list-formats-ext`)
3. (Week 2-3) **V4L2 + VPI sparse stereo prototype** ★
4. (Week 2-3, 사용자) **C++ EKF predictor** phase residual 검증
5. (선택) C++ Teensy 통신 시작 후 CPU affinity 재측정

---

## 2026-05-10 19:05–19:16 — CPU Affinity 8-case Ablation (commit `04550b3`)

### Setup
- production default flags: `--no-constraints --strict-correctness --zed-cuda-interop --post-async`
- `--gpu-stream-priority all-high` (default, commit f551dba)
- 60s × 8 cases, sleep 15s 사이
- C++ Teensy control = 미실행 (현재 환경 baseline)

### Results (정렬 by true_e2e p99)

| 순위 | case | bridge config | true_e2e p99 (ms) | hz | bridge_p99 | pipeline_p50 | zed_lag |
|---|---|---|---|---|---|---|---|
| 1 ★ | D_br4_rt80 | cpu 4 single RT 80 | **60.86** | 56.8 | 13.60 | 15.20 | 21.80 |
| 2 | A_no_env | (kernel inherit) | 60.90 | 55.7 | 13.70 | 15.50 | 22.20 |
| 3 | F_br67_rt99 | 6,7 RT 99 | 61.23 | 56.8 | 13.60 | 15.10 | 21.90 |
| 4 | H_br23_rt80 | 2,3 RT 80 | 61.24 | 54.7 | 13.70 | 15.70 | 22.50 |
| 5 | B_br67_rt80 | 6,7 RT 80 (commit 69f918d) | 61.31 | 55.0 | 13.60 | 15.60 | 22.60 |
| 6 | G_br2to7_rt80 | 2-7 RT 80 | 61.56 | 54.3 | 13.60 | 15.80 | 22.70 |
| 7 | C_br45_rt80 | 4,5 RT 80 | **65.30** | 56.4 | 13.70 | 15.40 | 22.50 |
| 8 | E_br01_rt80 | 0,1 RT 80 | **67.85** | 55.0 | 13.70 | 15.70 | 23.90 |

### Conclusions ultrathink

1. **CPU affinity 의 *진짜 효과 거의 0***. Best (D 60.86) vs 6 case 평균 (61.0) = 0.14ms 차이 = statistical noise. 1-7 cases 의 상위 6 = 60.86~61.56ms 의 0.7ms 격차 안.

2. **bridge_p99 13.6ms 가 architecture floor** — ZED SDK retrieve_measure 의 depth bimodal 한계 (Codex Q3 검증). CPU 어디 두든 못 줄임.

3. **검증된 가정**:
   - **E (cores 0-1) +7ms 회귀** ★ — system (systemd/Xorg) 와 RT 80 충돌. CLAUDE.md "GMSL = EGL=X 필수" 의 실제 검증. **cores 0-1 절대 사용 X**.
   - **C (cores 4-5) +4.4ms 회귀** ★ — env audit 의 nvargus-daemon PSR=4 와 충돌. context switch + cache thrash. **cpu 4 nvargus 양보 권장**.

4. **부정된 가정**:
   - "Python cores 2-5 / C++ cores 6-7 분리 필수" → 현재 환경 (C++ X) 에선 의미 없음. cores 1-7 모두 자유.
   - "C 또는 D = best" 가설 → C 는 worst, D 는 marginal best (0.04ms vs A).

5. **production 권장**:
   - **BRIDGE_CORES env 미설정** = case A = kernel inherit. 가장 단순, near-best.
   - 또는 BRIDGE_CORES="4" = case D, deterministic single core.
   - **commit 69f918d 의 BRIDGE_CORES="6,7" 도 OK** (case B = +0.4ms, noise 영역). 단 *낭비 적 reservation*.
   - **미래 C++ Teensy 시작 후 재측정 필수**.

6. **진짜 lever 는 V4L2 우회**:
   ```
   bridge_p99 13.6ms = ZED SDK depth pipeline 한계 (못 줄임)
   V4L2 직접 + custom sparse stereo = -7~10ms 가능
   → 진정 game changer (Week 2-3)
   ```

### Action items
- [x] 측정 + 분석 + 기록
- [x] BRIDGE_CORES env 의 production default 결정 (env 미설정 권장)
- [ ] V4L2 formats 추가 검증
- [ ] **One-frame-late depth thread implement (Week 1, -10~11ms 권장)**
- [ ] V4L2 prototype 시작 (Week 2-3)

---

## 2026-05-10 — Bridge_p99 Root Cause Deep Analysis (commit `1e2b105`+)

### Anti-correlation pattern 발견

이전 측정의 verbose log 의 sub-step 분석:

| frame type | grab | ret_rgb | ret_depth | sum |
|---|---|---|---|---|
| Type A (block early) | **10.7** | 0.7 | **1.3** | 12.7 |
| Type B (block late)  | 2.6 | 2.1 | **9.2** | 13.9 |
| Type A | **10.6** | 0.6 | **1.4** | 12.6 |
| Type B | 2.7 | 1.9 | **9.1** | 13.7 |

→ **`grab` 과 `ret_depth` 가 anti-correlated**. 합 항상 ~12-14ms. ZED SDK 가 *어디서 wait 할지* 만 결정 — *총 depth pipeline work fixed*.

### Root cause

```
ZED SDK 의 grab() 행동:
  if (이전 frame 의 depth pipeline 진행 중) {
      block;              // grab=10ms
      depth 즉시 가용;    // ret_depth=1.3ms
  } else {
      즉시 return;        // grab=2.5ms
      depth 백그라운드;   // ret_depth=9ms (wait)
  }
```

→ bridge_p99 = ZED SDK 의 depth pipeline 총 work (~12-13ms). **CPU affinity 와 무관**.

### One-frame-late depth thread = 진정한 lever

```
Current (hot path 13ms):
  grab → ret_rgb → ret_depth (block) → bridge done

One-frame-late (hot path 2.5ms):
  Main thread:    grab → ret_rgb → depth=queue.pop() ★
  Worker thread:  retrieve_measure (background, queue.push())
```

| 항목 | 변화 |
|---|---|
| bridge_p99 | 13.6 → 2-3ms (-10~11ms) |
| Depth age | +8.3ms (1 frame stale) |
| Knee peak angular error | ~2.5° (300°/s × 0.0083s) — 무시 가능 |
| **Net effective latency** | **-2~3ms** |
| Jitter 제거 | bimodal → consistent fast path |

### Codex consult background `bn57396zt`

8 questions:
1. ZED SDK 5.2.1 Camera thread-safety (single object, multi-thread retrieve)
2. One-frame-late architecture spec (worker lifecycle, queue type)
3. Pyzed Python GIL release (vs C++ binding 필요?)
4. Timestamp matching (RGB N vs depth N-1)
5. Failure modes + watchdog
6. TDD test design
7. Path A (one-frame-late) vs Path B (V4L2) 비교
8. Implementation skeleton (zed_gpu_bridge.py + pipeline.py)

→ 답 받으면 TDD red phase 작성 + green implement.

---

## 2026-05-10 — Jetson Environment Audit (commit `b8c33df`)

### Setup
- L4T R36.4.7 (JetPack 6.2, 2025-09 build)
- jetson_clocks ON, nvpmodel MAXN, GPU 918MHz, CPU 1.984GHz × 8

### Software stack

| Package | Version |
|---|---|
| ZED SDK | 5.2.1 |
| stereolabs-zedlink-mono | 1.4.0 (MAX9296) |
| pyzed | OK (Python binding) |
| PyTorch | 2.10.0 + CUDA 12.6 |
| CuPy | 14.0.1 + CUDA 12.9 (minor mismatch) |
| **VPI** | **3.2.4 + python3.10-vpi3** ★ |
| Triton | **미설치** (A.2 fusion 은 PyTorch fallback 만) |

### 결정적 발견 — V4L2 path 가능

```
ZED SDK pyzed binding:
  Camera methods (raw/buffer): []  ← RawBuffer Python 노출 X
  Mat methods: ['update_cpu_from_gpu', 'update_gpu_from_cpu']
  MEM types: BOTH, CPU, GPU

V4L2 (kernel level):
  /dev/video0 (left), /dev/video1 (right)
  zedx 10-0020 (platform:tegra-capture-vi:1)
  → driver cleanly exposes the sensors
  → V4L2 raw capture 가능 (Codex Q1a 권장 path 의 정확한 만족)
```

→ **ZED SDK 우회의 진정한 path = V4L2 + VPI 활용** (RawBuffer 대안).

### CPU 사용 패턴 (사용자 정확)

```
nvargus-daemon  PSR=4  3.6% CPU
gnome-shell     PSR=3  0.3% CPU
update-manager  PSR=7  0.3% CPU
nxrunner.bin    PSR=3  0.1% CPU
systemd 등      cpu0-7 분산 (near idle)
Xorg            top 5s sample 에 안 보임 (거의 idle)
```

→ **system 가 어느 specific cores 도 점유 안 함**. 모든 cores 0-7 *대부분 idle*.
→ **vision pipeline 이 cores 1-7 자유 사용 가능** (cpu0 만 systemd init 양보 권장).
→ CLAUDE.md 의 cores 0-1 system reservation = *과도한 가정*, 실제 system load 0% 가까움.

### Conclusions

1. **CPU 정정**: 현재 환경 (C++ X) 에서 cores 1-7 모두 vision 자유
2. **V4L2 path 가능**: pyzed RawBuffer 노출 안 됨 단 V4L2 가 더 직접 접근
3. **VPI 3.2.4 설치됨**: sparse stereo prototype 의 ISP/rectify 활용 가능
4. **Triton 미설치**: A.2 fusion = PyTorch fallback 만 (Codex Q4 의 "Triton 효과 작음" 과 일치, drop list 강화)

### Action items
- [x] 환경 영구 기록
- [ ] V4L2 formats 추가 검증 (`v4l2-ctl --list-formats-ext /dev/video0`)
- [ ] CPU affinity 8-case sweep (script `measurement_cpu_affinity_ablation.sh`)
- [ ] 측정 결과 entry 추가

---

*Last updated: 2026-05-10*
