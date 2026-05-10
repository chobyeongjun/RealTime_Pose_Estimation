# Master Plan 2026-05 — 6-Week Clinical Path

**작성**: 2026-05-11.
**Codex orchestration consult**: `bvfvkxo1m` (token 920059, high reasoning).
**사용자 ultrathink reset 결정**: vision repo mission = Plan D quality input source. latency 마지막 1ms 추구 trap 인지. quality + reliability + Plan D contract 우선.

---

## 0. Big Picture (Codex final synthesis)

> "4-6주 optimal path = latency chase 중단, current 60ms pipeline = production baseline lock, Week 0 SHM v2 + quality/timestamp/stress gates. **Vision repo 의 책임 = '빠른 pose' 가 아니라 'timestamp/covariance/validity 가 정직한 pose'**. Control repo 의 책임 = stale/uncertain vision 을 안전하게 무시하고 bounded prediction 만 쓰는 것. V4L2/sparse stereo = secondary upside, clinical dry-run 을 block 하면 버림."

**Vision repo (제) 의 mission**:
- ✗ "fast pose" (이미 architecture floor)
- ✓ **"honest pose" (정확 timestamp + covariance + validity)**

**Control repo (사용자) 의 mission**:
- Plan D EKF predictor (vision-only constraint)
- Stale/uncertain vision 의 safety 처리
- Bounded prediction with watchdog

---

## 1. Hardware (locked, 변경 X)

```
Jetson Orin NX 16GB (8x Cortex-A78AE @ 1.984GHz, 1024 CUDA core, LPDDR5 102.4GB/s)
ZED X Mini SVGA 960×600 @ 120fps
GMSL2 (MAX9296 deserializer)
Global Shutter ★ — quality lever (motion blur 절감, heel strike geometric coherence)
JetPack 6.2 R36.4.7 (2025-09)
ZED SDK 5.2.1 + zedlink-mono 1.4.0
PyTorch 2.10.0 + CUDA 12.6 + CuPy 14.0.1 + VPI 3.2.4
TRT FP16 YOLO26s-lower6 + custom CUDA kernels
```

→ 사용자 자부 — academic SOTA hardware. 변경 X.

---

## 2. Production Baseline (LOCKED)

```
Latency:        true_e2e p99 = 60.86ms (commit f551dba: --gpu-stream-priority all-high)
Throughput:     56.8 Hz
HARD limit 20ms (vision proc): 100% violation (vision 자체 architecture, C++ control 의 hard limit 별도)
SOFT WARN 18ms: ~13%
Architecture floor: zed_lag 22ms + bridge_p99 13.6ms + queue 7ms + pipeline 18ms

Production flags:
  --no-constraints --strict-correctness
  --zed-cuda-interop --post-async
  --gpu-stream-priority all-high   (default)
```

→ **이 baseline lock**. *latency 추가 추구 X*. quality + Plan D 우선.

---

## 3. ABANDONED levers (Codex 검증)

| Lever | 이유 | 출처 |
|---|---|---|
| A.2 Triton post fusion | -2~3ms 작음 + Triton 미설치 | env audit + Codex Q4 |
| A.4 CUDA graph 확장 | launch overhead 증거 X (Nsight 안 함) | Codex Q7 |
| B yolo26n FP16 | Jetson 만 -20% (T4 -3x publish) | Codex Q3 |
| C RTMPose-T | 2-3주 migration risk | Codex Q4 |
| CPU affinity | 0.7ms noise 영역 | 8-case ablation |
| One-frame-late depth | ZED thread-safety 미증명 + kill_test silent exit 5회 | Codex Q1 + 측정 |
| Dense ZED bypass | 8-12+주 budget X | Codex Q5 |
| Full libargus rewrite | 6-10+주 budget X | Codex Q1 |

→ **모두 abandon**. 시간 / token / focus 분산 회피.

---

## 4. 6-Week Master Plan

### Week 0 (이번 주, 3-4일) — SHM v2 + Quality Harness ★ Critical Path

**Vision repo (제) 작업**:

| 작업 | 출력 | 의무 |
|---|---|---|
| SHM v2 spec 문서 | `docs/lessons/shm_v2_packet_spec.md` | ✓ 이미 작성 |
| SHM v2 publisher implement | `src/perception/CUDA_Stream/shm_publisher.py` v2 | next |
| Quality dataset dump tool | `src/perception/CUDA_Stream/dump_quality_dataset.py` | next |
| Stress quality gate | `scripts/stress_quality_gate.py` (30/60/120 min) | next |
| Mocap RMSE eval | `scripts/eval_mocap_pose_rmse.py` | next |
| Timestamp probe | `scripts/timestamp_probe.py` (LED/GPIO, optional) | next |
| Schema test | `tests/test_plan_d_packet_schema.py` | next |
| Timestamp test | `tests/test_timestamp_monotonic.py` | next |
| Master plan 문서 | `docs/lessons/master_plan_2026_05.md` | ✓ 이게 이 file |

**사용자 (control repo) 작업** (병렬):
- SHM v2 reader skeleton (Codex Q5 schema 의 C++ struct)
- Plan D L1 (constant velocity) — SHM v2 input 활용

**Week 0 의 falsification gate** (Pass 시 Week 1 진입):
- ✓ SHM v2 publisher + reader cross-validation 통과
- ✓ Quality gate test 가 *현재 baseline 60.86ms* 의 RMSE 측정 가능
- ✓ Timestamp test 가 monotonic + frame-to-frame ~8.3ms 검증

### Week 1 — Quality Dataset + Plan D L2 (병렬)

**Vision (제)**:
- Mocap (Vicon/OptiTrack) 또는 markered leg + calibrated stereo fixture 의 dataset 수집
- 현재 pipeline 의 *clinical-quality gate* 통과 검증 (Codex Q2):
  - 2D RMSE ≤ 6 px
  - 3D RMSE hip/knee ≤ 15 mm, ankle ≤ 25 mm
  - per-kp valid rate ≥ 95%
- V4L2 formats 검증 (`v4l2-ctl --list-formats-ext /dev/video0 /dev/video1`)
  - NV12/YUYV 면 V4L2 path 2-3주 가능 (Codex Q3)
  - Bayer raw 면 4-8주 → abandon

**사용자 (control repo)**:
- Plan D L2 (constant acceleration)
- SHM v2 reader 통합

### Week 2-3 — V4L2 sparse (option) + Plan D L3 (병렬)

**Vision (option)**:
- V4L2 NV12 capture (libargus 또는 V4L2 standard)
- VPI 3.2.4 ISP debayer + rectify
- Custom CUDA sparse stereo kernel (Census 9x9, 6 pts, L/R consistency)
- 측정 vs current ZED SDK (target -7~10ms)
- **Quality gate 통과 시만 production**

**사용자 (control repo)**:
- Plan D L3 (phase-locked EKF):
  - State `x = [phi, omega, alpha]`
  - Cycle template `mu(phi)` (3+ strides, 128 bins, β=0.03-0.10)
  - Cross-correlation phase estimation
  - HS event prediction `t_HS = wrap(phi_HS - phi) / omega`
- Watchdog policy:
  - Per-kp depth confidence + L/R consistency + age + innovation residual
  - Stale > 2 frames → invalid
  - EKF innovation > 3σ → conservative impedance
  - Failure > 100ms → disable timing-critical assist
  - "Prediction-only max 100ms" hard stop

### Week 3-4 — Integration

- Vision (V4L2 또는 ZED SDK) + Plan D EKF 통합
- HS p95 error 측정 (Codex Q2 gate: ≤ 30ms)
- Phantom/missed HS rate (gate: ≤ 2% per 30 steps)
- Effective control latency 측정 (target <30ms with EKF predict)

### Week 4-5 — Stress + Falsification

- 60/120 min reliability (drift, crash, valid_mask integrity)
- Clinical-like stress:
  - Occlusion (walker frame)
  - Slow pathological gait
  - Freezing of gait
  - Asymmetric / hemiparetic
  - Spastic catch / sudden velocity spike
- Watchdog correctness 검증
- Fallback chain: L3 → L2 → L1 → hold ≤50ms → pretension safe

### Week 6 — Clinical Dry-Run + 환자 실험

- Patient dry-run with consent
- Event log 분석 (HS error histogram)
- Energy / RPE / VAS 측정 (clinical metric)
- 환자 실험 진행

---

## 5. Critical Path + Drop Order (Codex Q6)

**Critical path** (block 시 모든 후속 task block):
1. SHM v2 schema 결정 (Week 0 day 1-2)
2. Plan D L1 + L2 (사용자, Week 1)
3. Quality dataset + gate (Week 1)
4. Plan D L3 (Week 2-3)
5. Integration (Week 3-4)
6. Stress (Week 4-5)
7. Dry-run (Week 6)

**Drop order** (schedule slip 시 abandon 순서):
1. ★ V4L2 sparse stereo (clinical block 시 즉시 drop)
2. Mocap dataset (markered leg fixture 로 fallback)
3. 60+ min stress (30 min 으로 축소)
4. **NEVER drop**: SHM v2, watchdog, timestamp accuracy, valid-mask gates

---

## 6. Quality Gate (Codex Q2 — 시작 thresholds)

```
Vision (per session, post-warmup):
  2D RMSE per keypoint        ≤  6 px
  3D RMSE hip/knee            ≤ 15 mm
  3D RMSE ankle               ≤ 25 mm
  per-kp valid rate           ≥ 95 %
  
Plan D (clinical-relevant):
  HS p95 error                ≤ 30 ms
  phantom HS rate             ≤  2 % / 30 steps
  missed HS rate              ≤  2 % / 30 steps
  
System (reliability):
  30/60/120 min run           no crash, no estop
  valid_mask integrity        no flip from valid → invalid → valid (oscillation)
  timestamp drift             < 100 us / minute
  
Plan D timeout / fallback:
  L3 → L2 transition          < 100 ms after divergence detected
  L2 → L1                     < 50 ms
  Pretension safe             < 200 ms (C++ watchdog)
```

**Ground truth ranking**:
1. Mocap (Vicon/OptiTrack/Qualisys 등) — 1순위
2. Markered leg + calibrated stereo fixture
3. Synthetic — unit test only
**ZED SDK depth ≠ ground truth**, regression baseline 만.

---

## 7. Hidden Risks + Mitigation (Codex Q8)

| Risk | Mitigation |
|---|---|
| ZED self-calibration drift | per-session snapshot + disable self-calib (`zed_calib_load.py`) |
| GMSL2 timestamp = deserializer (not photon) | LED/GPIO probe 검증 + drift 측정 (`timestamp_probe.py`) |
| Patient gait variability (cycle assumption 깨짐) | Plan D fallback chain + watchdog (사용자) |
| Walker occlusion / harness shadow | per-kp valid_mask + EKF innovation gate |
| C++ control loop RT jitter | SCHED_FIFO 90 + cores 6-7 reserved (control repo 의무) |
| Hardware (Jetson thermal, ZED USB/GMSL) | thermal logging + reconnect policy |

---

## 8. Communication Protocol — Vision (제) ↔ Control (사용자)

| 시점 | Vision repo 의무 | Control repo 의무 |
|---|---|---|
| **Week 0 day 1** | SHM v2 spec commit + push (이 문서) | spec review + 의문 질문 |
| **Week 0 day 2-3** | SHM v2 publisher implement + tests | SHM v2 reader skeleton |
| **Week 0 day 4** | cross-validation script (publisher → reader) | reader test 통과 |
| **Week 1 day 1-2** | Quality dataset 수집 시작 | Plan D L1 + L2 |
| **Week 1 day 3** | quality gate test result paste (vision quality 통과 검증) | reader 통합 test |
| **Week 1 day 5** | weekly sync (skype/Discord/in-person) | weekly sync |
| **이후** | weekly sync + GitHub issue 의 status update | weekly sync + status |

---

## 9. References

- Codex orchestration consult `bvfvkxo1m` (2026-05-11, 920K tokens)
- 이전 Codex consults: phase A/B/C plan (1.5M), predictive + lever (340K), ZED bypass + EKF validity (80K)
- Stereolabs ZED X GMSL2 docs
- NVIDIA VPI 3.2.4 stereo disparity docs
- Stenum 2021 OpenPose gait video (heel strike MAE 20ms healthy, worst 60ms)
- Atalante 2025 (IMU on vest, NOT vision)
- Honda Walking Assist 2013 (hip motor sensors, NOT vision)
- Thatte EKF prosthesis, Kang real-time phase, Righetti/Ijspeert AFO

---

## 10. Status Tracking

| Week | Goal | Status | Falsification Gate |
|---|---|---|---|
| 0 | SHM v2 + Quality | 진행 중 (이 문서) | publisher+reader cross-validation 통과 |
| 1 | Dataset + Plan D L1+L2 | scheduled | quality gate (2D ≤6px, 3D hip/knee ≤15mm) 통과 |
| 2-3 | V4L2 (option) + Plan D L3 | scheduled | V4L2 quality gate 통과 (또는 abandon) |
| 3-4 | Integration | scheduled | HS p95 ≤30ms, phantom/missed ≤2% |
| 4-5 | Stress + Falsification | scheduled | 60min no crash, valid_mask 정직 |
| 6 | Clinical dry-run + 환자 | scheduled | event log 분석 통과 |

---

*Last updated: 2026-05-11 (Codex orchestration consult 후 reset)*
