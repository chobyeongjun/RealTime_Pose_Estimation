# Plan D — Phase-Locked EKF Predictor (C++ Control Loop)

**작성**: 2026-05-10. Codex consult Predictive + ZED bypass + EKF vision-only validation (총 ~2M tokens, high reasoning).
**위치**: 별도 C++ control repo (이 vision repo 가 아님).
**역할**: vision-only constraint 하 effective control latency <30ms 의 *유일한 path*.

## ⚠️ Codex brutal honest assessment (2026-05-10)

- **"Vision-only EKF = research path, NOT defensible clinical SOTA claim."**
- Published exoskeleton phase work = encoder/IMU/force/pressure. Vision-only closed-loop clinical = **no precedent**.
- "Effective 0ms" 는 marketing. 정직 표현: **"latency-compensated control with bounded prediction error"**.
- 3 bad frames = 25ms gap, 6 bad = 50ms (= heel strike transient 자체).
- → 사용자 vision-only 의지 = academic contribution + clinical risk 책임.

## Clinical success metric (Codex 강조)

**Raw latency 가 아님**. 진짜 metric:
- Heel strike p95 error
- Dropout behavior
- Watchdog correctness under slow pathological gait + walker occlusion + bad texture
- Predicted HS error histogram (mean + p99 + worst case)
- Missed/false trigger rate

---

## 1. Why this is the main deliverable

Codex brutal assessment:
- Vision sensor latency floor ~50ms (ZED ISP + processing). 4-6주 안 30ms 못 줄임.
- Atalante = IMU on vest, NOT vision. Honda WSAD = hip motor sensors, NOT vision.
- Stenum 2021 OpenPose: HS MAE 20ms 단 worst HS 60ms, TO 110ms (healthy 만, 환자 X).
- **Vision-only 의 SOTA = gait analysis. Closed-loop clinical assistance = research path.**
- 우리 vision-only 목표 = academic contribution 영역.

**유일한 30ms effective latency path** = vision 50ms + phase-locked predictor → 미래 50-70ms forward 예측.

---

## 2. EKF/PLL hybrid spec

### 2.1 State

```
x = [phi, omega, alpha]
- phi   : gait phase (rad, 0..2π)
- omega : cadence (rad/s)
- alpha : cadence acceleration (rad/s²)
```

### 2.2 Prediction (forward integration)

```
phi   += omega*dt + 0.5*alpha*dt²
omega += alpha*dt
alpha  = (수렴 가정) 또는 random walk
```

### 2.3 Measurement

**입력은 raw keypoints 아닌 joint angles**:
```
q = [hip_flex_L, knee_L, ankle_L, hip_flex_R, knee_R, ankle_R]
```

**Template**: `q ≈ μ(phi) + noise`

**Phase update via template matching**:
```
phi_meas = argmin_phi (q - μ(phi))^T Σ(phi)^-1 (q - μ(phi))
```

EKF update on phase residual:
```
e        = wrap(phi_meas - phi)        # angular wrap (-π..π)
phi     += Kp * e
omega   += Ki * e * dt
P_post   = (I - K*H) * P_pred
```

→ **EKF 권장** (PLL 보다) — covariance + rejection gates 가능.

### 2.4 Cycle template μ(phi)

| 항목 | 값 |
|---|---|
| 최소 stride | 3 strides 후 L3 활성 |
| Phase bins | 128 per cycle |
| Update rule | recursive `μ_new(phi) = (1-β)*μ_old(phi) + β*q_current` |
| β (clinical) | 0.03–0.10 |
| Effective memory | 3-5 strides |
| Left/right | 분리 유지 (asymmetric gait 대응) |

**Pitfall**: 1-stride memory 절대 X (pathology/noise chase).

### 2.5 Phase estimation method ranking (Codex)

| 방법 | 권장 | 비고 |
|---|---|---|
| **Cross-correlation / template matching** | ✓ 1순위 | online correction best |
| Autocorrelation | △ cadence bootstrap only | |
| FFT | ✗ | 너무 windowed/laggy |
| Hilbert transform | ✗ | asymmetric/non-sinusoidal gait fragile |

### 2.6 Prediction at future τ

```
τ       = now_control - vision_timestamp + command_lead
phi_pred = phi + omega*τ + 0.5*alpha*τ²
q_pred   = μ(phi_pred)             # cubic Hermite interp over phase bins
P_pred   = J P J^T + Σ_template(phi_pred) + σ_model² * τ²
```

**Interpolation**: linear OK (4-6주 budget), cubic Hermite (derivative discontinuity 줄임).

**Uncertainty gate**: predicted HS uncertainty > 25ms 또는 angle uncertainty > actuator-safe bound 면 assistance disable.

### 2.7 Heel strike event prediction

```
phi_HS_L = 0           # calibration 후 정의
phi_HS_R = π
t_HS     = wrap(phi_HS - phi) / omega
```

**Trigger 조건 (전부)**:
- `0 ≤ t_HS ≤ 100ms` (실용 윈도우)
- residual template error 낮음
- visual confidence 높음

**절대 X**: 1 frame delayed input 만으로 HS trigger.

---

## 3. Cold start cascade

```
시점                          | predictor                | rationale
─────────────────────────────┼──────────────────────────┼──────────
vision pose 첫 frame          | L1 constant velocity      | 즉시 사용 가능
vision pose ≥3 valid frames   | L2 constant acceleration  | jerk 보임
3 strides + criteria          | L3 phase-locked           | cycle 학습 완료
```

**L3 활성 criteria (전부)**:
- 3 complete strides 완료
- phase residual RMS < 10-15% of cycle
- cadence CV < 10%

**L3 비활성 trigger** (다시 L2 또는 L1 fallback):
- missed detections (>60ms gap)
- cadence jump > 20% within 1 stride
- template residual spike > 3σ

---

## 4. Failure modes + Fallback chain

### 4.1 Cycle assumption breaks (clinical patient)

| 패턴 | 예측 깨짐 |
|---|---|
| Shuffling gait | small omega, ambiguous phase |
| Freezing | omega→0, alpha undefined |
| Start/stop | 모든 cycle 정보 fresh |
| Turning | bilateral asymmetry |
| Therapist-assisted steps | external force, cadence noise |
| Hemiparetic (stroke) | L/R timing 비대칭 |
| Toe-walking | ankle phase pattern 변경 |
| Foot drag | swing phase ankle template 깨짐 |
| Crouch gait | 모든 joint mean 다름 |
| Spastic catch | sudden velocity spike |
| Missed/extra steps | phase 점프 |
| Intentional non-periodic | template match 실패 |

### 4.2 Divergence detection

| 신호 | threshold |
|---|---|
| Template residual | > 3σ |
| Phase innovation | > 25% cycle |
| Cadence change in 1 stride | > 20% |
| Missing core joints | > 60ms gap |
| Predicted HS not confirmed | within 100ms |
| L/R phase separation | ≠ near π |

### 4.3 Fallback chain

```
L3 phase-locked
   ↓ (divergence detected)
L2 constant acceleration
   ↓
L1 constant velocity
   ↓
hold-last-good ≤ 50ms
   ↓
pretension safe mode (5N, C++ watchdog)
```

**Bounded by**: torque/pretension limits, angle-rate limits, predictor uncertainty.
**Graceful degrade**: high uncertainty → reduce assistance amplitude *전에* 완전 dropout.

### 4.4 Watchdog policy (Codex Q7)

**모든 keypoint depth 가 carry**:
- confidence
- L/R consistency
- disparity bounds
- EKF innovation residual
- age (latest measurement timestamp)

**Trigger conditions**:

| 조건 | 행동 |
|---|---|
| Sparse depth fail > 2 frames | freeze phase advance |
| EKF innovation > 3σ threshold | conservative impedance / transparent mode |
| Failure persists > 100ms | disable timing-critical assist for that stride |
| Detector confident on wrong limb/device edge | hard fault (silent failure worst case) |

**철칙**: prediction-only control 절대 indefinitely 계속 X. 항상 measurement validation.

---

## 5. C++ implementation skeleton

### 5.1 Placement in control loop

```cpp
// 100Hz+ control loop
while (running) {
    auto now = clock_gettime(CLOCK_MONOTONIC);

    // 1. Read latest vision pose from SHM (seqlock, non-blocking)
    PoseSample sample;
    if (shm_reader.read_latest(&sample)) {
        // 2. Convert keypoints → joint angles
        JointAngles q = compute_joint_angles(sample.kpts_3d_m);
        
        // 3. Predictor update with vision timestamp
        predictor.update(q, sample.ts_ns);
    }

    // 4. Predict to actuator-applied time
    int64_t target_ns = now + actuator_lead_ns;
    PredictedState pred = predictor.predict(target_ns);

    // 5. Uncertainty gate
    if (pred.cov_phi > MAX_PHI_COV || pred.t_HS_unc > 25e-3) {
        impedance_ctrl.set_pretension_safe();
    } else {
        impedance_ctrl.apply(pred.q_predicted, pred.cov_q);
    }

    // 6. Heel strike event emission (latched)
    if (pred.t_HS_L > 0 && pred.t_HS_L < 100e-3 && pred.confidence > 0.8) {
        events.emit_HS_left(now + pred.t_HS_L * 1e9);
    }
    // ... same for R

    rate_limiter.sleep_until_next_period(100); // Hz
}
```

### 5.2 Predictor budget

| 항목 | 예산 |
|---|---|
| `predictor.update()` | < 0.1 ms |
| `predictor.predict()` | < 0.1 ms |
| **Total per loop iteration** | **< 0.2 ms** |
| **Hard ceiling** | 0.5 ms (이상이면 implementation 잘못) |

100Hz loop = 10ms cycle → predictor 가 < 5% 차지.

---

## 6. Prior art (Codex cited)

| Paper | 기여 |
|---|---|
| Righetti, Ijspeert (2008) | Adaptive Frequency Oscillators (AFO) — phase lock 기초 |
| Thatte et al. | EKF gait phase control |
| Kang et al. | Real-time phase estimation for exoskeleton control |
| Stenum (2021) PLOS Comp Bio | OpenPose vision gait events — MAE 20ms baseline |
| MediaPipe gait (2023) | event detection 20-30ms MAE |

---

## 7. 작업 순서 (4-6주 budget)

| Week | 작업 |
|---|---|
| 1 | L1 + L2 implementation, vision SHM consumer wire-up |
| 2 | Cycle template μ(φ) bootstrap, phase estimator (cross-correlation), cold start cascade |
| 3 | EKF wrapping + uncertainty + rejection gates, divergence detection |
| 4 | Fallback chain + pretension safe mode + C++ watchdog integration |
| 5 | Bench validation (recorded vision data) + falsification tests |
| 6 | Patient dry-run + event log analysis |

### Falsification tests (critical)

- Healthy subject 30 step → predicted HS error histogram (target: 95% within 30ms of true)
- Asymmetric gait simulation → fallback rate < 20%
- Sudden stop → L3 → L2 transition < 100ms
- Cadence change ramp → predictor tracking error < 15% phase

---

## 8. Validation metric (clinical success ≠ raw latency)

Codex 강조: **raw latency alone is NOT clinical success metric**.

진짜 metric:
- Predicted HS error histogram (mean + p99 + worst case)
- Missed HS rate (false negative)
- Phantom HS rate (false positive)
- Assistance timing relative to true gait events
- Patient subjective fit (RPE, VAS)
- Energy consumption (%MVC reduction)

---

## 9. References

- Stereolabs depth: https://www.stereolabs.com/docs/depth-sensing/using-depth
- Stereolabs calibration: https://www.stereolabs.com/docs/video/camera-calibration
- TensorRT DLA: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html
- NVIDIA VPI stereo: https://docs.nvidia.com/vpi/algo_stereo_disparity.html
- Stenum 2021 video gait: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008935
- Atalante 2025 (IMU/operator): https://link.springer.com/article/10.1186/s12984-025-01621-z
- Honda Walking Assist (hip motor sensors): https://global.honda/en/newsroom/worldnews/2013/c131112Walking-Assist-Device.html
