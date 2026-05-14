# Plan D EKF 완전 이해 — 한국어 Tutorial

> **목적**: 사용자가 Plan D EKF를 진짜 이해해서 직접 디버깅 / 튜닝 / 개선할 수 있도록.
> 빠른 답이 아니라 **확실한 이해**가 목표. 비유 → 수학 → 코드 trace → 우리 시스템 specific 적용.

## 목차
1. [왜 EKF가 필요한가](#1-왜-ekf가-필요한가)
2. [State는 무엇인가](#2-state는-무엇인가)
3. [Process model — 시간 진행 방정식](#3-process-model)
4. [Measurement model — 측정과 state 의 연결](#4-measurement-model)
5. [Kalman gain — 최적 융합의 본질](#5-kalman-gain)
6. [Predict + Update — 매 frame 무슨 일이 일어나는가](#6-predict-update)
7. [Cascade L1 / L2 / L3 — 정확도 단계](#7-cascade)
8. [실시간 timeline — 우리 시스템에서](#8-realtime-timeline)
9. [무엇이 잘못될 수 있는가](#9-failure-modes)
10. [Paper-grade references](#10-references)

---

## 1. 왜 EKF가 필요한가

### 문제 설정

H-Walker가 사람 보행을 보조하려면 다음을 알아야:
- **지금 사람이 보행 cycle의 몇 % 위치에 있는가** (= phase φ)
- **얼마나 빠르게 진행 중인가** (= cadence ω)
- **50ms 후 어디 있을지 예측 가능한가** (= forecast)

Vision (ZED 카메라 + YOLO)이 매 frame 측정 가능한 것:
- ✓ Hip, knee, ankle 의 **3D 위치**
- ✓ 각 joint **각도**

Vision이 직접 측정 못 하는 것:
- ✗ **Phase φ** — 위치만 보면 "지금 cycle 어디인지" 모름
- ✗ **Cadence ω** — 1 frame만 보면 속도 모름
- ✗ **다음 순간 어디 있을지** — 모델 없으면 예측 불가

### 비유 1 — 시계 추 사진

흔들리는 시계 추 사진 1장:
```
   ●
  ╱        ← 추가 어디에 있는지 알지만
─────         어느 방향으로 움직이고
              얼마나 빨리 흔들리는지 모름
```

사진 여러 장 시간순:
```
t=0s     t=0.5s    t=1.0s
  ●        ●         ●
 ╱         │         ╲     ← 이제 알 수 있음:
─────    ─────    ─────       방향 (오른쪽 진행)
                              주기 (2초/cycle)
                              미래 (다음 0.5초 후 오른쪽 끝)
```

**이게 EKF가 하는 일** — 시간순 관측을 모아 *직접 측정 불가능한 dynamics*를 추정.

### 비유 2 — GPS 내비게이션

내비게이션이 차 속도를 어떻게 알까?

```
GPS 측정만:
  매 1초마다 위치 (±5m 노이즈)
  속도 = Δ위치 / Δt → 노이즈 ×√2 더 커짐
  → 그냥 쓰면 부정확

Kalman Filter:
  + 물리 가정: "차는 갑자기 속도 안 바뀜"
  + 물리 가정: "차는 도로 위에 있음"
  = 노이즈 GPS + 가정 → 정확한 위치 + 속도
```

우리 시스템 동일:
| GPS 내비 | H-Walker |
|---|---|
| GPS 위치 ±5m | Vision hip_z ±5mm |
| Kalman filter | Extended Kalman Filter (EKF) |
| 가정: 차는 도로 위 | 가정: 사람은 일정 cadence로 걷는다 |
| 출력: 정확한 위치 + 속도 | 출력: 정확한 phase + cadence |

### 비유 3 — 메트로놈 박자 분석

```
"딸깍! ... 딸깍! ... 딸깍! ..."

1번째 듣고:  박자 모름
2번째 듣고:  "1초 박자네?" (1초 후 들음)
3번째 듣고:  확신 ↑
10번째 듣고: 다음 딸깍 99% 예측 가능
```

**Plan D = hip 위아래 진동을 메트로놈 박자처럼 듣는 분석기**.

### 왜 EKF인가, 왜 다른 방법 안 쓰는가

| 방법 | 장점 | 단점 | Plan D 사용? |
|---|---|---|---|
| **단순 시간 차분 ω = Δφ/Δt** | 단순 | 노이즈 ×√2, 음수 ω 가능 | ✗ |
| **FFT (주파수 분석)** | 노이즈 robust | 1.5초+ 지연, 실시간 X | ✓ (Hilbert에서 부분 사용) |
| **HS event interval** | 가장 정확 | 1번째 HS까지 못 씀, event 누락 시 fail | ✓ (L2 anchor) |
| **EKF (state estimation)** | 모든 정보 융합, 매 frame 출력, predict 가능 | 튜닝 필요 | ✓ (메인) |
| **AFO (적응 발진기)** | 가장 robust | 구현 복잡, paper scope | ✗ (대안) |

EKF는 세 가지를 **하나의 framework로 융합**:
- 시간 차분 (predict step)
- 측정 model (measurement update)
- Process 가정 (sense model)

---

## 2. State는 무엇인가

### State 정의

```
x_k = [φ_k, ω_k, α_k]ᵀ ∈ ℝ³

φ ∈ [0, 2π)   gait cycle phase (현재 cycle 진행도)
ω ∈ ℝ          cadence (rad/s = 1/cycle 동안 phase 진행 속도)
α ∈ ℝ          cadence acceleration (rad/s² = ω의 변화 속도)
```

### 단위 직관

```
ω = 2π rad/s  ↔  1 cycle/sec  =  1 Hz  =  사람이 1초에 한 발자국
ω = π rad/s   ↔  0.5 cycle/sec = 0.5 Hz = 2초에 한 발자국 (느린 walking)
ω = 4π rad/s  ↔  2 cycle/sec  = 2 Hz   = 0.5초에 한 발자국 (빠른 walking)
```

성인 healthy walking cadence ≈ 0.8–1.5 Hz, 즉 ω ≈ 5.0–9.4 rad/s.

### 왜 phase가 state 인가 (alternative와 비교)

**대안 A**: 6개 joint 각도를 state로
```
x = [q_hip_L, q_knee_L, ..., q̇_hip_L, ...]  ← 12-state
```
- 단점: joint 들이 *독립이 아님*. 모두 같은 cycle에 lock. 12 차원이지만 실제 유의미한 변동은 1 차원 (cycle 진행).
- **본질 1차원** — phase 가 그것.

**대안 B**: Hip 위치만 state
```
x = [hip_x, hip_y, hip_z, ẋ, ẏ, ż]  ← 6-state
```
- 단점: cycle 정보 없음. "다음 step 어디일지" 모름.

**우리 채택**: Phase + cadence
- **1 phase 로 6 joint 위치 예측 가능** (template lookup)
- Predict = "phase 50ms 진행" 만 하면 모든 joint 미래 위치 산출

### α 가 왜 필요 (L1 → L2 → L3 차이)

```
L1 (Warmup):  α = 0 가정 ── "사람이 일정 속도로 걷는다"
              → ω constant 가정
              → 가속/감속 시 추적 못 함
              → cold-start 만 목적

L2 (Stride-locked):  α 가 random walk
                     → cadence drift 추적 가능
                     → 일반 walking 정확

L3 (Template-locked):  α + cycle template
                       → 매 cycle 형태 자체 학습
                       → 가장 정확
```

---

## 3. Process Model

### 방정식

```
φ_{k+1} = φ_k + ω_k·Δt + ½·α_k·Δt²    (mod 2π)
ω_{k+1} = ω_k + α_k·Δt
α_{k+1} = α_k + w_α                    (white noise driver)
```

여기서 Δt = 1/60 s ≈ 16.7 ms (frame 간격).

### 직관

- **φ**: phase = ω · t + ½α · t² (등가속 운동학과 동일)
- **ω**: cadence는 cadence_accel × dt 만큼 변함
- **α**: 어떻게 변하는지 *모름* → "random walk" 모델링

### 왜 random walk?

```
실제: α는 사람 의도 (천천히 걸어야지) + 환경 (오르막 만났네) 으로 변함
       정확한 model 없음. 우리가 알 수 없는 process.

해결: w_α ~ N(0, σ²_α)  ← Gaussian noise 로 모델링
       "매 frame α 가 약간 변동" 가정.

효과: σ_α 크면  → reactive (빠른 변화 추적, 노이즈에 약함)
      σ_α 작으면 → smooth (느린 적응, 노이즈에 강함)
```

이게 **process noise covariance Q** 의 한 element.

### Q 매트릭스 — 튜닝 가장 중요한 부분

```
Q = diag(σ²_φ·Δt, σ²_ω·Δt, σ²_α·Δt)
```

- **σ_φ = 0.01 rad/√s**: phase에 무작위 노이즈 매 frame 0.013 rad (≈ 0.7°)
- **σ_ω = 0.5 rad/s/√s**: cadence drift 허용량
- **σ_α = 1.0 rad/s²/√s**: cadence 가속 변동성

너무 작으면 (Q ≈ 0): state stiff, measurement 안 따라감. 사람이 갑자기 멈춰도 EKF 계속 같은 ω 사용 → 부정확.

너무 크면 (Q 큼): state 흔들림, 노이즈가 그대로 학습됨.

**튜닝 방법** (Bar-Shalom 2001, Ch. 6):
1. 안정 walking 10초+ 측정
2. ω의 std 계산
3. σ_ω ≈ measured_std × √(2/Δt) (Brownian motion principle)
4. Innovation 의 크기로 검증 (너무 크면 R 키움)

---

## 4. Measurement Model

### Vision이 보는 것 = 측정 vector z

```
z = [hip_vertical, q_left_thigh, q_left_knee, q_left_shank, q_right_thigh, q_right_knee, q_right_shank]ᵀ
```

7개 measurement (1 hip + 6 joint).

### Predicted observation h(x̂)

**L1 (Hilbert cold-start)**:
```
ẑ_hip = A·cos(φ - φ_0)  + bias
         ↑ amplitude (envelope)
         ↑ phase offset (Hilbert 시작점)
         ↑ DC offset
```

**L3 (Template lookup)** — 가장 정확:
```
ẑ_q[i] = template[i](φ)    ← 각 joint 의 평균 cycle profile

template[i] = 함수: φ → 각도
              학습됨: 매 stride 마다 update
              N_bins = 128 (cycle 을 128개 구간으로)
              interpolation: linear / Catmull-Rom
```

### Measurement noise R

```
R = diag(σ²_hip_z, σ²_q[0], ..., σ²_q[5])
```

각 component:
- **σ_hip_z** = stereo depth uncertainty + 사람 motion variability
  - ZED PERFORMANCE depth: σ_z = Z²·σ_d/(fx·b) ≈ 5 mm at 1 m
- **σ_q[i]** = keypoint detection σ × bone length 영향
  - 기본 0.05 rad (~3°)
  - depth_hold held 시 0.20 rad (~11°)

### Innovation gating

```
y = z - ẑ                      ← innovation (잔차)
χ² = yᵀ·S⁻¹·y                  ← Mahalanobis 거리
if χ² > threshold (보통 9):
    measurement reject          ← outlier (occlusion, NaN burst)
```

---

## 5. Kalman Gain — 최적 융합의 본질

### 핵심 식

```
S = H·P·Hᵀ + R       ← innovation covariance (3x3 → 7x7)
K = P·Hᵀ·S⁻¹         ← Kalman gain (3x7)
```

### 직관 — "측정을 얼마나 믿을지" 자동 결정

```
K 가 크면  → measurement 신뢰 (state 크게 update)
K 가 작으면 → process model 신뢰 (state 거의 안 변함)
```

**언제 K 가 작아지나** (process 신뢰):
- R 큼 = measurement noisy (NaN burst, depth_hold held)
- P 작음 = state 이미 정확함

**언제 K 가 커지나** (measurement 신뢰):
- R 작음 = measurement 정확
- P 큼 = state uncertainty 큼 (cold-start, gap 후 등)

### 수학적 의미

```
K = "측정 정보량 / (측정 정보량 + process 정보량)"
   = "관측 information 대비 prior information 비율"
```

**Optimal in MMSE** (Minimum Mean Square Error) sense. Bar-Shalom 정리.

---

## 6. Predict + Update — 매 frame 무슨 일

### 매 16.7 ms 마다 실행되는 한 사이클

```python
# === Step 1: Predict (시간 진행) ===
F = [[1, dt, 0.5*dt²],   # process Jacobian (3x3)
     [0,  1,    dt   ],
     [0,  0,    1    ]]

x_prior = F @ x_post_prev
P_prior = F @ P_post_prev @ F.T + Q

# === Step 2: Measurement (vision input) ===
z = [hip_vertical_measured, q[0], q[1], ..., q[5]]

# === Step 3: Predicted observation ===
z_pred = h(x_prior)
       = [A*cos(phi - phi0) + bias,
          template[0](phi), template[1](phi), ...]

# === Step 4: Innovation ===
y = z - z_pred

# === Step 5: Jacobian H of h (linearize around x_prior) ===
H = [[-A*sin(phi - phi0), 0, 0],    # ∂z_hip/∂phi
     [dtemplate[0]/dphi,  0, 0],    # ∂z_q[0]/∂phi
     ...
     [dtemplate[5]/dphi,  0, 0]]

# === Step 6: Innovation covariance + Kalman gain ===
S = H @ P_prior @ H.T + R           # 7x7
K = P_prior @ H.T @ inv(S)          # 3x7

# === Step 7: Posterior update ===
x_post = x_prior + K @ y            # state corrected
P_post = (I - K @ H) @ P_prior      # uncertainty reduced

# === Step 8: Output ===
publish(phi=x_post[0], omega=x_post[1], alpha=x_post[2])
publish(forecast_50ms = predict_at(x_post, dt=0.050))
```

### 코드 trace — 실제 우리 시스템

`src/perception/plan_d_prototype/predictor.py:80`:
```python
def feed(self, t_now, q, sigma_per_joint, hip_z_world_m):
    self.cascade.feed(t_now, q, sigma_per_joint, hip_z_world_m)
```

`cascade.py:feed()` 안에서:
```python
# 1. Hilbert envelope update (sliding window)
self.hilbert.update(t_now, hip_z)
# → instantaneous phase estimate (cold-start용)

# 2. EKF L1 always
self.l1.predict(dt)         # F@x, F@P@F.T+Q
self.l1.update_phi(hilbert.phi_estimate, sigma)   # Kalman update

# 3. Heel-strike detection
if hs_detected:
    stride_count += 1
    ω_from_HS = 2π / (t_now - prev_HS_time)
    self.l2.anchor(phi=0, omega=ω_from_HS)

# 4. L2 activation (after 1st stride)
if stride_count >= 1 and not l2_active:
    l2_active = True

if l2_active:
    self.l2.predict(dt)         # 더 정확한 EKF
    self.l2.update(q, sigma)    # 6-joint measurement

# 5. L3 activation (3+ strides + template ready)
if stride_count >= 3 and template.touched_fraction >= 0.5:
    l3_active = True

if l3_active:
    self.l3.predict(dt)
    self.l3.update(q, sigma)     # template-based measurement
    if innovation_χ² > threshold:
        l3.demote_to_l2()        # rollback (Codex P7 fix needed)

# 6. Output (priority: L3 > L2 > L1)
self.phi = l3.phi if l3_active else l2.phi if l2_active else l1.phi
self.omega = ...
```

---

## 7. Cascade L1 / L2 / L3

### L1 — Warmup / Hilbert cold-start

```
State: x = [φ, ω]ᵀ       ← 2-state (α=0 가정)
Measurement: hip_vertical 1-D
Update source: Hilbert envelope (1.5s sliding window FFT)
σ_φ output: 1~2 rad     ← 큰 uncertainty
```

**언제 사용**: 처음 1.5초 (Hilbert window 채워질 때까지).

### L2 — Stride-locked

```
State: x = [φ, ω, α]ᵀ      ← 3-state
Measurement: hip_vertical + 6 joint angles
Update source: HS event 마다 phase=0 anchor
σ_φ output: 0.3~0.6 rad
```

**언제 활성화**: stride_count ≥ 1 (첫 HS 검출 후).

### L3 — Template-locked

```
State: x = [φ, ω, α]ᵀ
Measurement: 6 joint angles
Update source: cycle template lookup (학습된 평균 cycle)
σ_φ output: 0.05~0.15 rad    ← 가장 정확
```

**언제 활성화**: stride_count ≥ 3 AND template_touched ≥ 50%.

### Transition rules

```
L1 → L2: stride_count ≥ 1
L2 → L3: stride_count ≥ 3 AND template.touched ≥ 0.5 AND innovation OK
L3 → L2: innovation χ² > threshold (template과 어긋남)
L2 → L1: cadence jump > 20% (사람 갑자기 멈춤/뛰기)
L1 → hold: vision loss > 60ms (pretension fallback)
```

---

## 8. 실시간 Timeline — 우리 시스템

```
─────────── frame k (t = k × 16.7ms) ───────────

t=0ms ZED grab
t=12ms YOLO TRT inference → 6 keypoint 2D pixels
t=12.1ms ZED depth retrieve → keypoint Z values
t=12.2ms batch 3D backproject → 6 keypoints 3D positions
t=12.3ms depth_hold (if held) → fill missing keypoints
t=12.4ms bone_constraint apply → outlier projection
t=12.5ms compute_joint_state(world_up_vec=...) → state.q (6 angles)
                ↓
t=13ms predictor.feed(t_now, q, sigma, hip_vertical)
       │
       ├─ hilbert.update (sliding 1.5s window)        0.05ms
       ├─ l1.predict + update                         0.10ms
       ├─ HS event check + stride_count++             0.02ms
       ├─ l2.predict + update (if active)             0.20ms
       ├─ l3.predict + update (if active)             0.20ms
       ├─ template.update (every stride)              0.05ms
       └─ output: phi, omega, alpha 계산              0.01ms
                ↓
t=13.5ms forecast.publish:
         phi_now = 4.21 rad
         omega = 6.31 rad/s (1.004 Hz)
         q_pred[6] = template_lookup(phi + omega*0.050)
         t_HS_next = (2π - phi) / omega = 0.33s
                ↓
t=14ms SHM /hwalker_forecast 업데이트 (seqlock write)
                ↓
─────────── frame k+1 (16.7ms later) ───────────

총 vision-side latency: 14ms (TRT inference 가 dominant)
EKF cost: 0.5ms 만 (전체의 4%, 무시 가능)
```

### 첫 60초 walking 동안 EKF 학습 과정

```
t=0~1.5s   L1 cold-start
           Hilbert 가 frequency 검출 중
           ω = default 6.28 rad/s 유지
           φ 부정확

t=1.5s    Hilbert 첫 출력
          → ω cold-start (예: 6.0 rad/s)

t=1.5~3s  L1 진행
          ω 점진 학습
          첫 HS event 임박

t=3s      첫 heel-strike 검출
          stride_count = 1
          L2 activate
          φ = 0 anchor

t=3~6s    L2 학습
          매 cycle ω 정확해짐
          template 학습 시작 (cycle profile 평균)

t=6s      stride_count = 3, template_touched = 50%
          L3 activate

t=6s~     L3 운영
          σ_φ = 0.1 rad (paper-grade)
          q_pred[6] 정확
          forecast 50ms 사용 가능
```

---

## 9. 무엇이 잘못될 수 있는가 (failure modes)

### Failure 1 — Wrong hip signal convention

우리 시스템에서 발생한 사건. Codex consult #5 + Phase 2B:
```
Plan D Hilbert envelope 은 hip VERTICAL motion 기대
우리 코드 가 ZED Z (= optical axis = horizontal distance) 입력
→ walking signal 이 quasi-DC (변화 거의 없음)
→ Hilbert 가 dominant frequency 못 찾음
→ ω 잘못 학습 (0.10 Hz)
→ cascade L1 stuck
```

**Fix**: world_up projection (Phase 2B 완료).

### Failure 2 — Predictor never called

Codex #1 + Phase 2A:
```
pipeline_main.py 가 world_up_vec 안 전달
→ state.{thigh,shank}_inclination = None
→ six_valid check fail
→ RuntimeError "not all 6 joints valid"
→ bare except 가 silent하게 swallow
→ predictor.feed() 매 frame 안 호출
→ cascade transition = 0 (EKF 작동 안 함)
```

**Fix**: world_up_vec 전달 + feed counter logging (Phase 2A 완료).

### Failure 3 — Held keypoint sigma not propagated

Codex #7:
```
depth_hold held 시 그 keypoint 의 σ 만 증가
하지만 6 joint 각도는 keypoint 여러 개로 계산
예: left_ankle held → q[1] (knee_flex) + q[2] (shank_inc) 둘 다 영향
하지만 σ_per_joint[2] 만 증가
→ knee_flex 에 stale ankle 정보가 fresh 처럼 들어감
```

**Fix**: keypoint → angle 영향 매핑 (Phase 3 완료).

### Failure 4 — Template self-confirming wrong phase

Codex #12:
```
Hilbert 가 arbitrary phase offset 으로 시작
Template 이 그 잘못된 phase 에 lock
HS event 가 phase reset 없으면 영원히 어긋남
```

**Status**: 부분적으로 다뤄짐. HS reset code 있지만 stride 1번 후에야.

### Failure 5 — Cadence too slow (< 0.4 Hz)

```
사용자가 매우 천천히 walking
ω < 1.5 rad/s (0.24 Hz)
→ Hilbert window (1.5s) 안에 cycle 완성 안 됨
→ frequency 검출 fail
→ L1 stuck
```

**Mitigation**: window 늘리기 (2.5s) — 반응 늦지만 학습 가능.

---

## 10. Paper-grade References

| Paper | Year | Venue | 우리 적용 |
|---|---|---|---|
| **Lerner et al.** "Robotic exoskeleton for crouch gait" | 2018 | IEEE T-NSRE | Phase-locked impedance control 권위 |
| **Quinlivan et al.** "Soft exosuit assist vs metabolic cost" | 2017 | Science Robotics | Cable-driven + phase trigger |
| **Plant et al.** "Hilbert envelope for gait phase" | 2016 | — | 우리 cold-start 직접 reference |
| **Asseldonk et al.** "Adaptive oscillator phase estimation" | 2014 | — | Plan D 접근 alternative |
| **Thatte, Shah, Geyer** "EKF-Based Gait Phase Estimation" | 2019 | IEEE RA-L | 우리 L3 직접 비교 baseline |
| **Embry et al.** "Phase + Task Variable Kinematics" | 2018 | IEEE TNSRE | State expansion reference |
| **Medrano et al.** "Real-Time Gait Phase + Task" | 2023 | IEEE T-RO | Walker-user distance state 근거 |
| **Righetti, Ijspeert** "Adaptive Frequency Oscillators" | 2009 | — | AFO alternative (random-walk ω 한계) |
| **Bar-Shalom et al.** "Estimation with Applications" | 2001 | Wiley | EKF textbook standard |

### Reading 우선순위 (우리 paper writing)

1. **Lerner 2018** — phase-locked control 의 의미 정확히 이해
2. **Quinlivan 2017** — cable-driven assist 의 phase trigger 사용 예시
3. **Plant 2016** — Hilbert envelope cold-start 원리 (우리 L1 직접 기반)
4. **Thatte 2019** — EKF 가 phase estimation 에 가장 유사한 접근
5. **Embry 2018** + **Medrano 2023** — state expansion 으로 paper 강화 시 reference
6. **Asseldonk 2014** + **Righetti 2009** — 대안 method 의 짧은 언급 (limitation section)

---

## 11. 빠른 참조 — 코드 위치

| 개념 | 파일 | 라인 |
|---|---|---|
| State + cascade entry | `src/perception/plan_d_prototype/predictor.py` | 1-100 |
| L1 EKF (2-state) | `src/perception/plan_d_prototype/ekf_l1.py` | 전체 |
| L2 EKF (3-state) | `src/perception/plan_d_prototype/ekf_l2.py` | 전체 |
| L3 EKF (template-driven) | `src/perception/plan_d_prototype/ekf_l3.py` | 전체 |
| Cascade transitions | `src/perception/plan_d_prototype/cascade.py` | 200-400 |
| Hilbert cold-start | `src/perception/plan_d_prototype/hilbert_phase.py` | 전체 |
| Template learning | `src/perception/plan_d_prototype/cycle_template.py` | 전체 |
| Joseph form covariance | `src/perception/plan_d_prototype/utils.py` | 73 |
| EKF feed entry (production) | `src/perception/realtime/pipeline_main.py` | 630-820 |
| Visualisation | `scripts/visualize_ekf_learning.py` | 전체 |

---

## 다음 단계 — Mac에서 가능한 작업

1. **이 문서 다 읽기** (5-10분)
2. **Synthetic walking signal generator 실행** — 진짜 gait signal로 EKF 작동 시각화
3. **NPZ replay** — `scripts/visualize_ekf_learning.py` 사용 (이미 만듦)
4. **Code walkthrough** — `predictor.py` 한 줄씩 보면서 위 코드 trace 직접 따라가기

질문 있으면 그 줄을 짚어서 ask.
