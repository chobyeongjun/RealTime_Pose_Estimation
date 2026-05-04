---
date: 2026-05-04
project: realtime-vision-control
topic: Paper 1 MPU — IMU vs Vision 비교 실험 (Exp A/B/C)
result: planned
related_doc: ../paper/vision-necessity.md
related_plan: ~/.claude/plans/scalable-splashing-puffin.md
---

# Paper 1 MPU — IMU vs Vision 비교 실험 protocol

## 목표

Paper 1의 bulletproof contribution을 입증하는 3개 실험.

> Bulletproof: "Vision은 cable controller의 reference signal(walker-segment 절대 거리)을 stroke 환자에게도 deploy 가능하게 만드는 유일한 수단이다."

3개 실험은 각각 vision unique 결과 1/2/3에 1:1 대응.

## Hardware checklist

| 장비 | 보유? | Exp A | Exp B | Exp C | 대체 |
|---|---|---|---|---|---|
| ZED X Mini | ✅ | ✅ | ✅ | ✅ | — |
| Jetson Orin NX | ✅ | ✅ | ✅ | ✅ | — |
| AK60 motor + cable | ✅ | ✅ | — | — | — |
| **EBIMU 6개** | **⚠️ 확인 필요** | ✅ | ✅ | ✅ | 단일 EBIMU (약한 baseline) |
| Wheel encoder (walker frame) | ⚠️ 확인 필요 | ✅ | — | — | Cable encoder만 |
| Load cell (cable) | ⚠️ 확인 필요 | ✅ | — | — | Motor current proxy |
| Treadmill | ✅ (별 lab 협력?) | ✅ | — | ✅ | Overground (Exp A는 어려움) |
| Ankle weight 1-2kg | ❌ → 가정용품 | — | — | ✅ | Strap-on weight |
| Knee brace | ❌ → 가정용품 | — | — | ✅ | Compression sleeve |

**Critical bottleneck**: EBIMU 6개. 없으면 Exp A/B/C 모두 baseline 비교 불가능.
- 옵션 1: 구입 (EBMotion V5 1센서 ~290,000원 × 6 = ~174만원, 수신기 별도)
- 옵션 2: 단일 EBIMU로 한 segment만 측정 (약한 baseline. Reviewer가 "3-joint 비교 부족" 지적 가능)
- 옵션 3: Teensy 내장 IMU 기반 합성 baseline (매우 약함, 권장 안 함)

## IRB 요구사항

Paper 1 MPU는 **모두 healthy subject** + **simulated patient (Exp B)** → minimal risk → expedited IRB or informed consent memo.

- Exp A: healthy adults, 트레드밀 30분 walk. Force <20N (pretension fallback). Non-invasive.
- Exp B: healthy adults + 모의 환자 (walker 잡고 일어서기 어려운 척). 시간 측정 + Likert scale.
- Exp C: healthy adults + ankle weight/brace simulated asymmetry. Non-invasive.

정식 stroke patient 임상은 Paper 2 영역.

---

## Exp A — Driftless cable length tracking ⭐ CORE

### Hypothesis
Vision-based cable length reference (`L = ‖walker − ankle_vision‖`)가 30분 보행 후 drift <2cm를 유지하고, IMU+encoder 기반 추정은 >5cm drift를 보인다.

### Setup
- 피험자: N=4-6 healthy adults
- 환경: 트레드밀 0.8-1.2 m/s, 30분 × 2 session (반복성)
- 측정: 시작/종료 시점에 동일 자세 사진 (mocap reference 1점) + 매 frame 3 channel record
- Channels:
  1. Vision: `‖walker_frame_origin − ankle_3D_world‖` (m)
  2. IMU+encoder: standing reference + 적분된 segment angle → forward kinematics로 추정 거리
  3. Mocap (있으면): ground truth

### Conditions
- C1: Vision-only L_target
- C2: IMU+encoder L_target
- C3: Vision+IMU EKF fusion (선택, 시간 남으면)

### Primary metric
- **Drift over 30min** (cm) = |L(t=30min) − L(t=0)| at same standing pose
- 단위: cm

### Secondary metrics
- Drift rate (cm/min)
- Cable length RMSE between methods (frame-by-frame)
- Sagittal plot 왜곡도 (각 method의 hip-knee-ankle 수직성)

### Statistical analysis
- Paired t-test (Vision vs IMU drift, N=4-6)
- 또는 Friedman test (비모수, 3 condition)
- Effect size (Cohen's d) 보고

### Predicted result
- Vision: <2cm drift (gravity-aligned world frame, no integration)
- IMU+encoder: 5-10cm drift (segment angle 적분 + STA)
- **차이 statistically significant (p<0.05)** 기대

### Failure mode
- Vision도 >3cm drift → world frame 검증 필요 (`verify_world_frame.py` 다시 실행). 카메라 mounting drift 의심.
- IMU+encoder도 <2cm → STA가 예상보다 작거나 적분 알고리즘이 우수. 우리 주장 weakened. 단 stroke 환자 case (Exp B)에서는 여전히 standing reference 못 잡음 → 살아남음.

### Time
- 1주 (피험자 모집 + 실험 + 분석)

---

## Exp B — Setup time + clinical deployability

### Hypothesis
Vision-only 시스템 setup time 3-5분 << IMU-based 시스템 15-20분. **Stroke 환자 모의 시나리오에서 IMU calibration은 standing balance risk 때문에 deploy 불가능**.

### Setup
- 피험자: N=2-3 healthy adults + N=2 모의 환자 (walker 잡고 일어서기 어려운 척, balance 잃을 수 있다고 시뮬레이션)
- 환경: 평지 walker 옆, calibration session
- 측정 항목:
  1. Setup wall-clock time (sec)
  2. 환자 standing time during calibration (sec)
  3. Subjective comfort (Likert 5점, 1=불편 5=편함)
  4. Hardware fail event count (sensor 떨어짐, calibration 재시도)

### Conditions
- C1: Vision-only setup (camera mount + verify_world_frame.py 실행)
- C2: IMU-only setup (6 sensor 부착 + N-pose calibration + walking calibration)

### Primary metric
- **Setup time (min)**
- **모의 stroke 환자 deploy 성공 여부** (binary: 환자가 standing calibration 못 끝내면 fail)

### Secondary metrics
- Subjective comfort Likert (5점)
- 30분 후 피부 자극 정도 (IMU adhesive)
- Re-calibration 빈도 (slip / drift 의심 이벤트)

### Statistical analysis
- 기술 통계 (mean ± SD)
- Likert: Mann-Whitney U test
- Setup time: paired t-test

### Predicted result
- Vision: 3-5min setup, 모의 환자 100% 성공
- IMU: 15-20min setup, 모의 환자 0-50% 성공 (standing 못 버티면 fail)
- Likert: Vision 4.5/5, IMU 2.5/5

### Failure mode
- Setup time 차이 <5min → 두 방식 모두 실용적. 하지만 **모의 환자 deploy 성공률**이 우리 주장의 핵심 — 그게 binary 차이면 OK.
- 모의 환자가 너무 협조적이라 IMU도 100% 성공 → 시나리오 더 엄격하게 (실제 stroke 환자 보고서 기반 standing 시간 제약).

### Time
- 1주

---

## Exp C — Pathological gait asymmetry detection

### Hypothesis
Vision은 좌우 step length / swing duration asymmetry를 5%+ 감지하고, IMU acceleration pattern은 <2% 감지에 그친다.

### Setup
- 피험자: N=4-6 healthy adults (자신의 대조군, within-subject crossover)
- 환경: 트레드밀 1.0 m/s, 각 조건 3분 × 2 반복
- Within-subject conditions:
  1. **Baseline**: 정상 보행
  2. **Ankle weight**: L-ankle에 1-2kg 추가 (swing phase 지연 + toe-drag)
  3. **Knee brace**: R-knee에 30° flexion 제한

### Measurements
- Vision: 좌우 ankle dorsiflexion peak angle (swing phase), step length (gait cycle 별)
- IMU: 좌우 foot pitch rate (accelerometer 미분), stride duration

### Primary metric
- **Gait Symmetry Index (GSI)** = |X_L − X_R| / ((X_L + X_R)/2) × 100 (%)
  - X = ankle peak angle (swing) or step length
  - 목표: Vision baseline → ankle-weight 후 GSI **>5% 증가**, IMU **<2% 증가**

### Secondary metrics
- Stance time asymmetry
- Cadence variability (CV)
- Step length asymmetry
- Sensitivity / specificity for asymmetry > 5% threshold

### Statistical analysis
- Paired t-test (baseline vs perturbed, each modality)
- Effect size (Cohen's d)
- ROC curve for asymmetry detection threshold

### Predicted result
- Vision: GSI 5-15% 변화 detect (ankle position 직접 측정)
- IMU: GSI 1-3% 변화 detect (acceleration 적분 + frequency-domain만)
- Vision detection sensitivity > 2× IMU

### Failure mode
- 둘 다 asymmetry 감지 못함 → weight 증량 (2-3kg) or knee 제한 더 엄격 (15°). 재실험.
- IMU도 vision 수준으로 감지 → 우리 차별점 약화. 단 vision의 **walker-frame 기준 absolute step length** vs IMU의 **상대 acceleration pattern**의 임상 해석 차이는 여전.

### Time
- 1주

---

## Out-of-MPU (Paper 2 후보)

### Exp D — Lookahead gain scheduling
- 2026-05-04 결정으로 **MPU 제외**
- Steelman 분석: 시스템 total latency 30-40ms vs 예측 horizon 50-100ms → 실제 여유 10-20ms뿐. Predictive 효과 작거나 null.
- Paper 2에서 RL policy + sim-to-real 영역으로 흡수

### Exp E — Stroke patient pilot
- IRB 필수. Approval 2-3개월 → 실험 2-3개월 = Paper 2 timeline
- Setup time + walking endurance + gait quality with vision-based impedance
- 진짜 "vision이 임상 적용 가능"을 증명하는 실험

---

## 종합 Timeline (3개월 MPU)

```
Week 1-2:  EBIMU 6개 확보 결정 (구입 / 협력 lab 빌리기 / 단일 baseline)
           Load cell 부착, wheel encoder 확인
Week 3:    Exp B 실행 (setup time, hardware 부담 적음) — 1주
Week 4-5:  Exp A 데이터 수집 (N=4-6, 트레드밀 30분 × 2 session) — 2주
Week 6:    Exp C 데이터 수집 — 1주
Week 7-8:  분석 + 그래프 + 통계
Week 9-12: Paper 1 manuscript 작성 (RA-L target)
Week 13:   Submit
```

→ 약 3개월 후 RA-L 제출 가능

## Verification

다음 5개 모두 ✅이면 MPU 성공 → Paper 1 manuscript 진입:

- [ ] Exp A: Vision drift <2cm vs IMU >5cm, p<0.05
- [ ] Exp B: Setup time Vision <5min vs IMU >15min, 모의 환자 deploy 성공률 차이 명확
- [ ] Exp C: Vision GSI sensitivity >2× IMU
- [ ] EBIMU 또는 대체 baseline hardware 확보 완료
- [ ] IRB approval (expedited) 완료

---

## 변경 이력
- 2026-05-04: 초안. Plan `scalable-splashing-puffin.md`의 결정 반영. Exp D는 Paper 2로 미룸.
