---
title: Why Vision in a Cable-Driven Walker Assist System
project: realtime-vision-control
purpose: Paper 1 introduction draft + 교수님 critique 1:1 대응
created: 2026-05-04
updated: 2026-05-04
status: draft (논문 introduction의 토대)
related_plan: ~/.claude/plans/scalable-splashing-puffin.md
---

# Vision의 필요성 — Cable-Driven Walker Assist System

## TL;DR — Bulletproof Core (한 문장)

> **Vision은 cable controller의 reference signal(walker-segment 절대 거리)을 stroke 환자에게도 deploy 가능하게 만드는 유일한 수단이다.**
> *Vision-based exocentric absolute distance feedback enables drift-free cable impedance control for stroke patients who cannot perform standing calibration — a clinical constraint that all patient-mounted inertial systems silently violate.*

이 한 문장이 Paper 1의 contribution 전체. 모든 실험과 주장이 여기로 수렴해야 함.

---

## Why this document exists

2026-05-04 미팅에서 교수님이 Paper 1의 가장 약한 곳을 정확히 찔렀다:

1. IMU를 segment에 부착해서 3 joint 운동을 파악하는 것과 차이가 뭔가? Vision 사용 정당성 약함.
2. Vision으로 walker-사람 거리 알고 slack 제어 정확하게 한다는데, 다른 방법으로도 가능하지 않나?
3. Vision이 walker assist에 왜 필요한지 + 특별한 장점 + 차별점 + 그 장점으로 만들 제어 결과의 구체적 주장이 필요하다.

이전 우리 주장 5개 중 3개는 reviewer가 깨뜨릴 수 있다 (특히 "predictive 50-100ms lookahead"는 시스템 total latency 30-40ms 앞에서 fantasy). 이 문서는 *깨지지 않는 핵심*만 남긴 정밀화된 논리.

---

## 핵심 framing: "Reference signal"

거리 그 자체가 중요한 것이 아니라, **거리가 cable controller의 reference signal**이라서 필수적이다.

Cable-driven walker의 물리 구조:

```
[walker frame] ─┬─ motor + 도르래 (anchor, 고정 위치)
                │
              (cable 본체)
                │
                └─ 환자 정강이 (cable end)

두 끝 사이의 거리 L = 매 순간 필요한 cable length
```

Cable controller는 매 순간 "지금 cable이 얼마여야 하나?"의 답이 필요하다. 그 답 = walker-정강이 거리.

이걸 모르면 controller 자체가 성립 안 함:

| 상황 | 결과 |
|---|---|
| Cable 너무 길 | slack(쳐짐) → force=0 → 환자 보조 0 |
| Cable 너무 짧 | 환자 다리 잡아당김 → 부상 |
| Cable 적정인데 환자 움직임과 비동기 | jerk → 환자 불쾌, 보행 disrupt |

**Cable length는 측정 불가능한 옵션이 아니라 controller 생존의 필수 reference signal.**

진짜 질문은 "그 거리를 어떻게 측정할 것인가" — 이 답이 vision의 존재 이유.

---

## 1:1 답변 — 교수님 3 critique

### Critique 1 — "IMU on segment로 3 joint 운동 측정 가능한데 vision 왜?"

**측정 영역이 다르다.**

- IMU on segment → joint angle (한 segment 안의 운동학)
- Vision → walker-frame absolute position of joint center (segment 과 walker 사이의 운동학)

Cable-driven walker assist의 본질은 "환자 다리가 walker에 대해 어디 있는가". Cable은 walker frame에서 출발해 환자 segment에 도달한다. 이건 **between-segment-and-external-anchor** 문제이지 within-segment 문제가 아니다.

IMU 단독으로는 segment-to-walker 위치를 줄 수 없다. 추가 anchor (UWB, ultrasonic, vision)가 필요하다. "Walker-mounted IMU + ultrasonic" 대안의 약점:
- Ultrasonic = closest surface까지의 scalar 거리. Cable이 attach되는 정확한 anatomical landmark (예: 정강이 중앙) 위치 모름.
- 결국 anatomical landmark precision이 필요한 순간, vision이 unique.

### Critique 2 — "Walker-사람 거리 다른 방법으로도 가능?"

**거리 scalar는 다른 방법으로 가능하지만, cable slack control은 scalar 거리 문제가 아니다.**

Cable slack의 정확한 결정 요소:
1. 정강이 3D 위치 (cable이 attach된 지점)
2. Walker frame에서 cable origin 위치 (고정)
3. 케이블 routing geometry (pulleys)
4. 케이블 elasticity model

Slack을 정밀하게 제거하려면 다음 4가지가 동시에 필요:

| 대안 | Per-body-part? | Walker frame? | Drift-free? | 환자 instrumentation 0? |
|---|---|---|---|---|
| Walker IMU + Ultrasonic | ❌ | ✅ | ✅ | ✅ |
| 환자 IMU + Walker IMU fusion | ✅ | △ (drift) | ❌ | ❌ |
| UWB tag on patient | △ (~10cm) | ✅ | ✅ | ❌ |
| Cable encoder + force sensor | ❌ | △ | ✅ | ✅ |
| **Vision (ours)** | ✅ | ✅ | ✅ | ✅ |

**Vision만이 4가지 모두 동시 만족.**

### Critique 3 — "장점들로 어떤 제어 결과를 만들어낼 수 있나?"

측정 가능한 3가지 결과 (각각 vision이 unique하게 가능하게 하는 것):

#### 결과 1 — Driftless cable length tracking
- Mechanism: `L_target(t) = ‖walker_origin − ankle_3D_vision(t)‖ + offset(gait_phase)`. Vision frame 기준 → drift 없음.
- Metric: 30분 보행 후 L_target과 실제 cable length(motor encoder)의 RMSE.
- Predicted: Vision <2cm RMSE, IMU+encoder >5cm.
- Vision unique 이유: standing reference 없이도 absolute distance feedback 가능.

#### 결과 2 — Stroke patient adaptation without calibration
- Mechanism: 환자가 walker 잡자마자 vision tracking 시작. Setup time 측정.
- Metric: Setup time. Vision ~3min vs IMU 6개 ~15-20min.
- **Stroke 환자에 대해 IMU-based calibration은 standing balance risk라서 임상 deploy 자체가 불가능**. Vision은 가능.
- Vision unique 이유: exocentric + zero patient instrumentation.

#### 결과 3 — Pathological gait asymmetry detection
- Mechanism: L/R hip-knee-ankle 3D position에서 좌우 step length / swing duration 직접 측정.
- Metric: Gait Symmetry Index. Ankle weight 1-2kg + knee brace로 asymmetry 유발 후 detection sensitivity.
- Predicted: Vision >5% GSI 변화 detect, IMU acceleration pattern <2%.
- Vision unique 이유: walker frame 기준 absolute step length 직접 측정 (IMU는 적분 + drift).

---

## 폐기된 주장 (reviewer가 깨뜨릴 수 있는 것)

향후 발표/문서에서 다음 주장은 **사용 금지** (이미 옛 자료에 들어가 있다면 정정):

| 폐기 주장 | Reviewer 반박 |
|---|---|
| "Anatomical landmark (no STA)" | Cappozzo functional calibration (2005~)이 4-6mm로 STA 보정. 20년 검증된 표준. |
| "Position absolute (no drift)" 단독 | 폐루프 control은 drift에 robust. 다만 **stroke 환자가 reference 못 잡는다**는 임상 제약과 결합되어야 함. |
| "Zero patient instrumentation" 단독 | Walker-mounted IMU도 환자 contact 없음. 다만 **시간 + comfort + clinical** 결합으로 의미. |
| **"Predictive slack 50-100ms lookahead"** | **시스템 total latency 30-40ms (Vision 14ms + 모터 응답 15ms) → 예측 horizon에서 실제 여유 10-20ms. Fantasy. 폐기.** |
| "Gait phase + asymmetry from kinematics" 단독 | IMU-based phase detection이 96% 정확. 차별화 아니라 책임. **Walker-frame absolute position과 결합되어야 unique.** |

이 주장들을 단독으로 발표하면 reviewer가 1:1 매칭으로 깨뜨린다. "Driftless reference signal + clinical constraint" 한 문장으로 묶어야 살아남음.

---

## Q-rebuttals (예상되는 follow-up)

### Q1: "Cable encoder + force sensor만으로 closed-loop 제어 가능하지 않나?"

가능하지만 두 가지가 빠짐:
1. **Reference (desired length) 어떻게 정하나?** Encoder는 motor가 명령한 length만 안다. Force는 0이 아니어야 한다는 정도만 안다. 환자 segment가 walker로부터 얼마나 떨어져 있는지 = vision이 줘야 함.
2. **Slack과 over-tension 둘 다 같은 force=0 region에 있다.** Force만으론 둘을 구별 못 함. Vision은 직접 봄.

### Q2: "그래서 walker-사람 거리 재는 게 그렇게 중요한가?"

거리 자체가 중요한 게 아니라 cable controller가 그 거리를 reference signal로 쓰기 때문. 위 "핵심 framing" 섹션 참조.

### Q3: "Camera occlusion / lighting 노이즈가 cable 제어를 불안정하게 만들지 않나?"

이건 valid한 우려이고 우리 7-layer 안전 chain이 답:
- Python e2e > 20ms → valid=False
- Bone length / velocity constraint → outlier reject
- Sticky publish max 5 frames (60ms) → 짧은 detection 손실 흡수
- C++ stale 0.2s → pretension 5N fallback
- 5중 force clamp (max 70N)

즉 vision이 잠깐 못 봐도 cable이 안전한 default로 fallback. 7-layer가 "vision이 sometimes fail"이라는 문제의 답. 이건 Paper 1의 부수적 contribution이지만 reviewer 안심을 위해 명시.

### Q4: "Stroke 환자 IRB 없이 어떻게 임상 deploy 주장하나?"

Paper 1은 healthy subject + simulated patient (Exp B에서 walker 잡고 일어서기 어려운 척하는 모의 환자)로 evidence 제시. 정식 stroke patient 임상은 Paper 2 (IRB approval 후). Paper 1의 임상 주장은 **"deployable" = 시스템이 임상 환경에서 작동 가능한 형태로 설계됨**으로 한정. **"clinically validated"** 주장은 Paper 2 영역.

---

## 향후 검증 (Paper 1 MPU)

3개 실험으로 위 3가지 결과를 측정 가능하게 입증. 자세한 protocol: `docs/experiments/2026-05-imu-vs-vision-mpu.md`.

- **Exp A** — Driftless cable length tracking (30분 drift 측정) → 결과 1 검증
- **Exp B** — Setup time + comfort → 결과 2 검증
- **Exp C** — Asymmetry detection → 결과 3 검증

Exp D (lookahead gain scheduling)는 위 "폐기된 주장" 표의 predictive 카테고리이므로 **Paper 1 MPU에서 명시적 제외**. Paper 2 또는 future work 후보.

---

## 인용 후보 (Related Work)

향후 논문 작성 시 다음을 인용해 우리 주장 위치 잡기:

- **Cappozzo et al. (2005)** — STA + functional calibration의 표준. 우리는 vision으로 이 calibration 자체를 우회.
- **Taborri et al. (2016, IEEE Sensors)** — IMU-based gait phase 96% accuracy. 우리 차별화는 phase detection 자체가 아니라 walker-frame absolute reference 결합.
- **Kanko et al. (2021), Uhlrich et al. (2022)** — Markerless depth ±2-4° vs marker-based mocap. 우리 vision pose accuracy 근거.
- **Patel (2009)** — Exocentric / walker-mounted sensing. 우리는 이 방향의 cable-driven 적용.
- **Kim et al. (2024)** — Adaptive stiffness model + body movement (이미 vault에 있음). Cable assist의 stiffness gain scheduling 참고.

---

## 변경 이력

- 2026-05-04: 초안 작성. Plan `scalable-splashing-puffin.md`의 결정 반영. "Predictive" 주장 폐기, "Driftless reference signal"로 framing 확정.
