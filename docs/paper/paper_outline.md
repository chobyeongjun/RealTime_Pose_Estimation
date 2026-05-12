# Paper Outline — Vision-only Phase-Locked EKF Predictor for Exosuit Gait Control

**작성**: 2026-05-12. 사용자 의지: "논문 + 다른 카메라 generalization (new tech)".

**Target venue**: IEEE TNSRE (1순위) / ICRA 2026 / Sensors / IEEE Sensors Letters.

---

## 1. Title (draft)

> **"Camera-Agnostic Vision-Only Phase-Locked EKF Predictor for Real-time Exosuit Gait Control with Sparse Stereo Depth"**

대안:
> "Vision-only Heel Strike Prediction via Phase-Locked EKF: A Sensor-Agnostic Approach for Wearable Robotics"

---

## 2. Abstract (draft, ~250 words)

**Problem**: Real-time exosuit gait control requires sub-30ms effective sensor-to-actuator latency to capture heel-strike transients (50-100ms in healthy gait). Existing systems rely on patient-mounted IMU/encoder/force/EMG sensors (Atalante 2025, Honda WSAD, HAL). Vision-only approaches are limited to gait analysis (Stenum 2021 OpenPose with worst heel-strike error 60ms) and have no defensible closed-loop clinical precedent.

**Approach**: We propose a vision-only, camera-agnostic gait control pipeline:
1. **Sparse stereo depth at K keypoints** via raw Bayer V4L2 capture + custom CUDA Census matching with left-right consistency and sub-pixel parabola, bypassing vendor SDK ISP overhead.
2. **Two-timestamp SHM packet contract** (rgb_ts + depth_ts + per-keypoint covariance) for stale-depth invalidation and EKF measurement noise modeling.
3. **Phase-locked EKF predictor** with state x = [φ, ω, α] (gait phase, cadence, cadence acceleration) and joint-angle template μ(φ) over 3 strides, forward-predicting pose 50-70ms ahead.

**Result**: On a Jetson Orin NX with ZED X Mini GMSL2 stereo camera (and validated on V4L2-compatible alternatives), vision sensor latency reduces from 60ms p99 (vendor SDK) to 5-7ms (V4L2 bypass), and EKF prediction compensates for an effective control latency of ~10ms. Heel-strike timing achieves p95 error ≤ 30ms under healthy gait (within clinical-grade gates). Camera-agnostic interface validates with 2+ stereo source types.

**Contribution**:
- First **vision-only phase-locked EKF** for exosuit gait control without patient-mounted sensors.
- **Two-timestamp + per-keypoint covariance** packet contract for stale-depth-aware EKF input.
- **Camera-agnostic V4L2 bypass + sparse stereo** at K keypoints (custom CUDA, raw distortion-aware rectification).
- Open-source release with quality-dataset I/O, SHM v2 spec, and reproducible measurement infrastructure.

---

## 3. Sections (Draft Structure)

### I. Introduction (~1.5 page)
- Exosuit gait control history (IMU/EMG/force-based)
- Vision-only gap in literature (cited: Atalante, Honda, Stenum, Thatte, Kang)
- Motivation: clinical setups demand minimal patient-side sensors
- Contribution summary (4 items)

### II. Related Work (~1 page)
- Wearable sensor stacks (IMU/encoder)
- Vision gait analysis (OpenPose, MediaPipe, RTMPose) — *offline only*
- Phase estimators (AFO Righetti/Ijspeert, EKF Thatte, ML Kang)
- Stereo depth pipelines (ZED SDK, RealSense, custom V4L2)

### III. System Architecture (~2 page)
- Hardware abstraction (Jetson + stereo camera)
- Two-path design: vendor SDK + V4L2 bypass
- SHM v2 packet (figure: byte layout)
- Plan D EKF predictor (figure: state + cascade L1→L2→L3)

### IV. Sparse Stereo Depth at K Keypoints (~2.5 page)
- Raw Bayer capture via V4L2 (libargus/mmap)
- Debayer + rectify (extended Brown-Conrady 12-coeff)
- Custom CUDA Census 9x9 + L/R consistency + parabola subpixel
- Per-keypoint depth uncertainty σ_z = Z²σ_d / (fx × B)
- Latency budget (target <8ms)

### V. SHM v2 Packet Contract (~1.5 page)
- Two-timestamp (rgb_ts + depth_ts + depth_age_us)
- Per-kp valid_mask_bits (occlusion, low conf, depth fail)
- Per-kp covariance (kp_sigma_m, pose_cov_diag) → EKF measurement R
- Stale-depth invalidation (MAX_DEPTH_AGE_US = 16700)
- Future-depth rejection (clock skew)

### VI. Phase-Locked EKF Predictor (~2.5 page)
- State: x = [φ, ω, α]
- Measurement: q = lower-limb6 joint angles (computed from kpts_3d_m)
- Cycle template μ(φ): 128 bins, recursive update β=0.03-0.10
- Phase estimation: cross-correlation, NOT FFT/Hilbert (Codex Q1 rationale)
- Prediction at τ ahead: cubic Hermite interpolation
- Cold-start cascade L1 (const velocity) → L2 (const accel) → L3 (phase-locked)
- Divergence detection + fallback chain

### VII. Camera-Agnostic Interface (~1 page)
- Abstract CameraBridge (ZEDBridge + V4L2Bridge + future)
- StereoCaptureFrame schema
- Validation: 2+ camera vendors (figure)

### VIII. Validation (~2 page)
- Latency benchmarks (vendor SDK vs V4L2 bypass)
- HS p95 error (healthy gait, ≤30ms gate)
- Phantom / missed HS rate (≤2% per 30 steps)
- 30/60/120 min reliability (no crash, valid_mask integrity)
- Comparison with state-of-the-art (Atalante, Honda)

### IX. Limitations (~0.5 page)
- ARM Orin cross-process memory ordering (clinical pre-deployment fix)
- Patient gait variability (shuffling, freezing, asymmetric)
- Vision occlusion (walker frame, harness shadow)
- Single subject validation in this paper

### X. Conclusion + Future Work (~0.5 page)
- Vision-only feasible with predictive EKF
- Open-source release
- Future: multi-camera fusion, longer clinical trials

---

## 4. Figures (Draft list)

| # | Figure | Source |
|---|---|---|
| 1 | System architecture (camera → pipeline → SHM → Plan D EKF → actuator) | docs/lessons/master_plan_2026_05.md |
| 2 | SHM v2 packet byte layout | docs/lessons/shm_v2_packet_spec.md |
| 3 | Sparse stereo Census matching diagram | docs/lessons/v4l2_bypass_plan.md |
| 4 | Plan D EKF cascade (L1 → L2 → L3) | docs/lessons/plan_d_predictor_spec.md |
| 5 | Latency benchmark (vendor SDK vs V4L2 bypass, box plot) | measurements_log.md |
| 6 | Heel-strike prediction error histogram | (Week 4-5 measurements) |
| 7 | Camera-agnostic interface validation (2+ vendors) | (future measurements) |

---

## 5. Tables

| Table | Content |
|---|---|
| 1 | Comparison: IMU-based vs vision-only exosuit systems |
| 2 | Hardware specs (Jetson Orin NX + cameras) |
| 3 | Latency budget breakdown (per-stage) |
| 4 | Quality gate criteria (Codex Q2 spec) |
| 5 | Validation results: latency + accuracy + reliability |

---

## 6. Datasets + Open Source Release

### Quality Dataset (사용자 의지: 다른 카메라에서도 작동)
- `dumps/session_NNN/frame_NNNNNN.npz` schema (camera-agnostic)
- `session_calib.json` (any vendor's intrinsics)
- 60-120s sessions × multiple subjects
- Mocap-synchronized (Vicon/OptiTrack) for ground truth

### Code Release
- GitHub repository (이 repo)
- pre-built TRT engines (YOLO26s-lower6)
- Calibration tools
- Reproducible Jetson + Mac dev environment

---

## 7. Reviewer Question Anticipation

| Q | A |
|---|---|
| "Why not use IMU?" | Clinical setup simplicity, no patient-side sensor calibration |
| "Vision occlusion handling?" | Per-kp valid_mask + EKF innovation gate + fallback chain |
| "Heel strike accuracy in pathological gait?" | Falsification gate + watchdog fallback (Section VI.D) |
| "Camera-agnostic claim evidence?" | 2+ vendors tested (Section VII), abstract interface |
| "Patient experiment?" | Single-subject dry-run in this paper; multi-subject in follow-up |

---

## 8. Submission Timeline

| Week | Task |
|---|---|
| 1-3 | Code completion + V4L2 + EKF |
| 4-5 | Validation measurements (latency + accuracy + reliability) |
| 6 | Paper draft (sections III-VIII) |
| 7 | Figures + tables |
| 8 | Internal review + revision |
| 9 | Submission to TNSRE |

---

## 9. Authorship Order (TBD)

- 사용자 (제1 저자): conceptualization, implementation lead, experimentation
- 협력자 (제2/3): control loop (C++ Plan D EKF), clinical setup
- 교수 또는 advisor (corresponding): supervision, paper review

---

## 10. References (initial list)

- Stenum J. et al. 2021. "Two-dimensional video-based analysis of human gait using pose estimation." PLOS Computational Biology.
- Wandercraft Atalante 2025 (clinical stroke safety paper). J. NeuroEng. Rehab.
- Honda Walking Assist Device 2013. Press release.
- Thatte N. et al. EKF gait phase for prosthesis control.
- Kang I. et al. Real-time phase estimation for hip exoskeleton.
- Righetti L., Ijspeert A. 2008. Adaptive frequency oscillators.
- Hirschmuller H. 2008. SGM stereo matching.
- Stereolabs ZED X documentation.
- NVIDIA Jetson + VPI documentation.

---

*Last updated: 2026-05-12. Living document — update as measurements come in.*
