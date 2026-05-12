# Plan D Phase 1 — Self-review checklist

**작성**: 2026-05-12. Python prototype Phase 1 (utils + L1 + CycleTemplate + PhaseEstimator).

## Test summary

| Suite | Count | Status |
|---|---|---|
| test_plan_d_utils | 29 | PASS |
| test_plan_d_l1 | 26 | PASS |
| test_plan_d_cycle_template | 22 | PASS |
| test_plan_d_phase_estimator | 13 | PASS |
| **Total** | **90** | **PASS** |

기존 Mac POSIX SHM tests 16개는 Mac 환경 limitation (Jetson 에서만 PASS).

## Self-review checklist (14 items)

| # | 항목 | 상태 | 검증 |
|---|---|---|---|
| 1 | Joseph form P update (positive definite) | ✓ | test_joseph_update_keeps_psd_random, test_l1_update_keeps_P_psd |
| 2 | Phase wrap in innovation [-π, π] | ✓ | test_wrap_to_pi_innovation_use_case, test_l1_update_innovation_phase_wrap |
| 3 | bin_of_phase roundoff-safe | ✓ | test_bin_of_phase_roundoff_safe_at_all_centers (regression) |
| 4 | dt validation (negative/NaN/huge) | ✓ | test_validate_dt_*, test_l1_predict_*_skipped |
| 5 | Initial P large (cold-start uncertainty) | ✓ | _DEFAULT_INITIAL_P_PHI = π², test_l1_init_P_psd |
| 6 | Process noise discretization (Q × dt) | ✓ | ekf_l1.py:158 (Q_d = Q_continuous × dt) |
| 7 | Float64 throughout | ✓ | test_l1_init_dtype_float64, test_joseph_update_dtype_float64 |
| 8 | NaN/inf guards (q with NaN, z=NaN) | ✓ | test_l1_update_nan_observation_skip, test_template_update_nan_q_partial |
| 9 | Cubic Hermite C¹ continuity | ✓ | test_template_lookup_smoothness (max 2nd diff < 0.5) |
| 10 | Cross-correlation Mahalanobis weighting | ✓ | test_estimator_per_joint_sigma_weighting |
| 11 | β clamped to clinical [0.03, 0.10] | ✓ | test_template_beta_clamped_to_clinical |
| 12 | Defensive copy (no external mutation) | ✓ | test_template_mu_is_defensive_copy |
| 13 | Public API ≤ 5 methods per class | ✓ | EKFL1: predict, update, predict_ahead, reset, condition_number_P (5) |
| 14 | Condition number sanity (P numerical health) | ✓ | test_l1_condition_number_finite (< 1e10 after 500 updates) |

## 진정 발견된 bugs + fix

### Bug 1: bin_of_phase float roundoff (critical)

**증상**: `bin_i × 2π/128 × 128 / 2π = 21.999999999999996` (float). `int()` → bin 21 대신 bin 22 진입 의무.
**영향**: CycleTemplate update 시 bin_i ≥ 22 의 phi 가 bin_i - 1 에 mapped → μ[bin_i] 가 *0* 으로 영구 남음. 이게 lookup_jacobian 의 non-zero result 의 원인.
**Fix**: `int(math.floor(idx_float + 1e-9))` (1e-9 bias before floor).
**Regression test**: test_bin_of_phase_roundoff_safe_at_all_centers — 모든 128 bin 의 center 가 자기 bin 에 mapped.

이 bug 가 C++ port 에서도 동일 — *Eigen / std::floor 도 float roundoff* 동일 영향. *paper 의 implementation detail* 의 의무 기록.

## API surface

### `utils.py` (public)
- `TWO_PI: float`
- `wrap_to_pi(angle) -> angle`
- `wrap_to_2pi(angle) -> angle`
- `validate_dt(dt, max_dt_s=0.5) -> bool`
- `joseph_update(P, K, H, R) -> P_post`
- `bin_of_phase(phi, n_bins) -> int`

### `ekf_l1.EKFL1`
- `__init__(process_noise_phi, process_noise_omega, measurement_noise, initial_omega, ...)`
- `predict(t_now)` — time-update
- `update(z_phi, R_override=None)` — measurement-update
- `predict_ahead(tau_s) -> (phi, sigma_phi, omega, sigma_omega)` — forecast
- `reset(initial_omega=None)` — fallback re-init
- `condition_number_P() -> float` — diagnostic

### `cycle_template.CycleTemplate`
- `__init__(n_bins=128, n_joints=6, beta_default=0.05)`
- `update(phi, q, beta=None)`
- `lookup(phi) -> (n_joints,)` — cubic Hermite
- `lookup_jacobian(phi, eps=1e-3) -> (n_joints,)`
- `is_initialized`, `touched_fraction`, `total_updates`, `mu`, `reset()`

### `phase_estimator.CrossCorrPhaseEstimator`
- `__init__(template, min_touched_fraction=0.25)`
- `estimate(q, sigma_per_joint=None) -> PhaseEstimate`

## Real-time safety (C++ port readiness)

| 함수 | 복잡도 | 할당 | RT-safe |
|---|---|---|---|
| EKFL1.predict | O(1) | F, P_pred (2x2) | ✓ (stack-allocatable in C++) |
| EKFL1.update | O(1) | K, P_post (2x1, 2x2) | ✓ |
| EKFL1.predict_ahead | O(1) | F_tau (2x2) | ✓ pure read-only |
| CycleTemplate.update | O(n_joints) | none (in-place) | ✓ |
| CycleTemplate.lookup | O(n_joints) | result vector | ✓ (preallocate in C++) |
| CrossCorrPhaseEstimator.estimate | O(n_bins × n_joints) | diff, costs arrays | ⚠ Python (vectorized) — C++ should preallocate or use Eigen |

128 × 6 = 768 operations / call. Sub-µs on Jetson Orin.

## Codex consult prompt (다음 turn 에 사용)

```
[Codex Plan D Phase 1 Review — 90 tests PASS]

검토 항목:

1. 수치 안정성 (utils.py, ekf_l1.py):
   - Joseph form P update — 정확 구현?
   - bin_of_phase 의 1e-9 bias before floor — 다른 roundoff edge case 있나?
   - Cubic Hermite (Catmull-Rom) — 4-bin window 부족? Bezier 또는 natural cubic spline 의 trade-off?

2. EKF L1 default parameters:
   - process_noise_omega = 4e-2 ((rad/s)²/s) — 적정?
     (clinical gait: ω ranges 2.0-7.0 rad/s, 0.32-1.1 Hz stride)
   - measurement_noise = 0.05 rad² (σ ≈ 13°) — template match quality 의 진정 σ?
   - max_dt_s = 0.5s — 2 strides 위 — 적정 threshold?

3. CycleTemplate:
   - n_bins = 128 — 2.8° resolution OK for HS timing target ≤30ms p95?
   - β clamp [0.03, 0.10] — 너무 narrow? clinical β_min 의 진정?
   - NaN-aware per-joint update — μ 의 stale joint 가 drift 가능?

4. CrossCorrPhaseEstimator:
   - min_touched_fraction = 0.25 — cold-start 의 진정 threshold?
   - Confidence = best/second_best ratio — 더 robust metric 의무?
     (예: best vs valley depth, 또는 Mahalanobis χ² test)
   - Parabola subpixel — 3-point 만 — 5-point 또는 Gaussian fit 우월?

5. C++ port concerns:
   - Eigen vs raw arrays — *real-time deterministic*?
   - μ ∈ ℝ^(128, 6) — row-major 또는 column-major?
   - Hot path 의 dynamic allocation — 없음 확인?

6. Test coverage gaps:
   - Pathological gait test (freezing ω→0, shuffling small ω, asymmetric L/R) 의무?
   - L/R 분리 EKF instances 의무?
   - 환자 IMU 부재 시 *world frame transform 부재* — L1 이 cam frame OK?

7. Phase 2 priority:
   - L2 (3-state, const accel) — Q matrix 의 진정 discretization?
   - L3 (template-driven EKF) — H = ∂μ/∂φ * [1, 0, 0] — 수치 안정성?
   - Cascade activation criteria (3 strides, residual RMS < 15%, CV ω < 10%) — TDD 의 진정 test?
   - Divergence detection threshold (3σ innovation) — Mahalanobis 또는 단순 residual?

8. Clinical safety:
   - Cold-start (template empty): L1 의 phase observation 의 source 의무?
     (template 없으면 cross-correlation 불가 — *external phase signal* 의무?
      또는 *vertical hip oscillation* 의 *Hilbert envelope* 부분 사용 가능?)
   - 환자 freezing 시 L1 의 ω→0 — innovation gate 가 작동?
   - Watchdog: vision loss > 50ms → L1 hold-last → pretension 5N 의 *latency*?

Output: 항목별 PASS / NEEDS_FIX / IMPROVE + 구체적 file:line 행동.
```

## Phase 2 plan (다음 turn)

| 모듈 | 의무 | tests |
|---|---|---|
| ekf_l2.py | 3-state (φ, ω, α), F = [[1, dt, dt²/2], [0, 1, dt], [0, 0, 1]] | ~15 tests |
| ekf_l3.py | Template-driven EKF update via μ(φ) Jacobian | ~12 tests |
| cascade.py | L1→L2→L3 activation + fallback chain | ~10 tests |
| divergence.py | Innovation gate, template residual χ² test | ~8 tests |
| predictor.py | Top-level facade, predict_ahead(τ) integration | ~10 tests |

## Phase 3 plan (Phase 2 후)

| 모듈 | 의무 |
|---|---|
| End-to-end synthetic gait test | 환자 sinusoidal + noise → predictor → HS detection p95 |
| Stride detector | HS event (φ wrap 0/π) → cycle count |
| SHM v2 reader (Python) | 검증용 단순 reader |
| Integration test | SHM v2 publisher + reader + predictor 1000 frames |

진정 — Phase 1 의 90 tests PASS 가 Phase 2 의 foundation. 정확 구현.
