# Codex Review — Plan D Phase 1 Python Prototype

**작성**: 2026-05-12 (commit fb7e1b5d 후).
**Codex CLI**: v0.128.0, gpt-5.5, medium reasoning, web_search_cached.
**Tokens**: 98,516.
**Verdict**: *"not yet an algorithm validation reference for control — useful Phase 1 math sandbox, but ... must be fixed before the C++ port treats it as authoritative."*

## Hard Wall (Phase 2 진입 전 fix 의무)

**Circular bootstrap problem**:
- L1/L2 need `phi_meas` (phase observation)
- `CrossCorrPhaseEstimator` cannot produce `phi_meas` until template is touched
- `CycleTemplate.update()` needs `phi`

References: `phase_estimator.py:100`, `cycle_template.py:72`, `plan_d_phase2_design.md:116-121`.

**Action**: define external cold-start phase source before implementing cascade.
Codex options:
- Heel/hip vertical oscillator (Hilbert envelope) ← 선택
- Event detector (heel strike)
- Manual calibration stride
- Logged phase labels

→ **Plan D Phase 1.5 의 새 module**: `hilbert_phase.py` (hip vertical Hilbert envelope).

## 9 NEEDS_FIX (critical)

| # | File:Line | 문제 | Fix |
|---|---|---|---|
| 1 | `ekf_l1.py:166` | Q discretization 잘못 — `Q_c × dt` ignores integrated noise on φ from ω | `q_ω × [[dt³/3, dt²/2], [dt²/2, dt]]` + `q_φ × dt × [[1,0],[0,0]]` |
| 2 | `ekf_l1.py:187` | R_override unchecked — negative or NaN R poisons Joseph | Reject non-finite or ≤0 |
| 3 | `ekf_l1.py:54` | Comment 잘못: `σ ~ π (1 rad)` 단 P=π² 의미 σ=π rad | Fix comment + reduce initial P (or first-update gate) |
| 4 | `cycle_template.py:31` | β=[0.03,0.10] vs spec "3-5 strides" — 실제 10-33 strides | Reconcile (use β scheduling or update doc) |
| 5 | `cycle_template.py:68,102` | `_touched` per-bin not per-joint — 1 joint valid → bin marked all-valid | Reshape to (n_bins, n_joints) |
| 6 | `cycle_template.py:98-101` | First update: μ=0 → β×q (amplitude bias low) | Init untouched joints directly to q |
| 7 | `predictor_spec.md:91` impl | L/R separation 누락 — hemiparetic asymmetric gait 불가 | Two CycleTemplate instances (one per leg) at PredictorCascade level |
| 8 | `phase_estimator.py:40` | Confidence semantics inverted: code "lower=good" vs spec ">0.8 trigger=good" | Rename `confidence` → `ambiguity_ratio` (low=sharp) |
| 9 | `phase_estimator.py:129` | sigma floor 1e-6 → inv_var = 1e12 (numerical blowup) | Floor at realistic angle (e.g. 0.01 rad = 0.6°) |

## 5 IMPROVE (도움 됨)

- `utils.py:130` bin_of_phase 1e-9 bias → upper-boundary sliver to bin 0 (document or `np.nextafter` for C++)
- `ekf_l1.py:109` measurement_noise=0.05 guess — derive from template-match cost curvature or stereo replay residuals
- `ekf_l1.py:108` process_noise_omega=4e-2 too stiff for start/stop — adapt by gait state
- `ekf_l1.py:51` initial_omega=4.0 too slow for healthy gait — expose as session calibration
- `ekf_l1.py:113` max_dt_s=0.5 silent skip hides stale vision — return status + trigger watchdog
- `cycle_template.py:52` n_bins=128 ≈ 7.8ms at 1 Hz — validate with HS event error, not phase-bin math
- `cycle_template.py:133` Catmull-Rom with untouched neighbors → zeros — interpolate only valid per-joint bins
- `phase_estimator.py:63` min_touched_fraction=0.25 too permissive for control — require 3 strides + coverage
- `phase_estimator.py:152` parabola subpixel OK for prototype, test asymmetric templates before changing
- `phase_estimator.py:136-140` per-call allocations of `mu`, `diff`, `costs`, `mask` — C++ port must preallocate
- `cycle_template.py:67` row-major NumPy vs column-major Eigen — explicit `RowMajor` or flat `[bin][joint]`

## Phase 2 Risks (design adjustment)

- ✓ PASS `phase2_design.md:26-32`: analytical L2 Q correct — do NOT use van Loan in hot path
- IMPROVE `cycle_template.py:152`: central-diff `eps=1e-3` OK for 128 bins, but for L3 use analytic Catmull-Rom derivative (avoid 2 lookups per call)
- NEEDS_FIX `phase2_design.md:130-135`: stride detection belongs **inside cascade state** with gated phase + event confirmation. External stride counting desyncs fallback logic.
- NEEDS_FIX `phase2_design.md:140-147`: chi² thresholds by measurement DOF (not hardcoded 9.0), LDLT/LLT solve (not explicit `np.linalg.inv`)
- IMPROVE `phase2_design.md:87-90`: post-update template-update can self-reinforce wrong corrections — only update after innovation gate passes, with bounded β + residual logging

## 10 Coverage Gaps

NEEDS_FIX tests:
1. Empty-template cold start (phi observation source)
2. Omega near zero / freezing gait
3. Stop-start (cadence transients)
4. Asymmetric L/R phase (hemiparetic simulation)
5. Sparse joint masks (most NaN per frame)
6. Negative/zero R (defensive)
7. Exact L1 Q discretization (compare integrated vs naive)
8. Long dt gap status (return status, not silent skip)
9. Confidence/ambiguity semantic use (post-rename)
10. C++-equivalent no-allocation loops (preallocation pattern)

## ✓ PASS

- `utils.py:73` Joseph form + symmetrization correct
- `phase2_design.md:26-32` analytical L2 Q is the embedded RT choice

## Clinical bottom line (Codex 직접 인용)

> "this is not yet an algorithm validation reference for control. It is a useful Phase 1 math sandbox, but the cold-start source, per-joint template validity, confidence semantics, and watchdog/fallback status outputs must be fixed before the C++ port treats it as authoritative."

## Phase 1.5 implementation plan (사용자 결정 A — 전체 fix wave)

| Wave | 내용 | Effort |
|---|---|---|
| 1 | ekf_l1 fixes (Q disc + R val + initial_P + comment + status return) | ~45min |
| 2 | phase_estimator fixes (ambiguity_ratio rename + sigma floor) | ~30min |
| 3 | cycle_template redesign (per-joint touched + first-update init + β scheduling) | ~1.5hr |
| 4 | Hilbert envelope cold-start module (`hilbert_phase.py`) | ~1hr |
| 5 | 10 coverage tests | ~1hr |
| 6 | Phase 2 design doc adjust (stride inside cascade, χ² by DOF, LDLT) | ~30min |
| **Total** | | **~5hr** |

진정 — 진행. 작은 보폭, 정확.

## L/R separation 의 진정 location

Codex finding #7 = "L/R separation". 진정 location:
- **Phase 1 prototype**: CycleTemplate single instance OK (6 joints over single phase)
- **Phase 2 PredictorCascade**: TWO instances (one per leg, 3 joints per leg)
- → Phase 1.5 cycle_template redesign 의 *single template* 유지, *L/R 분리는 PredictorCascade 의 의무*

이 trade-off Codex 가 명시 X 단 *진정 정확*:
- single 6-joint template = healthy gait (asymmetry 작음) OK
- bilateral instances = hemiparetic 의무
- 두 implementation 모두 valid — PredictorCascade 에서 *configurable* (per-leg or single)

## 진정 Q discretization derivation

Continuous-time model:
```
x_dot = F_c x + G_c w,   w ~ N(0, q_omega)   (white noise on omega)
F_c = [[0, 1], [0, 0]]   (phi_dot = omega, omega_dot = 0)
G_c = [0, 1]^T            (noise drives omega only)
```

Discrete-time Q:
```
Q_d = ∫₀^Δt e^(F_c τ) G_c q_omega G_c^T e^(F_c τ)^T dτ
    = q_omega × ∫₀^Δt [[τ², τ], [τ, 1]] dτ
    = q_omega × [[Δt³/3, Δt²/2], [Δt²/2, Δt]]
```

Plus independent phi noise (if any, e.g., quantization):
```
Q_d_total = q_omega × [[Δt³/3, Δt²/2], [Δt²/2, Δt]] + q_phi × Δt × [[1, 0], [0, 0]]
```

진정 — 이 의무 correctly implement. *현재 코드*: `Q_continuous × dt` = `diag(q_phi×dt, q_omega×dt)` — off-diagonal coupling 누락.
