# Plan D Phase 2 — Design notes (pre-Codex draft)

**작성**: 2026-05-12 (commit fb7e1b5 후). Codex consult 결과 받은 후 *수정 의무*.

## Module breakdown (5 + facade)

### 1. `ekf_l2.py` — Level 2 (constant acceleration)

State:
```
x = [φ, ω, α]ᵀ   ∈ ℝ³
P ∈ ℝ^(3×3)
```

Dynamics (constant-acceleration discrete):
```
φ_{k+1} = φ_k + ω_k Δt + ½ α_k Δt²   (mod 2π)
ω_{k+1} = ω_k + α_k Δt
α_{k+1} = α_k   (random walk driven by process noise)

F = [[1, Δt, Δt²/2],
     [0, 1,  Δt   ],
     [0, 0,  1    ]]
```

Process noise (continuous-time piecewise white acceleration):
```
Q_continuous = q_α × diag(0, 0, 1)        (only α driven by white noise)
Q_d = ∫₀^Δt F(τ) Q_c F(τ)ᵀ dτ            (exact integral)
    = q_α × [[Δt⁵/20, Δt⁴/8,  Δt³/6],
             [Δt⁴/8,  Δt³/3,  Δt²/2],
             [Δt³/6,  Δt²/2,  Δt   ]]
```

Or van Loan's method numerically (more robust to arbitrary continuous Q):
```
M = exp([-F  Q_c; 0  F^T] × Δt)
Q_d = F_d × M[upper-right]
```

Measurement (same as L1):
```
z = φ_observed
H = [1, 0, 0]
R = σ_z²  (from CrossCorrPhaseEstimator)
```

API mirrors EKFL1 (same surface):
- `predict(t_now)`
- `update(z_phi, R_override=None)`
- `predict_ahead(tau_s) → (phi, sigma_phi, omega, sigma_omega, alpha, sigma_alpha)`
- `reset(initial_omega=None, initial_alpha=0.0)`
- `from_l1(l1_state) → EKFL2`  ← cascade promotion (φ, ω 복사 + α=0, P expanded)

Activation criterion: ≥ 1 stride observed (Codex Q1: "after 1 stride enough samples to estimate α").

### 2. `ekf_l3.py` — Level 3 (phase-locked, template-driven)

State (same as L2):
```
x = [φ, ω, α]ᵀ
```

Measurement model changes:
```
z = q  ∈ ℝ^K              (full K-joint observation)
h(x) = μ(φ)  ∈ ℝ^K        (cycle template lookup)
H = ∂h/∂x = ∂μ/∂φ × [1, 0, 0]   ∈ ℝ^(K×3)
   where ∂μ/∂φ is from CycleTemplate.lookup_jacobian(φ)
R = diag(σ_per_joint²)  ∈ ℝ^(K×K)   (from SHM v2 kp_sigma_m)
```

Innovation:
```
ỹ = z - h(x_pred) = q - μ(φ_pred)
  (no phase wrap needed — measurement is in joint-angle space, not phase space)
```

Update (standard EKF, K-dim measurement):
```
S = H P H^T + R                ∈ ℝ^(K×K)
K_gain = P H^T S^{-1}          ∈ ℝ^(3×K)
x_new = x_pred + K_gain × ỹ
P_new = Joseph(P, K_gain, H, R)
```

Side effect (the trick): **also update template with current observation**:
```
μ.update(φ_post, q, β)   ← uses POST-update phase estimate
```

API:
- `__init__(template: CycleTemplate, ...)`
- `predict(t_now)`
- `update(q, sigma_per_joint) → innovation_chi2`   ← returns Mahalanobis innov² for divergence gate
- `predict_ahead(tau_s) → (phi, sigma_phi, ..., q_predicted_K)`   ← q_predicted = μ(φ_predicted)
- `from_l2(l2_state, template) → EKFL3`

Activation criterion: ≥ 3 strides + template.touched_fraction ≥ 0.5 + residual RMS < 15% + CV ω < 10%.

## Phase 2 design adjustments (Codex 2026-05-12 review)

The original design is annotated with the following adjustments before
implementation:

### Adjustment 1 — Q discretization (Phase 1.5 fix propagates)
EKF L2 must use the integrated 3-state form (NOT `Q_c × dt`):
```
Q_d = q_α × [[Δt⁵/20, Δt⁴/8,  Δt³/6],
             [Δt⁴/8,  Δt³/3,  Δt²/2],
             [Δt³/6,  Δt²/2,  Δt   ]]
```
This is the analytical integral for piecewise-white acceleration. Codex
explicitly PASSed this choice over van Loan numerical (which is non-
deterministic and slower in the hot path).

### Adjustment 2 — Stride detection INSIDE cascade (Codex NEEDS_FIX)
The original sketch had stride detection as an external helper. Codex flagged
this: external counters desync from fallback logic. Move stride detection
into `PredictorCascade` state, gated on (a) phase wrap event AND (b) HS event
confirmation (innovation residual minimum at expected φ_HS).

```python
class PredictorCascade:
    state: {
        ...
        prev_phi: float
        stride_count: int
        last_hs_t: float
    }

    def _detect_stride_event(self, phi_post_update):
        # Wrap detection
        wrapped = (self.state.prev_phi > 3π/2) and (phi_post_update < π/2)
        # Confirmation: innovation small near expected HS phase
        if wrapped and self._last_innov_chi2 < threshold:
            self.state.stride_count += 1
            self.state.last_hs_t = self.t_now
        self.state.prev_phi = phi_post_update
```

### Adjustment 3 — Chi² thresholds BY DOF (Codex NEEDS_FIX)
The original divergence gate hardcoded `threshold_chi2 = 9.0` (3σ in 3D).
Codex: thresholds must scale with measurement DOF.

```python
CHI2_THRESHOLDS = {
    1: 6.64,     # 99% confidence, 1-DOF (phase observation)
    3: 11.34,    # 99% confidence, 3-DOF (joint angles, 3 joints)
    6: 16.81,    # 99% confidence, 6-DOF (joint angles, 6 joints)
}

def innovation_gate(innov, S_inv_innov, dof: int) -> bool:
    chi2 = float(innov @ S_inv_innov)
    return chi2 > CHI2_THRESHOLDS[dof]
```

### Adjustment 4 — LDLT/LLT solve, NOT explicit inverse (Codex NEEDS_FIX)
Original sketch:
```python
K = P @ H.T @ np.linalg.inv(S)  # ← Codex: avoid explicit inverse
```
Correct: solve for `S^{-1} (H P) ` via Cholesky/LDLT:
```python
# Python prototype:
HPHt = H @ P @ H.T
# K = P H^T S^{-1}  ⟺  S K^T = (P H^T)^T  ⟺  S K^T = H P
K_T = np.linalg.solve(S, H @ P)
K = K_T.T
```
For C++: `Eigen::LDLT<MatrixXd> ldlt(S); K = ldlt.solve(H * P).transpose();`

### Adjustment 5 — Template post-update ONLY after innovation gate (Codex IMPROVE)
Original L3 step had:
```python
mu.update(phi_post, q, β)   # always — risk: self-reinforce wrong corrections
```
Correct: only update template after innovation gate passes.
```python
if innovation_gate(innov, S_inv_innov, dof):
    # Diverged — DO NOT update template, demote to L2/L1
    self._demote()
else:
    # Accepted — safe to update template
    mu.update(phi_post, q, β)
    self._log_residual(...)
```

### Adjustment 6 — Cold-start phase source (Hard Wall fix)
NEW prerequisite (Phase 1.5): cascade consumes `HipVerticalPhaseEstimator`
output as L1's phase observation until template is touched and stride
count ≥ 3.

```python
class PredictorCascade:
    def __init__(self):
        self.hilbert = HipVerticalPhaseEstimator(...)
        self.estimator = CrossCorrPhaseEstimator(template)
        ...

    def step(self, t_now, q, sigma_per_joint, z_hip_m):
        self.hilbert.feed(t_now, z_hip_m)
        # Phase observation source:
        if self.stride_count < 3 or self.template.touched_fraction < 0.5:
            # Cold-start path — use Hilbert envelope
            phase_obs = self.hilbert.estimate()
            if phase_obs.valid:
                self.l1.update(phase_obs.phi)
        else:
            # Steady path — use template cross-correlation
            est = self.estimator.estimate(q, sigma_per_joint)
            if est.valid and est.ambiguity_ratio < 0.5:
                if self.level == 3:
                    self.l3.update(q, sigma_per_joint)
                elif self.level == 2:
                    self.l2.update(est.phi)
                else:
                    self.l1.update(est.phi)
```

---

### 3. `cascade.py` — L1 → L2 → L3 activation + fallback chain

```
class PredictorCascade:
    state: {
        level: int (1, 2, 3)
        l1: EKFL1
        l2: Optional[EKFL2]
        l3: Optional[EKFL3]
        template: CycleTemplate
        estimator: CrossCorrPhaseEstimator
        stride_count: int
        cold_start_frames: int
    }

    def step(t_now, q, sigma_per_joint) -> StepResult:
        # 1. Predict active level
        # 2. Get phase observation from estimator (if template ready)
        # 3. Update active level (L3 uses q directly, L1/L2 use phi_est from estimator)
        # 4. Update template (only if not in fallback)
        # 5. Check activation criteria for promotion (L1→L2 after 1 stride, L2→L3 after 3)
        # 6. Check divergence for demotion (L3→L2 if chi² > threshold, etc.)
        # 7. Return state + diagnostics

    def predict_ahead(tau_s) -> Forecast:
        # Delegate to active level
        # Add σ_model² × τ² uncertainty inflation
```

Stride detection (within cascade or external?):
```
prev_phi → curr_phi: if curr < prev (with wrap allowance), stride++
```

Q (Codex 의무 검토): stride detection 의 진정 location — cascade 내부? external API call?

### 4. `divergence.py` — Innovation gate + template residual χ² test

```
def innovation_gate(innov, S, threshold_chi2=9.0) -> bool:
    """3σ Mahalanobis test: innov^T S^{-1} innov > threshold → REJECT."""
    return float(innov.T @ np.linalg.inv(S) @ innov) > threshold_chi2

def template_residual_chi2(q, mu_at_phi, sigma_per_joint) -> float:
    """For L3 divergence detection."""
    diff = q - mu_at_phi
    return float(np.sum((diff / sigma_per_joint) ** 2))

def cadence_jump_detector(omega_history, ratio_threshold=0.20) -> bool:
    """> 20% cadence change in 1 stride → divergence."""
```

### 5. `predictor.py` — Top-level facade

```
class PlanDPredictor:
    def __init__(self, n_joints=6, ...):
        self.cascade = PredictorCascade(...)

    def feed(t_now, q, sigma_per_joint) -> None:
        self.cascade.step(...)

    def forecast(tau_s) -> ForecastResult:
        return self.cascade.predict_ahead(tau_s)
        # ForecastResult: (phi, sigma_phi, omega, q_predicted, sigma_q_per_joint,
        #                  level_active, divergence_flags)
```

## Phase 2 test plan (~55 tests)

| Module | Tests | Focus |
|---|---|---|
| ekf_l2 | 16 | dynamics, F integration, Q discretization, cascade promotion from L1, predict_ahead 3-state |
| ekf_l3 | 14 | K-dim measurement, Jacobian from template, Joseph in 3D, template side-update |
| cascade | 12 | activation criteria, demotion, fallback chain, stride detection |
| divergence | 8 | innovation gate, template residual, cadence jump |
| predictor (e2e) | 8 | synthetic gait → HS prediction error p95 within 30ms |
| **Total** | **58** | |

## Open Q's for Codex (review will answer)

1. Q-discretization: van Loan numerical vs analytical integral — preference for embedded RT?
2. Template side-update at *post*-update φ vs *pre*-update — bias trade-off?
3. Stride detection: cascade-internal vs external user code?
4. Divergence threshold χ² = 9 (3σ in 3D)? Or per-DOF threshold?
5. L1 cold-start phase observation source — Hilbert envelope or hip vertical sinusoid?
6. β scheduling: start β=0.10 (fast adapt), decay to 0.03 after N strides — vs constant 0.05?
7. L/R: single template vs separate (asymmetric gait support)?

## Phase 3 (after Phase 2)

- End-to-end synthetic gait test (sinusoidal + noise + occlusion → HS detection)
- Stride detector module (HS event detection from φ wrap)
- SHM v2 Python reader (verification)
- Integration test (publisher + reader + predictor 1000 frames)
- Performance benchmark (per-step latency on Jetson, target < 100µs)

진정 — 이 design 이 Codex review 후 *어디 수정* 의무 — 결과 받기 의무.
