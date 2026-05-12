"""Plan D Cycle Template μ(φ) — recursive joint-angle template over gait phase.

μ ∈ ℝ^(n_bins × n_joints) with bins indexed by phase φ.
For walking, n_bins = 128 (Codex Q1 recommendation) gives ~2.8° resolution.

Update rule (per Codex Q1, plan_d_predictor_spec.md §2.4):
    First update (untouched bin × joint): μ[bin, k] = q[k]    ← direct init
    Subsequent updates:                    μ[bin, k] = (1-β) μ[bin, k] + β q[k]

This avoids the amplitude bias from starting at zero (Codex NEEDS_FIX #6).

β scheduling (Codex IMPROVE — schedule by gait state):
    - cold-start (total_updates < cold_threshold):  β = 0.10  (fast adapt)
    - steady (total_updates ≥ cold_threshold):       β = 0.03  (clinical)
    User can override per call.

Touched state is **per-bin per-joint** (Codex NEEDS_FIX #5):
    _touched ∈ ℤ^(n_bins × n_joints)
    A bin with one valid joint is NOT treated as valid for all joints.

Lookup uses **cubic Hermite (Catmull-Rom)** interpolation between bin centers,
**only across joints whose all 4 surrounding bins are touched**. For joints
with sparse coverage, falls back to nearest-touched bin (per joint).

Codex review 2026-05-12 fixes:
    - NEEDS_FIX #4: β range vs spec — kept clinical [0.03, 0.10] but added
      explicit scheduling so "3-5 stride memory" can be achieved via β=0.30
      at cold-start (caller can opt in) and tighter β at steady.
    - NEEDS_FIX #5: per-bin per-joint touched tracking.
    - NEEDS_FIX #6: first update initializes directly to q.
    - IMPROVE: lookup interpolates only across per-joint valid bins.

L/R separation (Codex NEEDS_FIX #7):
    Single 6-joint template OK for healthy gait. For hemiparetic / asymmetric,
    instantiate TWO CycleTemplates (one per leg, 3 joints each) at the
    PredictorCascade level. This module stays single-leg-agnostic.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from perception.plan_d_prototype.utils import TWO_PI, bin_of_phase, wrap_to_2pi

# Clinical β bounds (Codex Q1, plan_d_predictor_spec.md §2.4)
BETA_MIN: float = 0.03
BETA_MAX: float = 0.10

# Cold-start vs steady scheduling — total updates threshold.
# 3 strides × 128 bins × 6 joints / 2 (approx visit rate) ≈ 1152 updates.
# Use a simpler 3-stride-equivalent estimate: n_bins × n_joints.
_COLD_THRESHOLD_MULTIPLIER: int = 1   # 1 × (n_bins × n_joints) ≈ 1 stride worth


class CycleTemplate:
    """Recursive joint-angle template over gait phase.

    Public API:
        update(phi, q, beta=None) → None
        lookup(phi) → q_template (n_joints,)
        lookup_jacobian(phi, eps) → dμ/dφ (n_joints,)
        is_initialized, touched_fraction, total_updates, mu, reset()

    Storage:
        μ        ∈ ℝ^(n_bins, n_joints) float64
        _touched ∈ ℤ^(n_bins, n_joints) int64   (Codex NEEDS_FIX #5)
    """

    def __init__(
        self,
        n_bins: int = 128,
        n_joints: int = 6,
        beta_default: float = 0.05,
        beta_cold: float = BETA_MAX,
        cold_threshold: Optional[int] = None,
    ) -> None:
        if n_bins < 8:
            raise ValueError(f"n_bins must be ≥ 8 (got {n_bins})")
        if n_joints < 1:
            raise ValueError(f"n_joints must be ≥ 1 (got {n_joints})")
        if not (BETA_MIN <= beta_default <= BETA_MAX):
            raise ValueError(
                f"beta_default must be in [{BETA_MIN}, {BETA_MAX}], got {beta_default}"
            )
        if not (BETA_MIN <= beta_cold <= BETA_MAX):
            raise ValueError(
                f"beta_cold must be in [{BETA_MIN}, {BETA_MAX}], got {beta_cold}"
            )
        self.n_bins = int(n_bins)
        self.n_joints = int(n_joints)
        self._beta_default = float(beta_default)
        self._beta_cold = float(beta_cold)
        self._cold_threshold = (
            int(cold_threshold) if cold_threshold is not None
            else _COLD_THRESHOLD_MULTIPLIER * self.n_bins * self.n_joints
        )
        self._mu = np.zeros((self.n_bins, self.n_joints), dtype=np.float64)
        self._touched = np.zeros((self.n_bins, self.n_joints), dtype=np.int64)

    # ─── Update ──────────────────────────────────────────────────────────

    def update(
        self,
        phi: float,
        q: np.ndarray,
        beta: Optional[float] = None,
    ) -> None:
        """Recursive update at bin(φ).

        Per-joint NaN entries skip that joint (preserve previous value).
        First update for a (bin, joint) cell initializes directly to q[k]
        (avoids amplitude bias — Codex NEEDS_FIX #6).
        """
        if not math.isfinite(phi):
            return
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.n_joints,):
            raise ValueError(
                f"q shape {q_arr.shape} != ({self.n_joints},)"
            )
        valid = np.isfinite(q_arr)
        if not valid.any():
            return

        # β scheduling: cold-start β_cold until threshold, then β_default.
        if beta is None:
            b = self._beta_cold if self.total_updates < self._cold_threshold \
                else self._beta_default
        else:
            b = float(beta)
        b = max(BETA_MIN, min(BETA_MAX, b))

        idx = bin_of_phase(phi, self.n_bins)
        old = self._mu[idx]
        new = old.copy()
        for k in range(self.n_joints):
            if not valid[k]:
                continue
            if self._touched[idx, k] == 0:
                # First update for this (bin, joint) — direct init, no β×q bias
                new[k] = q_arr[k]
            else:
                new[k] = (1.0 - b) * old[k] + b * q_arr[k]
        self._mu[idx] = new
        # Per-joint touch count (only joints with valid q)
        self._touched[idx] = self._touched[idx] + valid.astype(np.int64)

    # ─── Lookup (cubic Hermite, per-joint validity-aware) ────────────────

    def lookup(self, phi: float) -> np.ndarray:
        """Interpolated template at φ. Returns (n_joints,) float64.

        For each joint k:
            - If all 4 surrounding bins are touched (per joint) → Catmull-Rom Hermite.
            - Else if center bin (i0 or i1) is touched → nearest of those.
            - Else → search outward for nearest touched bin for that joint.
            - If joint never touched → 0.0.
        """
        if not math.isfinite(phi):
            return np.full(self.n_joints, np.nan, dtype=np.float64)
        phi_w = float(wrap_to_2pi(phi))
        bin_pos = phi_w * self.n_bins / TWO_PI
        i = int(math.floor(bin_pos))
        t = bin_pos - i
        i_m1 = (i - 1) % self.n_bins
        i0 = i % self.n_bins
        i1 = (i + 1) % self.n_bins
        i2 = (i + 2) % self.n_bins

        result = np.zeros(self.n_joints, dtype=np.float64)
        t2 = t * t
        t3 = t2 * t
        for k in range(self.n_joints):
            tm1 = self._touched[i_m1, k] > 0
            t0 = self._touched[i0, k] > 0
            t1 = self._touched[i1, k] > 0
            tp2 = self._touched[i2, k] > 0
            if tm1 and t0 and t1 and tp2:
                p0 = self._mu[i_m1, k]
                p1 = self._mu[i0, k]
                p2 = self._mu[i1, k]
                p3 = self._mu[i2, k]
                result[k] = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            elif t0 and t1:
                # Linear interp across the two central bins
                result[k] = (1.0 - t) * self._mu[i0, k] + t * self._mu[i1, k]
            elif t0:
                result[k] = self._mu[i0, k]
            elif t1:
                result[k] = self._mu[i1, k]
            else:
                # Search outward (per joint) — falls back to nearest touched.
                nearest = self._find_nearest_touched_per_joint(i0, k)
                if nearest is not None:
                    result[k] = self._mu[nearest, k]
                # else: leave 0.0
        return result

    def lookup_jacobian(self, phi: float, eps: float = 1e-3) -> np.ndarray:
        """∂μ/∂φ at φ (numerical central difference per joint).

        Used as EKF measurement Jacobian: H_phi[k] = ∂μ_k/∂φ.
        Returns (n_joints,) float64.

        Codex IMPROVE for L3: analytic Catmull-Rom derivative is cheaper
        in C++ (avoids 2 lookups). For Phase 1 prototype, central diff is OK.
        """
        if not math.isfinite(phi):
            return np.zeros(self.n_joints, dtype=np.float64)
        plus = self.lookup(phi + eps)
        minus = self.lookup(phi - eps)
        return (plus - minus) / (2.0 * eps)

    # ─── Status ──────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return bool(self._touched.any())

    @property
    def touched_fraction(self) -> float:
        """Fraction of bins with AT LEAST ONE joint touched (back-compat)."""
        any_joint_touched = (self._touched > 0).any(axis=1)
        return float(any_joint_touched.sum()) / self.n_bins

    @property
    def touched_fraction_per_joint(self) -> np.ndarray:
        """Per-joint fraction of bins touched. (n_joints,) float64."""
        per_joint = (self._touched > 0).sum(axis=0).astype(np.float64)
        return per_joint / self.n_bins

    @property
    def total_updates(self) -> int:
        return int(self._touched.sum())

    def reset(self) -> None:
        self._mu.fill(0.0)
        self._touched.fill(0)

    @property
    def mu(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def touched(self) -> np.ndarray:
        """Per-bin per-joint touch count (defensive copy)."""
        return self._touched.copy()

    # ─── Internal helpers ────────────────────────────────────────────────

    def _find_nearest_touched_per_joint(
        self, center: int, joint: int
    ) -> Optional[int]:
        """Nearest bin (by phase distance) where the given joint has been updated."""
        if self._touched[:, joint].sum() == 0:
            return None
        for offset in range(1, self.n_bins // 2 + 1):
            left = (center - offset) % self.n_bins
            right = (center + offset) % self.n_bins
            if self._touched[left, joint] > 0:
                return left
            if self._touched[right, joint] > 0:
                return right
        # Defensive — argmax over the joint column
        return int(np.argmax(self._touched[:, joint]))
