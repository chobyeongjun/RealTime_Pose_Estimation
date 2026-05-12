"""Plan D Cycle Template μ(φ) — recursive joint-angle template over gait phase.

μ ∈ ℝ^(n_bins × n_joints) with bins indexed by phase φ.
For walking, n_bins = 128 (Codex Q1 recommendation) gives ~2.8° resolution.

Update rule (per Codex Q1, plan_d_predictor_spec.md §2.4):
    μ_new[bin] = (1 - β) μ_old[bin] + β q

β bounded to clinical range [0.03, 0.10]:
    - β = 0.10: fast adapt, ~10-stride memory (cold-start)
    - β = 0.03: slow adapt, ~33-stride memory (steady state)

Lookup uses **cubic Hermite** interpolation between bin centers to provide
C¹ continuity (smooth derivatives for EKF measurement Jacobian).

Cold-start gating:
    - bin_touch_count[bin] tracks how many updates each bin has received
    - is_ready_for_l3 = (∑touched_bins / n_bins) ≥ 0.5 AND total_strides ≥ 3

Stride counting is external (caller wraps from φ_prev → φ_now around 0 or 2π).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from perception.plan_d_prototype.utils import TWO_PI, bin_of_phase, wrap_to_2pi

# Clinical β bounds (Codex Q1, plan_d_predictor_spec.md §2.4)
BETA_MIN: float = 0.03
BETA_MAX: float = 0.10


class CycleTemplate:
    """Recursive joint-angle template over gait phase.

    Public API:
        update(phi, q, beta=None) → None
        lookup(phi) → q_template (n_joints,)
        lookup_jacobian(phi) → dμ/dφ (n_joints,)
        is_initialized property (any bins touched)
        touched_fraction property [0, 1]
        reset() → None

    Storage: μ ∈ ℝ^(n_bins, n_joints) float64.
    """

    def __init__(
        self,
        n_bins: int = 128,
        n_joints: int = 6,
        beta_default: float = 0.05,
    ) -> None:
        if n_bins < 8:
            raise ValueError(f"n_bins must be ≥ 8 (got {n_bins})")
        if n_joints < 1:
            raise ValueError(f"n_joints must be ≥ 1 (got {n_joints})")
        if not (BETA_MIN <= beta_default <= BETA_MAX):
            raise ValueError(
                f"beta_default must be in [{BETA_MIN}, {BETA_MAX}], got {beta_default}"
            )
        self.n_bins = int(n_bins)
        self.n_joints = int(n_joints)
        self._beta_default = float(beta_default)
        self._mu = np.zeros((self.n_bins, self.n_joints), dtype=np.float64)
        self._touched = np.zeros(self.n_bins, dtype=np.int64)

    # ─── Update ──────────────────────────────────────────────────────────

    def update(
        self,
        phi: float,
        q: np.ndarray,
        beta: Optional[float] = None,
    ) -> None:
        """Recursive update at bin(φ).

        NaN entries in q skip that joint (preserve previous value).
        Out-of-bound q (inf) is also skipped.
        """
        if not math.isfinite(phi):
            return
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.n_joints,):
            raise ValueError(
                f"q shape {q_arr.shape} != ({self.n_joints},)"
            )
        b = float(self._beta_default if beta is None else beta)
        # Clamp β to clinical range (defensive — caller bug ≠ unsafe behavior)
        b = max(BETA_MIN, min(BETA_MAX, b))
        idx = bin_of_phase(phi, self.n_bins)
        valid = np.isfinite(q_arr)
        if not valid.any():
            return
        # Per-joint update, NaN-aware
        old = self._mu[idx]
        new = old.copy()
        new[valid] = (1.0 - b) * old[valid] + b * q_arr[valid]
        self._mu[idx] = new
        self._touched[idx] += 1

    # ─── Lookup (cubic Hermite over bins) ────────────────────────────────

    def lookup(self, phi: float) -> np.ndarray:
        """Interpolated template at φ. Returns (n_joints,) float64.

        Cubic Hermite over four adjacent bins (i-1, i, i+1, i+2) with
        phase wrap.  C¹ continuous; falls back to nearest-neighbor when
        the surrounding bins are untouched.
        """
        if not math.isfinite(phi):
            return np.full(self.n_joints, np.nan, dtype=np.float64)
        phi_w = float(wrap_to_2pi(phi))
        # Continuous bin position
        bin_pos = phi_w * self.n_bins / TWO_PI
        i = int(math.floor(bin_pos))
        t = bin_pos - i  # in [0, 1)
        # Indices with wrap
        i_m1 = (i - 1) % self.n_bins
        i0 = i % self.n_bins
        i1 = (i + 1) % self.n_bins
        i2 = (i + 2) % self.n_bins
        # If any of the central two bins are untouched, fall back to nearest
        # touched bin (avoid propagating zeros).
        if self._touched[i0] == 0 and self._touched[i1] == 0:
            # Search outward for nearest touched bin
            nearest = self._find_nearest_touched(i0)
            if nearest is None:
                return np.zeros(self.n_joints, dtype=np.float64)
            return self._mu[nearest].copy()
        # Cubic Hermite (Catmull-Rom): smooth between bin centers
        p0 = self._mu[i_m1]
        p1 = self._mu[i0]
        p2 = self._mu[i1]
        p3 = self._mu[i2]
        # Catmull-Rom basis
        t2 = t * t
        t3 = t2 * t
        result = (
            0.5
            * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
        )
        return result.astype(np.float64)

    def lookup_jacobian(self, phi: float, eps: float = 1e-3) -> np.ndarray:
        """∂μ/∂φ at φ (numerical central difference).

        Used as EKF measurement Jacobian: H_phi[k] = ∂μ_k/∂φ.
        Returns (n_joints,) float64.
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
        return float(self._touched.astype(bool).sum()) / self.n_bins

    @property
    def total_updates(self) -> int:
        return int(self._touched.sum())

    def reset(self) -> None:
        self._mu.fill(0.0)
        self._touched.fill(0)

    # Defensive read access (no external mutation)
    @property
    def mu(self) -> np.ndarray:
        return self._mu.copy()

    # ─── Internal helpers ────────────────────────────────────────────────

    def _find_nearest_touched(self, center: int) -> Optional[int]:
        """Find nearest bin (by phase distance) that has been updated."""
        if not self._touched.any():
            return None
        for offset in range(1, self.n_bins // 2 + 1):
            left = (center - offset) % self.n_bins
            right = (center + offset) % self.n_bins
            if self._touched[left] > 0:
                return left
            if self._touched[right] > 0:
                return right
        # Shouldn't reach here if any touched
        return int(np.argmax(self._touched))
