"""Cross-correlation phase estimator — template matching for φ observation.

Given q (single K-joint observation) + template μ(φ), find φ minimizing
Mahalanobis distance:

    cost(φ_c) = Σ_k (q_k - μ_k(φ_c))² / σ_k²

Search:
    1. Evaluate cost at all n_bins bin centers (vectorized).
    2. Pick best bin.
    3. Parabola subpixel fit between best ± 1 bin (sub-bin precision).
    4. Compute **ambiguity_ratio = best/second_best** (low = sharp minimum).

Why cross-correlation (not FFT/Hilbert) — per Codex Q1, plan_d_predictor_spec.md §2.5:
    - Online single-sample observation, not windowed signal.
    - Robust to asymmetric/non-sinusoidal gait (pathological patients).
    - O(n_bins × n_joints) per call — cheap for n_bins=128, n_joints=6.

Codex review 2026-05-12 fixes:
    - NEEDS_FIX #8: confidence semantics renamed to `ambiguity_ratio` (was
      inverted vs predictor_spec.md:256 which uses confidence > 0.8 = good).
      ambiguity_ratio ∈ [0, 1], **low = sharp = high quality observation**.
    - NEEDS_FIX #9: sigma_per_joint floor lifted from 1e-6 (inv_var=1e12) to
      a realistic angle floor (default 0.01 rad ≈ 0.57°).
    - IMPROVE: min_touched_fraction default raised; clinical control should
      gate on 3 strides + coverage, not this fraction alone.

Returns PhaseEstimate(phi, cost, ambiguity_ratio, n_bins_searched, valid).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_2pi


# Realistic sigma floor — joints rarely measure tighter than ~0.5° = ~0.01 rad.
# Codex NEEDS_FIX #9: 1e-6 allowed inv_var = 1e12 which is numerical poison.
SIGMA_PER_JOINT_FLOOR: float = 0.01   # rad


@dataclass
class PhaseEstimate:
    """Result of cross-correlation phase observation.

    Attributes:
        phi: estimated phase (rad in [0, 2π)) — NaN if template not ready.
        cost: best Mahalanobis cost (lower is better).
        ambiguity_ratio: cost_min / cost_second_min, ∈ [0, 1].
            **LOW = sharp minimum = trustworthy observation.**
            >0.9 → ambiguous (multiple candidate minima, do not control on it).
        n_bins_searched: bins evaluated (for diagnostics).
        valid: convenience boolean (template ready AND finite output).
    """

    phi: float
    cost: float
    ambiguity_ratio: float
    n_bins_searched: int
    valid: bool


class CrossCorrPhaseEstimator:
    """Template-matching phase estimator.

    Public API:
        estimate(q, sigma_per_joint=None) → PhaseEstimate
    """

    def __init__(
        self,
        template: CycleTemplate,
        min_touched_fraction: float = 0.25,
        sigma_floor: float = SIGMA_PER_JOINT_FLOOR,
    ) -> None:
        """
        Args:
            template: CycleTemplate (reference, not copied — caller mutates).
            min_touched_fraction: diagnostic readiness gate; require at least
                this fraction of bins touched before estimator returns valid.
                Codex IMPROVE: clinical control gate (3 strides + coverage)
                should live in PredictorCascade, not here. Default 0.25 is
                kept for diagnostic / Phase 1 testing.
            sigma_floor: realistic per-joint σ floor (rad). Codex NEEDS_FIX #9.
        """
        if template.n_joints < 1:
            raise ValueError("template must have ≥ 1 joint")
        if sigma_floor <= 0.0:
            raise ValueError(f"sigma_floor must be > 0, got {sigma_floor}")
        self._template = template
        self._min_touched = float(min_touched_fraction)
        self._sigma_floor = float(sigma_floor)

    # ─── Estimate ────────────────────────────────────────────────────────

    def estimate(
        self,
        q: np.ndarray,
        sigma_per_joint: Optional[np.ndarray] = None,
    ) -> PhaseEstimate:
        """Find φ minimizing weighted squared distance ||q - μ(φ)||²_Σ.

        Args:
            q: (n_joints,) joint angle observation.
            sigma_per_joint: (n_joints,) per-joint σ. If None, σ=1 for all.
                Use depth-uncertainty-derived σ for clinically faithful R.
                Floored at sigma_floor (default 0.01 rad).

        Returns:
            PhaseEstimate with subpixel φ + ambiguity_ratio.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self._template.n_joints,):
            raise ValueError(
                f"q shape {q_arr.shape} != ({self._template.n_joints},)"
            )

        # Template readiness gate (diagnostic only — clinical caller gates upstream)
        if self._template.touched_fraction < self._min_touched:
            return PhaseEstimate(
                phi=float("nan"),
                cost=float("inf"),
                ambiguity_ratio=1.0,
                n_bins_searched=0,
                valid=False,
            )

        # NaN joints → exclude from cost (effective per-joint mask)
        valid_mask = np.isfinite(q_arr)
        if not valid_mask.any():
            return PhaseEstimate(
                phi=float("nan"),
                cost=float("inf"),
                ambiguity_ratio=1.0,
                n_bins_searched=0,
                valid=False,
            )

        # Inverse variances with realistic sigma floor
        if sigma_per_joint is None:
            inv_var = np.ones(self._template.n_joints, dtype=np.float64)
        else:
            sig = np.asarray(sigma_per_joint, dtype=np.float64)
            if sig.shape != (self._template.n_joints,):
                raise ValueError("sigma_per_joint shape mismatch")
            # Clamp NaN/non-finite to floor as well
            sig = np.where(np.isfinite(sig), sig, self._sigma_floor)
            sig = np.maximum(sig, self._sigma_floor)
            inv_var = 1.0 / (sig * sig)
        # Drop NaN-joint contributions
        inv_var = inv_var * valid_mask.astype(np.float64)

        # Vectorized cost over all bins
        # mu: (n_bins, n_joints); diff[b, k] = q[k] - μ[b, k]
        # Codex IMPROVE: this allocates per call — C++ port must preallocate.
        mu = self._template.mu  # defensive copy from template
        q_clean = np.where(valid_mask, q_arr, 0.0)
        diff = q_clean[None, :] - mu  # (n_bins, n_joints)
        costs = (diff * diff * inv_var[None, :]).sum(axis=1)  # (n_bins,)

        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])

        # Subpixel parabola between best ± 1 bin (wrap)
        n_bins = self._template.n_bins
        i_m1 = (best_idx - 1) % n_bins
        i_p1 = (best_idx + 1) % n_bins
        c_m = float(costs[i_m1])
        c_0 = float(costs[best_idx])
        c_p = float(costs[i_p1])
        denom = c_m - 2.0 * c_0 + c_p
        if abs(denom) > 1e-12:
            delta = 0.5 * (c_m - c_p) / denom
            delta = max(-0.5, min(0.5, delta))
        else:
            delta = 0.0

        bin_pos = best_idx + delta
        phi_est = float(wrap_to_2pi(bin_pos * TWO_PI / n_bins))

        # Ambiguity ratio: cost_min / cost_second_best (excluding ±1 of best)
        # Low = sharp, high quality. Spec-aligned semantics.
        mask = np.ones(n_bins, dtype=bool)
        mask[best_idx] = False
        mask[i_m1] = False
        mask[i_p1] = False
        if mask.any():
            second_best = float(np.min(costs[mask]))
        else:
            second_best = best_cost
        if second_best <= 1e-12:
            # Two equal zeros → fully ambiguous
            ambiguity_ratio = 1.0
        else:
            ambiguity_ratio = max(0.0, min(1.0, best_cost / second_best))

        return PhaseEstimate(
            phi=phi_est,
            cost=best_cost,
            ambiguity_ratio=ambiguity_ratio,
            n_bins_searched=n_bins,
            valid=True,
        )
