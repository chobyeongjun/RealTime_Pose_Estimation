"""Cross-correlation phase estimator — template matching for φ observation.

Given q (single 6-joint observation) + template μ(φ), find φ minimizing
Mahalanobis distance:

    cost(φ_c) = Σ_k (q_k - μ_k(φ_c))² / σ_k²

Search:
    1. Evaluate cost at all n_bins bin centers (vectorized).
    2. Pick best bin.
    3. Parabola subpixel fit between best ± 1 bin (sub-bin precision).
    4. Compute confidence = (cost_min / cost_second_min) ratio.

Why cross-correlation (not FFT/Hilbert) — per Codex Q1, plan_d_predictor_spec.md §2.5:
    - Online single-sample observation, not windowed signal.
    - Robust to asymmetric/non-sinusoidal gait (pathological patients).
    - O(n_bins × n_joints) per call — cheap for n_bins=128, n_joints=6.

Returns PhaseEstimate(phi, cost, confidence, n_bins_searched).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.plan_d_prototype.cycle_template import CycleTemplate
from perception.plan_d_prototype.utils import TWO_PI, wrap_to_2pi


@dataclass
class PhaseEstimate:
    """Result of cross-correlation phase observation.

    Attributes:
        phi: estimated phase (rad in [0, 2π)) — NaN if template not ready.
        cost: best Mahalanobis cost (lower is better).
        confidence: cost_min / cost_second_min, ∈ [0, 1]. Lower = sharper minimum.
                    >0.9 → ambiguous (multiple candidate minima).
        n_bins_searched: bins evaluated (for diagnostics).
        valid: convenience boolean (template ready AND finite output).
    """

    phi: float
    cost: float
    confidence: float
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
    ) -> None:
        """
        Args:
            template: CycleTemplate (reference, not copied — caller mutates).
            min_touched_fraction: require at least this fraction of bins
                touched before estimator returns valid result. Default 0.25
                (32 bins out of 128 — about ¼ stride observed).
        """
        if template.n_joints < 1:
            raise ValueError("template must have ≥ 1 joint")
        self._template = template
        self._min_touched = float(min_touched_fraction)

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

        Returns:
            PhaseEstimate with subpixel φ + confidence.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self._template.n_joints,):
            raise ValueError(
                f"q shape {q_arr.shape} != ({self._template.n_joints},)"
            )

        # Template readiness gate
        if self._template.touched_fraction < self._min_touched:
            return PhaseEstimate(
                phi=float("nan"),
                cost=float("inf"),
                confidence=1.0,
                n_bins_searched=0,
                valid=False,
            )

        # NaN joints → exclude from cost (effective per-joint mask)
        valid_mask = np.isfinite(q_arr)
        if not valid_mask.any():
            return PhaseEstimate(
                phi=float("nan"),
                cost=float("inf"),
                confidence=1.0,
                n_bins_searched=0,
                valid=False,
            )

        # Inverse variances
        if sigma_per_joint is None:
            inv_var = np.ones(self._template.n_joints, dtype=np.float64)
        else:
            sig = np.asarray(sigma_per_joint, dtype=np.float64)
            if sig.shape != (self._template.n_joints,):
                raise ValueError("sigma_per_joint shape mismatch")
            # Avoid divide by zero / extreme weighting
            sig = np.maximum(sig, 1e-6)
            inv_var = 1.0 / (sig * sig)
        # Drop NaN-joint contributions
        inv_var = inv_var * valid_mask.astype(np.float64)

        # Vectorized cost over all bins
        # mu: (n_bins, n_joints); diff[b, k] = q[k] - μ[b, k]
        mu = self._template.mu  # already a copy
        # Replace NaN entries in q with 0 (masked via inv_var=0 anyway)
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
        # δ ∈ [-0.5, 0.5] — fractional offset from best_idx
        denom = c_m - 2.0 * c_0 + c_p
        if abs(denom) > 1e-12:
            delta = 0.5 * (c_m - c_p) / denom
            delta = max(-0.5, min(0.5, delta))
        else:
            delta = 0.0

        bin_pos = best_idx + delta
        phi_est = float(wrap_to_2pi(bin_pos * TWO_PI / n_bins))

        # Confidence: cost_min / cost_second_best (excluding ±1 of best)
        mask = np.ones(n_bins, dtype=bool)
        mask[best_idx] = False
        mask[i_m1] = False
        mask[i_p1] = False
        if mask.any():
            second_best = float(np.min(costs[mask]))
        else:
            second_best = best_cost
        if second_best <= 1e-12:
            # Two equal zeros → ambiguous
            confidence = 1.0
        else:
            confidence = max(0.0, min(1.0, best_cost / second_best))

        return PhaseEstimate(
            phi=phi_est,
            cost=best_cost,
            confidence=confidence,
            n_bins_searched=n_bins,
            valid=True,
        )
