"""Custom CUDA sparse stereo — Census 또는 SAD at K keypoints.

⚠️ ARCHIVED (2026-05-12 사용자 결정 C — V4L2 우회 abandon) ⚠️
알고리즘 self-test PASS (3/3 correct, depth 1.008m, sigma_z 8.4mm @ 1m).
단 V4L2 path 폐기 → production 미사용. 코드 유지 = future C++ libargus prototype reference.

Codex orchestration Q3 spec (V4L2 우회 path 의 핵심):
- Block 1 keypoint = 1 thread block
- Patch (9x9 or 11x11) matching
- L/R consistency check (left→right + right→left agreement)
- Subpixel parabola fit (cost 의 quadratic interpolation)
- Confidence based on cost ratio

CPU/PyTorch fallback (Mac executable for unit test) + Jetson 의 *production CUDA kernel*
은 별도 implement (이번 phase = skeleton).

Plan D EKF 의 *measurement R* source:
- Disparity error σ_d (subpixel) → Z error σ_z = Z² × σ_d / (fx × B)
- Per-keypoint depth uncertainty 의 진정 추정.

Reference:
- Hirschmuller 2008 — Semi-Global Matching (SGM)
- Censuses + Hamming distance = robust to illumination
- Sub-pixel parabola: d_sub = d - 0.5 * (C(d+1) - C(d-1)) / (C(d+1) - 2C(d) + C(d-1))
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def sparse_stereo_disparity_pytorch(
    left_gray: "torch.Tensor",      # (H, W) uint8 — left image
    right_gray: "torch.Tensor",     # (H, W) uint8 — right image (rectified)
    keypoints_xy: "torch.Tensor",   # (K, 2) float32 — left image coords (x, y)
    patch_size: int = 9,
    max_disparity: int = 128,
    disparity_prior: Optional["torch.Tensor"] = None,   # (K,) float32 EKF prior
    prior_range: int = 16,
    do_lr_consistency: bool = True,
    do_subpixel: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """PyTorch reference — SAD-based sparse stereo.

    Slow on CPU (Python loop). Production = CUDA kernel.

    Args:
        left_gray, right_gray: (H, W) uint8. Already rectified (epipolar lines = horizontal).
        keypoints_xy: (K, 2) — left image (x, y).
        patch_size: matching window (9x9 default).
        max_disparity: search range.
        disparity_prior: (K,) — EKF 의 previous disparity 예측. None = full search.
        prior_range: prior ± range 의 search window.
        do_lr_consistency: True = left→right + right→left 의 agreement check.
        do_subpixel: True = parabola fit for sub-pixel disparity.

    Returns:
        disparity (K,) float32. Invalid keypoint = NaN.
        confidence (K,) float32 in [0, 1]. 1 = perfect match.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch required") from e

    K = keypoints_xy.shape[0]
    H, W = left_gray.shape
    p_half = patch_size // 2
    disparity = torch.full((K,), float("nan"), dtype=torch.float32,
                            device=left_gray.device)
    confidence = torch.zeros(K, dtype=torch.float32, device=left_gray.device)

    left_f = left_gray.float()
    right_f = right_gray.float()

    for i in range(K):
        u = int(keypoints_xy[i, 0].item())
        v = int(keypoints_xy[i, 1].item())
        # Bounds check
        if (u - p_half < 0 or u + p_half >= W or
            v - p_half < 0 or v + p_half >= H):
            continue
        left_patch = left_f[v-p_half:v+p_half+1, u-p_half:u+p_half+1]

        # Disparity search range
        if disparity_prior is not None and torch.isfinite(disparity_prior[i]):
            d_prior = int(disparity_prior[i].item())
            d_min = max(0, d_prior - prior_range)
            d_max = min(max_disparity, d_prior + prior_range)
        else:
            d_min, d_max = 0, min(max_disparity, u - p_half - 1)

        if d_max <= d_min:
            continue

        # SAD over disparities (left→right search)
        costs = []
        for d in range(d_min, d_max + 1):
            u_r = u - d
            if u_r - p_half < 0:
                costs.append(float("inf"))
                continue
            right_patch = right_f[v-p_half:v+p_half+1, u_r-p_half:u_r+p_half+1]
            sad = (left_patch - right_patch).abs().sum().item()
            costs.append(sad)

        cost_arr = np.array(costs, dtype=np.float32)
        best_idx = int(np.argmin(cost_arr))
        best_d = d_min + best_idx
        best_cost = cost_arr[best_idx]

        # L/R consistency: right→left search around predicted u
        if do_lr_consistency:
            u_r = u - best_d
            if u_r - p_half >= 0 and u_r + p_half < W:
                right_patch_at_best = right_f[v-p_half:v+p_half+1,
                                                u_r-p_half:u_r+p_half+1]
                # right→left: search around u for best match to right_patch_at_best
                lr_costs = []
                for d_r in range(d_min, d_max + 1):
                    u_l = u_r + d_r
                    if u_l - p_half < 0 or u_l + p_half >= W:
                        lr_costs.append(float("inf"))
                        continue
                    left_patch_search = left_f[v-p_half:v+p_half+1,
                                                  u_l-p_half:u_l+p_half+1]
                    lr_costs.append(
                        (right_patch_at_best - left_patch_search).abs().sum().item()
                    )
                lr_best = d_min + int(np.argmin(lr_costs))
                if abs(lr_best - best_d) > 1:
                    # Inconsistent — invalid keypoint
                    continue

        # Subpixel parabola fit (Lucas-Kanade-like quadratic interpolation)
        d_subpixel = float(best_d)
        if do_subpixel and 0 < best_idx < len(cost_arr) - 1:
            c_minus = cost_arr[best_idx - 1]
            c = cost_arr[best_idx]
            c_plus = cost_arr[best_idx + 1]
            denom = c_minus - 2 * c + c_plus
            if abs(denom) > 1e-6:
                d_subpixel = best_d - 0.5 * (c_plus - c_minus) / denom

        # Confidence: inverse normalized cost + ratio of best/second-best
        cost_normalized = best_cost / (patch_size * patch_size * 255.0)
        # Second best (excluding ±1 of best)
        cost_arr_modified = cost_arr.copy()
        cost_arr_modified[max(0, best_idx-1):best_idx+2] = float("inf")
        second_best = float(np.min(cost_arr_modified)) if len(cost_arr_modified) > 3 else best_cost * 2
        ratio = best_cost / max(second_best, 1e-3)
        conf = (1.0 - cost_normalized) * (1.0 - ratio)
        conf = max(0.0, min(1.0, conf))

        # ★ Codex Jetson 2026-05-12 fix: numpy.float32 → torch tensor 직접 assign fail.
        # Python float 으로 cast 후 assign (torch 가 자동 wrap).
        disparity[i] = float(d_subpixel)
        confidence[i] = float(conf)

    return disparity, confidence


def disparity_to_depth(
    disparity: "torch.Tensor",
    fx: float, baseline_m: float,
) -> "torch.Tensor":
    """disparity (px) → depth (m). Z = fx × B / d.

    Returns:
        depth (K,) float32. NaN/inf input → NaN output.
    """
    import torch
    eps = 1e-6
    d = disparity.clamp(min=eps)
    depth = fx * baseline_m / d
    # Mark invalid (NaN input → NaN output)
    depth = torch.where(torch.isfinite(disparity), depth, torch.full_like(depth, float("nan")))
    return depth


def depth_uncertainty_sigma(
    disparity: "torch.Tensor",
    fx: float, baseline_m: float,
    sigma_d_subpixel: float = 0.25,
) -> "torch.Tensor":
    """depth uncertainty σ_z (m) — Plan D EKF measurement R 의 source.

    σ_z = Z² × σ_d / (fx × B)
    """
    import torch
    Z = disparity_to_depth(disparity, fx, baseline_m)
    return Z * Z * sigma_d_subpixel / (fx * baseline_m)


def _cpu_self_test() -> int:
    """Mac executable — synthetic stereo pair test (no CUDA)."""
    import torch
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOGGER.info("=== sparse stereo CPU self-test ===")

    rng = np.random.default_rng(42)
    H, W = 100, 200
    # Natural-like texture (gradient + noise)
    left = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            left[y, x] = ((y * 3 + x * 5) % 200) + rng.integers(-20, 20)
    left = np.clip(left, 0, 255).astype(np.uint8)

    # Right = left shifted by *known disparity* (30 px) — perfect stereo pair
    true_disparity = 30
    right = np.zeros_like(left)
    right[:, :-true_disparity] = left[:, true_disparity:]

    left_t = torch.from_numpy(left)
    right_t = torch.from_numpy(right)

    keypoints = torch.tensor([[100, 50], [150, 50], [100, 70]], dtype=torch.float32)
    disparity, confidence = sparse_stereo_disparity_pytorch(
        left_t, right_t, keypoints, patch_size=9, max_disparity=80,
        do_lr_consistency=True, do_subpixel=True,
    )

    LOGGER.info(f"true disparity: {true_disparity}")
    LOGGER.info(f"measured: {disparity.tolist()}")
    LOGGER.info(f"confidence: {confidence.tolist()}")

    # Check: each detected disparity within 2 px of true
    n_correct = 0
    for i in range(len(keypoints)):
        if torch.isfinite(disparity[i]) and abs(disparity[i].item() - true_disparity) < 2:
            n_correct += 1
    LOGGER.info(f"correct: {n_correct} / {len(keypoints)}")

    # Depth + uncertainty
    fx, baseline = 480.0, 0.063
    depth = disparity_to_depth(disparity, fx, baseline)
    sigma_z = depth_uncertainty_sigma(disparity, fx, baseline)
    LOGGER.info(f"depth: {depth.tolist()}")
    LOGGER.info(f"sigma_z (m): {sigma_z.tolist()}")

    if n_correct >= 2:
        LOGGER.info("✓ sparse stereo CPU self-test PASS")
        return 0
    LOGGER.error(f"✗ FAIL: only {n_correct} correct")
    return 1


if __name__ == "__main__":
    raise SystemExit(_cpu_self_test())
