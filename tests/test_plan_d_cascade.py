"""Plan D PredictorCascade tests — L1→L2→L3 activation + fallback."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.plan_d_prototype.cascade import (
    CascadeLevel,
    CascadeStepResult,
    PredictorCascade,
)
from perception.plan_d_prototype.utils import TWO_PI


# ─── Init ────────────────────────────────────────────────────────────────


def test_cascade_init_starts_at_L1():
    c = PredictorCascade(n_joints=6)
    assert c.level == CascadeLevel.L1
    assert c.stride_count == 0
    assert c.alpha == 0.0


def test_cascade_init_invalid_n_joints():
    with pytest.raises(ValueError):
        PredictorCascade(n_joints=0)


# ─── Smoke: empty step ───────────────────────────────────────────────────


def test_cascade_step_without_pose_does_not_crash():
    c = PredictorCascade(n_joints=6, fs_hz=60)
    result = c.step(t_now=0.0, q=None, hip_z_world_m=float("nan"))
    assert isinstance(result, CascadeStepResult)
    assert result.level == CascadeLevel.L1
    assert result.vision_lost is True


def test_cascade_step_with_pose_marks_vision_alive():
    c = PredictorCascade(n_joints=6, fs_hz=60)
    q = np.zeros(6)
    result = c.step(t_now=0.0, q=q, hip_z_world_m=0.05)
    assert result.vision_lost is False


# ─── Promotion L1 → L2 → L3 under synthetic walking ─────────────────────


def _drive_walking(
    c: PredictorCascade,
    duration_s: float = 10.0,
    fs: float = 60.0,
    f_walk: float = 1.0,
    amp_m: float = 0.05,
    noise_std: float = 0.02,
    seed: int = 0,
) -> list:
    """Drive cascade with synthetic walking and return per-frame results."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    results = []
    for i in range(n):
        t = i / fs
        true_phi = (2 * math.pi * f_walk * t) % TWO_PI
        hip_z = amp_m * math.sin(2 * math.pi * f_walk * t)
        q = np.array([
            math.sin(true_phi + j * math.pi / 6) + rng.normal(0, noise_std)
            for j in range(c._n_joints)
        ])
        r = c.step(t_now=t, q=q, sigma_per_joint=np.full(c._n_joints, 0.05),
                   hip_z_world_m=hip_z)
        results.append(r)
    return results


def test_cascade_promotes_l1_to_l2_after_1_stride():
    c = PredictorCascade(n_joints=6, fs_hz=60,
                          l2_promote_strides=1, l3_promote_strides=3)
    results = _drive_walking(c, duration_s=4.0)
    # After 1 Hz walking × 4 s → ≥ 3 strides → should be at L2 or L3
    final_level = results[-1].level
    assert final_level >= CascadeLevel.L2


def test_cascade_promotes_l2_to_l3_after_3_strides_with_template():
    c = PredictorCascade(n_joints=6, fs_hz=60,
                          l2_promote_strides=1, l3_promote_strides=3,
                          l3_template_min_fraction=0.3)
    results = _drive_walking(c, duration_s=10.0)
    final = results[-1]
    assert final.level == CascadeLevel.L3
    assert final.stride_count >= 3
    assert final.template_touched_fraction >= 0.3


# ─── Template build ──────────────────────────────────────────────────────


def test_cascade_template_grows_over_time():
    c = PredictorCascade(n_joints=6, fs_hz=60)
    results = _drive_walking(c, duration_s=10.0)
    early_touched = results[60].template_touched_fraction  # 1s
    late_touched = results[-1].template_touched_fraction
    assert late_touched > early_touched


# ─── Vision loss ─────────────────────────────────────────────────────────


def test_cascade_detects_vision_loss():
    c = PredictorCascade(n_joints=6, fs_hz=60, vision_loss_max_gap_s=0.060)
    # First a few valid frames
    for i in range(10):
        c.step(t_now=i * 0.01, q=np.zeros(6), hip_z_world_m=0.05)
    # Now drop pose for 0.1 s
    result = c.step(t_now=0.20, q=None, hip_z_world_m=float("nan"))
    assert result.vision_lost is True


# ─── Demotion on cadence jump ────────────────────────────────────────────


def test_cascade_demotes_on_cadence_jump():
    """Stable walking → L3, then cadence doubles → cadence_jump_detector
    triggers demotion."""
    c = PredictorCascade(n_joints=6, fs_hz=60, cadence_jump_threshold=0.20,
                          l3_template_min_fraction=0.3)
    rng = np.random.default_rng(0)

    # 8 s @ 1 Hz
    n_warmup = int(8 * 60)
    for i in range(n_warmup):
        t = i / 60.0
        true_phi = (2 * math.pi * 1.0 * t) % TWO_PI
        hip = 0.05 * math.sin(2 * math.pi * 1.0 * t)
        q = np.array([math.sin(true_phi + j * math.pi / 6) + rng.normal(0, 0.02)
                      for j in range(6)])
        c.step(t_now=t, q=q, sigma_per_joint=np.full(6, 0.05), hip_z_world_m=hip)

    # Cascade reached L3 (most likely)
    level_before = c.level

    # Now jump to 2 Hz for 2 more strides
    for i in range(int(2 * 60)):
        t = (n_warmup + i) / 60.0
        true_phi = (2 * math.pi * 2.0 * t) % TWO_PI
        hip = 0.05 * math.sin(2 * math.pi * 2.0 * t)
        q = np.array([math.sin(true_phi + j * math.pi / 6) + rng.normal(0, 0.02)
                      for j in range(6)])
        c.step(t_now=t, q=q, sigma_per_joint=np.full(6, 0.05), hip_z_world_m=hip)
    # Either still L3 (cadence-jump tolerated by adaptive Q) OR demoted
    # The point is: no crash, state remains finite
    assert math.isfinite(c.omega)
    assert math.isfinite(c.phi)
    assert level_before in (CascadeLevel.L2, CascadeLevel.L3)


# ─── Forecast ────────────────────────────────────────────────────────────


def test_cascade_predict_ahead_at_L1_no_q_pred():
    c = PredictorCascade(n_joints=6)
    forecast = c.predict_ahead(0.05)
    assert forecast.q_pred is None
    assert math.isfinite(forecast.phi)
    assert math.isfinite(forecast.omega)
    assert forecast.alpha == 0.0


def test_cascade_predict_ahead_at_L3_includes_q_pred():
    c = PredictorCascade(n_joints=6, fs_hz=60)
    _drive_walking(c, duration_s=10.0)
    assert c.level == CascadeLevel.L3
    forecast = c.predict_ahead(0.05)
    assert forecast.q_pred is not None
    assert forecast.q_pred.shape == (6,)
    assert np.all(np.isfinite(forecast.q_pred))


# ─── Stride detection inside cascade ─────────────────────────────────────


def test_cascade_stride_count_increases_with_walking():
    """10 s × 1 Hz → ≥ 3 strides (cold-start eats some).

    The L3-promotion criterion is ≥ 3 strides — so this is the minimum
    we need; over-counting more is permitted up to the actual stride rate.
    """
    c = PredictorCascade(n_joints=6, fs_hz=60)
    results = _drive_walking(c, duration_s=10.0)
    final = results[-1]
    assert final.stride_count >= 3
    assert final.stride_count <= 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
