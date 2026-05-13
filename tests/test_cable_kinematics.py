"""Unit tests for scripts/cable_kinematics.py.

Locks the slack-free cable algorithm against regression:
  1. 5 N pretension defines the reference.
  2. Δℓ = ℓ_free(t) - ℓ_free(t_cal).
  3. θ_motor_target = θ_ref + Δℓ / r_pulley.
  4. Stance / swing classification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cable_kinematics import (  # noqa: E402
    WalkerGeometry,
    CableReference,
    shank_attach_point,
    free_cable_length,
    calibrate_pretension,
    motor_target,
    is_stance,
)


def _default_geometry() -> WalkerGeometry:
    return WalkerGeometry(
        r_pulley=0.0125,
        sheath_internal_length=0.85,
        P_anchor_left=np.array([0.18, -0.05, 0.10]),
        P_anchor_right=np.array([-0.18, -0.05, 0.10]),
        shank_attach_alpha=0.7,
        pretension_force_N=5.0,
        slack_margin=-0.005,
    )


def test_shank_attach_at_alpha_endpoints():
    knee = np.array([0.0, 0.5, 1.0])
    ankle = np.array([0.0, 0.95, 1.05])
    assert np.allclose(shank_attach_point(knee, ankle, 0.0), knee)
    assert np.allclose(shank_attach_point(knee, ankle, 1.0), ankle)
    mid = shank_attach_point(knee, ankle, 0.5)
    assert np.allclose(mid, 0.5 * (knee + ankle))


def test_shank_attach_nan_propagates():
    knee = np.array([0.0, np.nan, 1.0])
    ankle = np.array([0.0, 1.0, 1.0])
    p = shank_attach_point(knee, ankle, 0.7)
    assert np.isnan(p[1])


def test_free_cable_length_basic():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert abs(free_cable_length(a, b) - 5.0) < 1e-9


def test_free_cable_length_nan():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, np.nan, 0.0])
    assert np.isnan(free_cable_length(a, b))


def test_pretension_calibration_records_reference():
    """Step 2 of user algorithm: 5 N tension instant captures (motor, ℓ_free, P_shank)."""
    geom = _default_geometry()
    knee = np.array([0.10, 0.50, 1.10])
    ankle = np.array([0.10, 0.95, 1.20])
    ref = calibrate_pretension(
        side="L", theta_motor_reading=0.0,
        knee_3d=knee, ankle_3d=ankle, geometry=geom,
    )
    assert ref.side == "L"
    assert ref.theta_motor_ref == 0.0
    # Expected ℓ_free
    p_shank = (1 - 0.7) * knee + 0.7 * ankle
    expected_ell = np.linalg.norm(geom.P_anchor_left - p_shank)
    assert abs(ref.ell_free_ref - expected_ell) < 1e-9


def test_motor_target_unchanged_at_reference():
    """At the pretension instant, Δℓ = 0 → θ_motor_target = θ_motor_ref + slack_margin/r."""
    geom = _default_geometry()
    knee = np.array([0.10, 0.50, 1.10])
    ankle = np.array([0.10, 0.95, 1.20])
    ref = calibrate_pretension("L", 0.0, knee, ankle, geom)
    p_shank_now = shank_attach_point(knee, ankle, geom.shank_attach_alpha)
    theta, delta_ell, ell_now = motor_target(p_shank_now, ref, geom, apply_margin=True)
    # delta_ell = slack_margin
    assert abs(delta_ell - geom.slack_margin) < 1e-9
    assert abs(ell_now - ref.ell_free_ref) < 1e-9


def test_motor_target_tracks_user_moving_away():
    """User shifts further from walker (Z increases) → free cable longer → motor pays out."""
    geom = _default_geometry()
    knee0  = np.array([0.10, 0.50, 1.10])
    ankle0 = np.array([0.10, 0.95, 1.20])
    ref = calibrate_pretension("L", 0.0, knee0, ankle0, geom)

    # User moves 10 cm further forward (along Z = horizontal distance)
    knee1  = knee0  + np.array([0.0, 0.0, 0.10])
    ankle1 = ankle0 + np.array([0.0, 0.0, 0.10])
    p_shank1 = shank_attach_point(knee1, ankle1, geom.shank_attach_alpha)
    theta, delta_ell, ell_now = motor_target(p_shank1, ref, geom, apply_margin=False)

    # Free cable longer
    assert ell_now > ref.ell_free_ref
    assert delta_ell > 0
    # Motor must rotate (pay cable out)
    assert theta > ref.theta_motor_ref


def test_motor_target_nan_propagation():
    geom = _default_geometry()
    knee = np.array([0.10, 0.50, 1.10])
    ankle = np.array([0.10, 0.95, 1.20])
    ref = calibrate_pretension("L", 0.0, knee, ankle, geom)
    bad = np.array([0.0, np.nan, 1.0])
    theta, dell, ell = motor_target(bad, ref, geom)
    assert np.isnan(theta) and np.isnan(dell) and np.isnan(ell)


def test_is_stance_left_at_phi_zero():
    """phi = 0 is HS_L → start of left stance."""
    assert is_stance(0.0, side="L")
    # Beyond stance fraction → swing
    assert not is_stance(0.65 * 2 * np.pi, side="L")


def test_is_stance_right_phase_shift():
    """Right side is shifted by π; right stance starts at phi = π and runs
    for stance_fraction · 2π.

    Note: phi=0 is HS_L. Right stance at that instant is just ENDING
    (its window [π, π + 0.6·2π) wraps to include [0, 0.2·2π)). So we test
    a phi that is unambiguously in left-stance / right-swing.
    """
    # phi = 0.3·2π — left mid-stance, right mid-swing → right NOT stance
    assert not is_stance(0.3 * 2 * np.pi, side="R")
    # phi = π — right HS → right stance starts
    assert is_stance(np.pi, side="R")
    # phi just inside right stance window
    assert is_stance(np.pi + 0.5 * 2 * np.pi, side="R")
    # phi past right stance end (0.65 · 2π after right HS)
    assert not is_stance(np.pi + 0.65 * 2 * np.pi, side="R")


def test_stance_fraction_override():
    """Custom stance fraction (per-subject template) is honored."""
    assert is_stance(0.55 * 2 * np.pi, side="L", stance_fraction=0.60)
    assert not is_stance(0.55 * 2 * np.pi, side="L", stance_fraction=0.50)


def test_walker_geometry_from_yaml():
    """Loader round-trips the shipped config."""
    geom = WalkerGeometry.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "walker_geometry.yaml"
    )
    assert 0 < geom.r_pulley < 0.1
    assert geom.P_anchor_left.shape == (3,)
    assert geom.P_anchor_right.shape == (3,)
    assert 0 <= geom.shank_attach_alpha <= 1
    assert geom.pretension_force_N > 0
    # Default ships with slightly-slack margin (or 0). Never positive (would over-extend).
    assert geom.slack_margin <= 0


def test_motor_displacement_for_5mm_user_movement():
    """Sanity: 5 mm cable-length change → ~5/12.5 = 0.4 rad ≈ 23° motor rotation."""
    geom = _default_geometry()
    knee0 = np.array([0.10, 0.50, 1.10])
    ankle0 = np.array([0.10, 0.95, 1.20])
    ref = calibrate_pretension("L", 0.0, knee0, ankle0, geom)

    # Shift user 5 mm forward (along the cable line approximately)
    knee1  = knee0  + np.array([0.0, 0.0, 0.005])
    ankle1 = ankle0 + np.array([0.0, 0.0, 0.005])
    p_shank1 = shank_attach_point(knee1, ankle1, geom.shank_attach_alpha)
    theta, dell, _ = motor_target(p_shank1, ref, geom, apply_margin=False)

    # Δℓ approximately equals the cable-direction projection of the user motion.
    # With anchor near (0.18, -0.05, 0.10) and user near (0.10, ..., 1.1),
    # the cable line is roughly along +Z, so Δℓ ≈ user Δz = 5 mm.
    assert 0.001 < dell < 0.006, f"Δℓ should be ~5 mm, got {dell*1000:.2f} mm"
    expected_theta = dell / geom.r_pulley
    assert abs(theta - expected_theta) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
