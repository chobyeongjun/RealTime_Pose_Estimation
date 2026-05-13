"""Phase 2A unit tests — world_up_vec plumbing + ZED quaternion convention.

Locks the world_up vector formula and the thigh/shank inclination math
against regression. Codex consult #2/#3/#9 all hit here.

Runs on Mac (no pyzed, no ZED hardware).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src" / "perception" / "realtime"))
sys.path.insert(0, str(_REPO / "src"))

from calibration import ZEDIMUWorldFrame  # noqa: E402
from joint_3d import compute_joint_state, JointState3D  # noqa: E402


# ───────────────────────── quaternion convention ─────────────────────────


def test_quaternion_identity_yields_world_up_y():
    """Codex #2: identity quaternion (camera level + aligned with world)
    must produce world-up = (0, 1, 0) in camera frame.

    ZED IMU convention: q = camera→world rotation. Identity means cam frame
    == world frame. World Y is up, so camera frame's up axis is also Y.
    """
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, z, w) — no rotation
    up = ZEDIMUWorldFrame._compute_world_up_camera(q_identity)
    assert np.allclose(up, [0.0, 1.0, 0.0], atol=1e-6), (
        f"identity q should give world-Y in camera frame, got {up}"
    )


def test_quaternion_pitch_30deg():
    """Camera pitched 30° forward about its X axis. The world-up vector
    should rotate toward camera-Z axis (downward toward optical axis).

    A pitch of +30° about X (right-hand rule, X = right) tilts the camera
    such that what used to be 'up' (camera Y) is now part Y, part -Z.

    For axis-angle (1,0,0,30°):
        q = (sin(15°), 0, 0, cos(15°))
    World up in camera = the column-1 result of R^T·(0,1,0). For a +30° pitch
    of the camera (camera→world tilts world axes the other way), the up
    vector expressed in camera frame ends up tilted by -30° about X.
    Expected: up = (0, cos30°, sin30°) for a passive frame rotation.
    """
    angle = math.radians(30.0)
    q = np.array([math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)])
    up = ZEDIMUWorldFrame._compute_world_up_camera(q)
    # Use the same formula as a reference: 2*(xy-wz), 1-2*(x²+z²), 2*(yz+wx)
    x, y, z, w = q
    expected = np.array([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dtype=np.float32)
    assert np.allclose(up, expected, atol=1e-6), (
        f"pitch-30 q: got {up}, expected {expected}"
    )
    # Sanity: still mostly +Y, with some Z bias from the pitch
    assert up[1] > 0.85, f"after 30° pitch, up_y should still dominate, got {up}"


def test_compute_world_up_camera_equals_R_column1():
    """Codex #2 follow-up: assert _compute_world_up_camera(q) is exactly the
    second column of _quat_to_R(q). The two formulations must be identical;
    if they ever diverge, one is wrong and the bug surfaces in inclination
    sign flips during walking.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        # random quaternion
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        up_formula = ZEDIMUWorldFrame._compute_world_up_camera(q)
        R = ZEDIMUWorldFrame._quat_to_R(q)
        up_R_col1 = R[:, 1].astype(np.float32)
        assert np.allclose(up_formula, up_R_col1, atol=1e-6), (
            f"formula mismatch: up={up_formula}, R[:,1]={up_R_col1}"
        )


# ───────────────────────── inclination sign ─────────────────────────


def _make_state(positions: dict) -> tuple[dict, dict, dict]:
    """Build (kp2d, raw_3d, confs) inputs for compute_joint_state.

    2D pixels are placeholders (positions only used in 3D inclination
    math). All confs set to 0.9 so the conf threshold passes.
    """
    kp2d = {n: (320.0, 240.0) for n in positions}
    raw_3d = {n: np.asarray(p, dtype=np.float32) for n, p in positions.items()}
    confs = {n: 0.9 for n in positions}
    return kp2d, raw_3d, confs


def test_standing_leg_inclination_near_zero():
    """Codex #3: a standing leg (knee directly below hip, ankle directly
    below knee, world up = world Y) must give thigh_inclination ≈ 0°.

    Camera convention: ZED Y axis points down in camera frame for
    OPENCV-style coords, but pipeline_main uses world up vector directly
    (not -up) in dot product. The current joint_3d.py:238 uses dot(thigh,
    -up). So if 'up' is the gravity-opposite (Y_world up = (0, 1, 0)
    expressed in cam frame), and the leg is along -world_Y (gravity dir),
    then thigh vector hip→knee points DOWN in world Y = +cam_Y (if up_cam
    = -cam_Y), and dot(thigh, -up) = dot((0,-1,0), -(0,1,0)*world→cam)...

    Simpler check: build positions in a frame where world up = (0, 1, 0)
    in that frame. A standing leg has thigh pointing in -up direction
    (knee below hip). joint_3d.py computes:
        cos(angle) = dot(thigh_unit, -up_unit)
    For standing: thigh_unit = -up_unit, so cos(angle) = 1, angle = 0°.
    """
    # All in "camera" frame where world up = +Y
    positions = {
        'left_hip':   (0.10, 1.00, 1.50),  # higher in world (+Y)
        'left_knee':  (0.10, 0.55, 1.50),  # below hip
        'left_ankle': (0.10, 0.10, 1.50),  # below knee
        'right_hip':  (-0.10, 1.00, 1.50),
        'right_knee': (-0.10, 0.55, 1.50),
        'right_ankle': (-0.10, 0.10, 1.50),
    }
    kp2d, raw_3d, confs = _make_state(positions)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    state = compute_joint_state(kp2d, raw_3d, confs,
                                timestamp_us=0.0, world_up_vec=world_up)
    assert state.valid, "standing-leg state must be valid"
    assert state.left_thigh_inclination is not None, (
        "world_up_vec passed but inclination still None — plumbing broken"
    )
    # Standing → thigh aligned with -up → inclination ≈ 0°
    assert abs(state.left_thigh_inclination) < 1.0, (
        f"standing leg thigh inclination should be ~0°, got "
        f"{state.left_thigh_inclination}° (sign flip?)"
    )
    assert abs(state.left_shank_inclination) < 1.0, (
        f"standing leg shank inclination should be ~0°, got "
        f"{state.left_shank_inclination}°"
    )
    assert abs(state.right_thigh_inclination) < 1.0
    assert abs(state.right_shank_inclination) < 1.0


def test_thigh_forward_tilt_30deg():
    """Hip held, knee swung forward by 30° (toward +Z), shank tilted
    similarly. thigh_inclination should report ≈ 30°.

    This locks the magnitude as well as the sign, ruling out 60°/120°
    confusions from the wrong axis convention.
    """
    # World-up = (0, 1, 0) in this synthetic frame
    # Thigh length 0.45 m. Knee = hip + 0.45 * (sin30°, -cos30°, 0)
    # → tilted 30° forward in +X (or +Z) plane.
    L = 0.45
    angle = math.radians(30.0)
    hip = np.array([0.10, 1.00, 1.50])
    # Tilt in X direction so that the projection onto -world_up gives cos(30°)
    knee_offset = np.array([L * math.sin(angle), -L * math.cos(angle), 0.0])
    knee = hip + knee_offset
    ankle = knee + knee_offset  # shank also tilted 30°, simplification

    positions = {
        'left_hip':   tuple(hip),
        'left_knee':  tuple(knee),
        'left_ankle': tuple(ankle),
        'right_hip':  tuple(hip + np.array([-0.20, 0, 0])),
        'right_knee': tuple(knee + np.array([-0.20, 0, 0])),
        'right_ankle': tuple(ankle + np.array([-0.20, 0, 0])),
    }
    kp2d, raw_3d, confs = _make_state(positions)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    state = compute_joint_state(kp2d, raw_3d, confs,
                                timestamp_us=0.0, world_up_vec=world_up)
    assert state.left_thigh_inclination is not None
    assert abs(state.left_thigh_inclination - 30.0) < 1.5, (
        f"30° forward tilt expected, got {state.left_thigh_inclination}°"
    )


def test_compute_joint_state_threads_world_up():
    """Integration: when world_up_vec is None, inclinations stay None
    (the legacy / pipeline_main.py:464 buggy path). When passed, they fill.

    This is the regression check for the original bug: pipeline_main never
    threaded world_up_vec, so state.thigh_inclination always stayed None,
    so six_valid always failed, so predictor.feed was never called.
    """
    positions = {
        'left_hip':   (0.0, 1.0, 1.5),
        'left_knee':  (0.0, 0.5, 1.5),
        'left_ankle': (0.0, 0.0, 1.5),
        'right_hip':  (-0.2, 1.0, 1.5),
        'right_knee': (-0.2, 0.5, 1.5),
        'right_ankle': (-0.2, 0.0, 1.5),
    }
    kp2d, raw_3d, confs = _make_state(positions)

    state_no_up = compute_joint_state(kp2d, raw_3d, confs, timestamp_us=0.0,
                                       world_up_vec=None)
    assert state_no_up.left_thigh_inclination is None
    assert state_no_up.left_shank_inclination is None

    state_with_up = compute_joint_state(kp2d, raw_3d, confs, timestamp_us=0.0,
                                         world_up_vec=np.array([0, 1, 0], np.float32))
    assert state_with_up.left_thigh_inclination is not None
    assert state_with_up.left_shank_inclination is not None


# ───────────────────────── world_up_in_camera() accessor ─────────────────────────


def test_world_up_in_camera_returns_none_before_init():
    """Until init() succeeds, world_up_in_camera() must be None so the
    pipeline does not feed garbage into compute_joint_state.
    """
    wf = ZEDIMUWorldFrame(zed_camera=None)
    assert wf.world_up_in_camera() is None


def test_world_up_in_camera_caches_after_manual_R():
    """If callers manually set _R + _imu_ok + _world_up_camera (used by
    calibration.py self-tests), the accessor must return the cached array.
    """
    wf = ZEDIMUWorldFrame(zed_camera=None)
    wf._R = np.eye(3, dtype=np.float32)
    wf._imu_ok = True
    wf._world_up_camera = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = wf.world_up_in_camera()
    assert up is not None
    assert np.allclose(up, [0, 1, 0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
