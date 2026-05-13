"""Cable kinematics for H-Walker — Python prototype, Mac-runnable.

Implements the user's slack-free algorithm:

  1) AK60 motor position is known precisely (encoder feedback on the actuator).
  2) Session calibration: ramp cable tension to ~5 N while the user stands
     still in the walker. Record the encoder position AND the free-cable
     length from vision; that pair defines the "no slack" reference state.
  3) During walking, motor position must adjust so the cable remains at the
     same tension — i.e. free-cable length changes (user moves) get matched
     by motor revolutions: Δθ_motor = Δℓ_free / r_pulley.
  4) Vision provides the depth signal that drives Δℓ_free in real time;
     Plan D forecast (50 ms ahead) lets us emit motor targets predictively
     so stance-phase cable release lands without force jerk.

This module is pure Python + numpy. The same kinematics will be re-coded in
C++ for the 200 Hz control loop later (Phase 5). Here we use it to:
  - Validate cable length traces from existing NPZ walking sessions.
  - Synthetic-walking checks for the predictive payout schedule.
  - Configuration loader for walker_geometry.yaml.

Tests in tests/test_cable_kinematics.py exercise every public function.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ─── Configuration loader ────────────────────────────────────────────────


@dataclass
class WalkerGeometry:
    """Mirrors configs/walker_geometry.yaml. All distances in meters."""
    r_pulley: float
    sheath_internal_length: float
    P_anchor_left: np.ndarray   # (3,) in camera frame
    P_anchor_right: np.ndarray  # (3,)
    shank_attach_alpha: float   # 0..1 along knee→ankle
    pretension_force_N: float
    slack_margin: float

    @staticmethod
    def from_yaml(path: str | Path) -> "WalkerGeometry":
        import yaml
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return WalkerGeometry(
            r_pulley=float(cfg["r_pulley_m"]),
            sheath_internal_length=float(cfg["sheath_internal_length_m"]),
            P_anchor_left=np.asarray(cfg["P_anchor_left_m"], dtype=np.float64),
            P_anchor_right=np.asarray(cfg["P_anchor_right_m"], dtype=np.float64),
            shank_attach_alpha=float(cfg["shank_attach_alpha"]),
            pretension_force_N=float(cfg["pretension_force_N"]),
            slack_margin=float(cfg["slack_margin_m"]),
        )


# ─── Geometry primitives ──────────────────────────────────────────────────


def shank_attach_point(
    knee_3d: np.ndarray,
    ankle_3d: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linear blend between knee and ankle 3D points.

    alpha = 0 → knee, alpha = 1 → ankle, default 0.7 (ankle-side band).
    Returns NaN-propagated 3D vector if either input is NaN.
    """
    knee = np.asarray(knee_3d, dtype=np.float64)
    ankle = np.asarray(ankle_3d, dtype=np.float64)
    return (1.0 - alpha) * knee + alpha * ankle


def free_cable_length(p_anchor: np.ndarray, p_shank: np.ndarray) -> float:
    """Euclidean distance from sheath exit to shank attach point.

    Returns NaN if any input coordinate is non-finite. This is ℓ_free in the
    paper notation; total cable engagement = sheath_internal_length + ℓ_free.
    """
    a = np.asarray(p_anchor, dtype=np.float64)
    b = np.asarray(p_shank, dtype=np.float64)
    diff = a - b
    if not np.all(np.isfinite(diff)):
        return float("nan")
    return float(np.sqrt(np.dot(diff, diff)))


# ─── Pretension calibration ─────────────────────────────────────────────


@dataclass
class CableReference:
    """Snapshot taken at the 5 N pretension instant.

    Stores motor encoder position AND free-cable length so subsequent control
    can compute Δℓ → Δθ_motor without any drift.
    """
    side: str                # 'L' or 'R'
    theta_motor_ref: float   # encoder reading at pretension instant (rad)
    ell_free_ref: float      # free cable length at that instant (m)
    P_shank_ref: np.ndarray  # (3,) shank attach in camera frame at that instant


def calibrate_pretension(
    side: str,
    theta_motor_reading: float,
    knee_3d: np.ndarray,
    ankle_3d: np.ndarray,
    geometry: WalkerGeometry,
) -> CableReference:
    """Record the pretension reference. Call once at session start while
    the user is standing still and the AK60 has just stabilized at 5 N.

    Returns CableReference holding the (motor, free_length, shank_point) tuple
    that defines "no slack" for the rest of the session.
    """
    p_shank = shank_attach_point(knee_3d, ankle_3d, geometry.shank_attach_alpha)
    p_anchor = (geometry.P_anchor_left if side.upper() == "L"
                else geometry.P_anchor_right)
    ell_free = free_cable_length(p_anchor, p_shank)
    return CableReference(
        side=side.upper(),
        theta_motor_ref=float(theta_motor_reading),
        ell_free_ref=ell_free,
        P_shank_ref=p_shank,
    )


# ─── Runtime motor target ────────────────────────────────────────────────


def motor_target(
    p_shank_now: np.ndarray,
    ref: CableReference,
    geometry: WalkerGeometry,
    apply_margin: bool = True,
) -> Tuple[float, float, float]:
    """Compute the motor target for the current shank position.

    Returns:
        theta_motor_target_rad : encoder target relative to absolute zero
        delta_ell_m            : Δℓ_free vs reference
        ell_free_current_m     : current free cable length (for telemetry)

    Returns (NaN, NaN, NaN) if shank position is invalid.
    """
    p_anchor = (geometry.P_anchor_left if ref.side == "L"
                else geometry.P_anchor_right)
    ell_now = free_cable_length(p_anchor, p_shank_now)
    if not np.isfinite(ell_now) or not np.isfinite(ref.ell_free_ref):
        return (float("nan"), float("nan"), float("nan"))
    delta_ell = ell_now - ref.ell_free_ref
    if apply_margin:
        delta_ell += geometry.slack_margin
    theta_target = ref.theta_motor_ref + delta_ell / geometry.r_pulley
    return theta_target, delta_ell, ell_now


# ─── Stance / swing classification from Plan D phi ────────────────────────


def is_stance(
    phi: float,
    side: str = "L",
    stance_fraction: float = 0.60,
) -> bool:
    """Classify a Plan D phase value as stance or swing.

    Convention used elsewhere in the codebase:
        phi = 0  is heel-strike of the LEFT foot (HS_L).
        Left stance occupies phi ∈ [0, stance_fraction · 2π).
        Right is shifted by π so right stance occupies [π, π + stance·2π).

    stance_fraction = 0.6 matches healthy adult gait. Per-subject template
    learning in Plan D L3 can refine this number.
    """
    phi_mod = float(phi) % (2.0 * np.pi)
    if side.upper() == "L":
        return phi_mod < stance_fraction * 2.0 * np.pi
    # Right side: shift by π
    phi_shift = (phi_mod - np.pi) % (2.0 * np.pi)
    return phi_shift < stance_fraction * 2.0 * np.pi


# ─── Predictive payout (50 ms lookahead) ─────────────────────────────────


def predictive_motor_target(
    forecast_q_pred: np.ndarray,   # (6,) hip/knee/ankle joint angles at t+τ
    shank_attach_kinematic: "callable",
    ref: CableReference,
    geometry: WalkerGeometry,
) -> Tuple[float, float]:
    """Compute motor target using a future-pose forecast.

    `forecast_q_pred` is the 6-joint angle vector at t+τ (50 ms).
    `shank_attach_kinematic(q_pred)` reconstructs the shank attach 3D point
    from joint angles + per-subject segment lengths (caller-supplied).

    The kinematic chain is non-trivial (hip translation, body lean, etc.),
    so this prototype delegates that step. Phase 5 C++ will inline it.

    Returns (theta_motor_target_rad, ell_free_future_m).
    """
    p_shank_future = shank_attach_kinematic(forecast_q_pred)
    if p_shank_future is None or not np.all(np.isfinite(p_shank_future)):
        return (float("nan"), float("nan"))
    theta_target, _, ell_now = motor_target(p_shank_future, ref, geometry)
    return theta_target, ell_now


# ─── NPZ-driven trace (Mac analysis, no Jetson) ──────────────────────────


def trace_from_npz(
    npz_path: str | Path,
    geometry: WalkerGeometry,
    side: str = "L",
    pretension_frame_idx: int = 60,   # ~1 s into session (avoid warmup)
) -> dict:
    """Reconstruct cable kinematics over time from an existing NPZ.

    Tries to use the v2 fields first (knee_3d/ankle_3d not in NPZ currently),
    falls back to estimating shank from left_hip_z + segment-length guess.
    Returns per-frame ℓ_free, Δℓ, θ_motor relative to a reference frame.
    """
    z = np.load(npz_path, allow_pickle=True)
    t = z["t_s"]
    n = len(t)
    side_upper = side.upper()

    # Best-effort shank reconstruction. Real Phase 5 uses raw_3d['left_knee']
    # and raw_3d['left_ankle'] directly. v1 NPZ has only joint angles + hip_z;
    # we approximate shank by extending knee_rad from hip with a fixed thigh
    # length. This is only for prototype plotting on existing NPZ files;
    # Phase 2-fixed recordings will eventually carry full 3D.
    THIGH_LEN_M = 0.45    # nominal adult
    SHANK_LEN_M = 0.43

    hip_z = z["left_hip_z"] if side_upper == "L" else z["right_hip_z"]
    knee_rad = z["left_knee_rad"] if side_upper == "L" else z["right_knee_rad"]
    hip_rad = z["left_hip_rad"] if side_upper == "L" else z["right_hip_rad"]

    # Approximate knee/ankle 3D in camera frame (very rough).
    # Camera-frame X is lateral, Y is down, Z is forward (= horizontal distance).
    # NPZ stores joint angles already in RADIANS (left_hip_rad / left_knee_rad
    # — Codex P2). Do NOT re-convert.
    side_x = 0.10 if side_upper == "L" else -0.10
    knee_x = np.full(n, side_x, dtype=np.float64)
    knee_y = np.full(n, 0.50, dtype=np.float64)          # rough hip Y in cam
    knee_z = hip_z + THIGH_LEN_M * np.cos(hip_rad)        # forward
    # shank attach (alpha along knee→ankle, ankle assumed below knee)
    ankle_x = knee_x
    ankle_y = knee_y + SHANK_LEN_M    # below knee in camera Y (down)
    ankle_z = knee_z + SHANK_LEN_M * np.cos(hip_rad - knee_rad)

    knee_3d = np.column_stack([knee_x, knee_y, knee_z])
    ankle_3d = np.column_stack([ankle_x, ankle_y, ankle_z])

    p_shank = np.array([
        shank_attach_point(knee_3d[i], ankle_3d[i], geometry.shank_attach_alpha)
        for i in range(n)
    ])

    p_anchor = (geometry.P_anchor_left if side_upper == "L"
                else geometry.P_anchor_right)
    ell_free = np.array([free_cable_length(p_anchor, p_shank[i]) for i in range(n)])

    # Synthetic pretension reference. Try the requested frame; if its ℓ_free
    # is NaN (joint angle reconstruction failed at that instant), fall back
    # to the nearest finite-ℓ_free frame so the trace doesn't go all-NaN.
    finite_idx = np.where(np.isfinite(ell_free))[0]
    if len(finite_idx) == 0:
        ref_idx = pretension_frame_idx
    else:
        # nearest finite frame to the requested index
        ref_idx = int(finite_idx[np.argmin(np.abs(finite_idx - pretension_frame_idx))])
    ref = CableReference(
        side=side_upper,
        theta_motor_ref=0.0,
        ell_free_ref=float(ell_free[ref_idx]),
        P_shank_ref=p_shank[ref_idx],
    )
    pretension_frame_idx = ref_idx  # for the return dict

    delta_ell = ell_free - ref.ell_free_ref + geometry.slack_margin
    theta_motor = delta_ell / geometry.r_pulley

    return {
        "t_s": t,
        "ell_free_m": ell_free,
        "delta_ell_m": delta_ell,
        "theta_motor_rad": theta_motor,
        "p_shank_3d": p_shank,
        "ref_frame_idx": pretension_frame_idx,
        "ref_ell_free_m": ref.ell_free_ref,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--config", default="configs/walker_geometry.yaml")
    ap.add_argument("--side", choices=["L", "R"], default="L")
    args = ap.parse_args()
    geom = WalkerGeometry.from_yaml(args.config)
    trace = trace_from_npz(args.npz, geom, args.side)
    print(f"frames: {len(trace['t_s'])}")
    print(f"ref ℓ_free at frame {trace['ref_frame_idx']}: {trace['ref_ell_free_m']:.3f} m")
    finite = np.isfinite(trace["ell_free_m"])
    if finite.any():
        e = trace["ell_free_m"][finite]
        print(f"ℓ_free range: {e.min():.3f}..{e.max():.3f} m (mean {e.mean():.3f})")
        d = trace["delta_ell_m"][finite]
        print(f"Δℓ range: {d.min()*1000:.1f}..{d.max()*1000:.1f} mm")
        th = trace["theta_motor_rad"][finite]
        print(f"θ_motor range: {th.min():.3f}..{th.max():.3f} rad ({np.rad2deg(th.max()-th.min()):.1f}° peak-to-peak)")
