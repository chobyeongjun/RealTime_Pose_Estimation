"""Generate realistic synthetic walking signals that match Plan D EKF assumptions.

Unlike the simple ±cos(phi) that fails Plan D's Hilbert envelope, real
gait has:
  - Asymmetric stance (60%) and swing (40%) phases
  - Hip vertical: MINIMUM at heel-strike (HS), MAXIMUM around mid-swing
  - Knee flexion: small in stance, large peak in swing (≈ 60°)
  - Hip flexion: peaks slightly after toe-off

This module produces signals that the existing Hilbert cold-start +
EKF cascade can lock onto, so we can validate EKF behaviour on Mac
without needing a Jetson walking session.

Public API:
    generate_walking_session(...)  → (t, hip_vertical, q_6_rad, true_phi)

Tests in tests/test_synth_walking.py verify the waveform shape.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def gait_waveform_hip_vertical(phi: np.ndarray) -> np.ndarray:
    """Realistic hip vertical motion as a function of gait phase.

    Convention: phi=0 is left heel-strike (HS_L). Hip is at MINIMUM at HS,
    rises during mid-stance (≈ 30% GC), falls again into terminal stance,
    second minimum around right HS (phi=π), rises again in left swing.

    Healthy adult: hip vertical motion ~ 30-50 mm peak-to-peak (we use 35 mm).
    """
    # Two oscillations per gait cycle (one for each stance) with offset.
    # Modeled as the difference of two -cos waves with phase shift.
    # Magnitude ≈ 17.5 mm amplitude → 35 mm pk-pk.
    return 0.0175 * (
        -np.cos(2 * phi)             # 2nd harmonic dominant (two stances per cycle)
        + 0.25 * np.cos(phi)         # 1st harmonic asymmetry
    )


def gait_waveform_knee_flexion(phi: np.ndarray, side: str = "L") -> np.ndarray:
    """Knee flexion in radians as a function of phase.

    Healthy adult knee flexion:
      - Mid-stance: ~ 10-15° (slight bend)
      - Pre-swing: ~ 35° (knee bend before toe-off)
      - Peak swing: ~ 60° (max flexion mid-swing)
      - Late swing: ~ 5° (extension before HS)

    Right side phase-shifted by π.
    """
    offset = 0.0 if side.upper() == "L" else np.pi
    phi_local = (phi + offset) % (2 * np.pi)
    # Peak around 75-80% GC (mid-swing). Stance < swing.
    # Roughly: knee = baseline + 60° × sin²(phi_local × scaling)
    swing_window = np.clip((phi_local - 0.6 * 2 * np.pi) / (0.4 * 2 * np.pi), 0, 1)
    swing_peak = 60.0 * np.pi / 180.0
    stance_baseline = np.where(
        phi_local < 0.6 * 2 * np.pi,
        # Stance: small flexion peak at ~15% GC (loading response)
        15.0 * np.pi / 180.0 * np.exp(-((phi_local - 0.15 * 2 * np.pi) / 0.6) ** 2),
        0.0,
    )
    swing_curve = swing_peak * np.sin(np.pi * swing_window) ** 2
    return stance_baseline + swing_curve


def gait_waveform_thigh_inclination(phi: np.ndarray, side: str = "L") -> np.ndarray:
    """Thigh inclination (hip flexion proxy) in radians.

    Hip flexion swings ~ 20-30° peak-to-peak.
    Maximum flexion at terminal swing (just before HS).
    Minimum (extension) at toe-off (~60% GC).
    """
    offset = 0.0 if side.upper() == "L" else np.pi
    return 0.25 * np.cos((phi + offset + np.pi) % (2 * np.pi) - np.pi)


def gait_waveform_shank_inclination(phi: np.ndarray, side: str = "L") -> np.ndarray:
    """Shank inclination (ankle dorsiflexion proxy) in radians.

    Shank rotates forward during swing, backward during stance push-off.
    Peak forward ~ mid-swing.
    """
    offset = 0.0 if side.upper() == "L" else np.pi
    phi_local = (phi + offset) % (2 * np.pi)
    return 0.30 * np.sin(phi_local)


def generate_walking_session(
    fs_hz: float = 60.0,
    duration_s: float = 30.0,
    cadence_hz: float = 1.0,
    noise_hip_m: float = 0.005,
    noise_joint_rad: float = 0.02,
    cadence_drift_amplitude: float = 0.0,
    cadence_drift_period_s: float = 20.0,
    seed: int = 0,
    side: str = "L",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a realistic walking session.

    Returns:
        t            (N,)    time stamps [s]
        hip_vertical (N,)    hip vertical position [m, oscillating around 0.5]
        q_6joints    (N, 6)  joint angles in Plan D spec order [rad]:
                             [L_thigh, L_knee, L_shank, R_thigh, R_knee, R_shank]
        true_phi     (N,)    ground-truth phase used for generation [rad]

    Args:
        cadence_drift_amplitude: peak cadence drift relative to nominal
            (e.g., 0.05 → ±5% slow oscillation). 0 = constant cadence.
        cadence_drift_period_s: drift period.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs_hz))
    t = np.arange(n) / fs_hz

    omega_nominal = 2.0 * np.pi * cadence_hz
    if cadence_drift_amplitude > 0:
        # Smooth slow drift
        drift = 1.0 + cadence_drift_amplitude * np.sin(
            2 * np.pi * t / cadence_drift_period_s
        )
        omega_t = omega_nominal * drift
    else:
        omega_t = np.full(n, omega_nominal)

    # Integrate omega to get true phase
    dt = 1.0 / fs_hz
    true_phi = np.cumsum(omega_t) * dt
    true_phi = true_phi % (2 * np.pi)

    # Generate signals
    hip_v_clean = gait_waveform_hip_vertical(true_phi) + 0.5
    hip_v = hip_v_clean + rng.normal(0.0, noise_hip_m, size=n)

    q = np.zeros((n, 6), dtype=np.float64)
    q[:, 0] = gait_waveform_thigh_inclination(true_phi, "L") + rng.normal(0, noise_joint_rad, n)
    q[:, 1] = gait_waveform_knee_flexion(true_phi, "L") + rng.normal(0, noise_joint_rad, n)
    q[:, 2] = gait_waveform_shank_inclination(true_phi, "L") + rng.normal(0, noise_joint_rad, n)
    q[:, 3] = gait_waveform_thigh_inclination(true_phi, "R") + rng.normal(0, noise_joint_rad, n)
    q[:, 4] = gait_waveform_knee_flexion(true_phi, "R") + rng.normal(0, noise_joint_rad, n)
    q[:, 5] = gait_waveform_shank_inclination(true_phi, "R") + rng.normal(0, noise_joint_rad, n)

    return t, hip_v, q, true_phi


if __name__ == "__main__":
    """Render an example session for sanity inspection."""
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--cadence", type=float, default=1.0)
    ap.add_argument("--out", default="/tmp/synth_walking_preview.png")
    args = ap.parse_args()

    t, hip, q, true_phi = generate_walking_session(
        duration_s=args.duration, cadence_hz=args.cadence, seed=0,
    )

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, hip - 0.5, lw=0.8)
    axes[0].set_ylabel("hip_vertical − 0.5 [m]")
    axes[0].set_title(f"Synthetic walking: {args.cadence:.1f} Hz, {args.duration:.0f} s")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].plot(t, np.degrees(q[:, 0]), label="L thigh inc")
    axes[1].plot(t, np.degrees(q[:, 3]), label="R thigh inc", alpha=0.6)
    axes[1].set_ylabel("thigh inclination [°]")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(t, np.degrees(q[:, 1]), label="L knee flex")
    axes[2].plot(t, np.degrees(q[:, 4]), label="R knee flex", alpha=0.6)
    axes[2].set_ylabel("knee flexion [°]")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    axes[3].plot(t, (np.degrees(true_phi) % 360))
    axes[3].set_ylabel("true φ (deg, mod 360)")
    axes[3].set_xlabel("t [s]")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"Saved {out}")
