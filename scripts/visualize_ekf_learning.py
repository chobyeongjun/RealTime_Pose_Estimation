"""Visualize Plan D EKF ω/φ learning timeline from an existing NPZ.

Headless (matplotlib agg). Saves three figures:
  1. hip signal (input) over time
  2. Hilbert instantaneous frequency vs EKF L1 ω trace vs final ω
  3. Per-cascade-level ω convergence; HS events marked

This is the visual answer to "Vision에서 ω, α 어떻게 구하는가" — shows
the three learning paths (Hilbert / EKF innovation / HS interval) on real
data, frame-by-frame.

Usage:
    python3 scripts/visualize_ekf_learning.py recordings/walking_*/walking_*.npz
    python3 scripts/visualize_ekf_learning.py recordings/walking_*/walking_*.npz --out /tmp/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from perception.plan_d_prototype import PlanDPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out) if args.out else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    t = z["t_s"]
    n = len(t)
    valid = z["valid"] if "valid" in z.files else np.ones(n, bool)
    schema = int(z["schema_version"]) if "schema_version" in z.files else 1

    # Hip signal — prefer hip_vertical_m (v2), else hip_z_world_m (v1)
    if schema >= 2 and "hip_vertical_m" in z.files:
        hip = np.asarray(z["hip_vertical_m"], dtype=np.float64)
        signal_name = "hip_vertical_m (world-up projection)"
    else:
        hip = np.asarray(z["hip_z_world_m"], dtype=np.float64)
        signal_name = "hip_z_world_m (ZED Z, may be horizontal)"

    fs = 1.0 / float(np.median(np.diff(t)))
    print(f"Loaded {n} frames @ {fs:.1f} Hz, schema v{schema}")
    print(f"Hip signal: {signal_name}")

    # Build q (6 joints if v2, else 4 — best effort)
    if schema >= 2 and "left_thigh_inclination_rad" in z.files:
        q = np.column_stack([
            z["left_thigh_inclination_rad"],
            z["left_knee_rad"],
            z["left_shank_inclination_rad"],
            z["right_thigh_inclination_rad"],
            z["right_knee_rad"],
            z["right_shank_inclination_rad"],
        ])
    else:
        q = np.column_stack([
            z["left_hip_rad"], z["left_knee_rad"],
            z["right_hip_rad"], z["right_knee_rad"],
        ])

    # Run Plan D, recording per-frame trace
    predictor = PlanDPredictor(
        n_joints=q.shape[1], fs_hz=fs, initial_omega=2 * np.pi,
    )
    sigma = np.full(q.shape[1], 0.05, dtype=np.float64)

    phi_trace = np.zeros(n)
    omega_trace = np.zeros(n)
    alpha_trace = np.zeros(n)
    level_trace = np.zeros(n, dtype=int)
    stride_trace = np.zeros(n, dtype=int)
    template_trace = np.zeros(n)

    for i in range(n):
        if valid[i]:
            try:
                predictor.feed(
                    t_now=float(t[i]), q=q[i],
                    sigma_per_joint=sigma,
                    hip_z_world_m=float(hip[i]),
                )
            except Exception:
                pass
        phi_trace[i] = predictor.phi
        omega_trace[i] = predictor.omega
        alpha_trace[i] = getattr(predictor, "alpha", 0.0)
        level_trace[i] = int(predictor.level)
        stride_trace[i] = int(predictor.stride_count)
        template_trace[i] = float(predictor.template_touched_fraction)

    # ─── Figure 1: hip signal + EKF outputs (4 panels) ───────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, hip, lw=0.6, color="tab:blue")
    axes[0].set_ylabel(signal_name.split("(")[0].strip())
    axes[0].set_title(f"Hip signal fed to Plan D ({signal_name})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.degrees(phi_trace) % 360, lw=0.6, color="tab:green")
    axes[1].set_ylabel("φ (deg, mod 360)")
    axes[1].set_title("EKF phase estimate (should be a sawtooth when walking)")
    axes[1].set_ylim(-10, 370)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, omega_trace, lw=0.8, color="tab:orange",
                 label="EKF ω")
    axes[2].axhline(2 * np.pi, color="gray", ls="--", alpha=0.5,
                    label="1 Hz reference (2π)")
    axes[2].axhline(-2 * np.pi, color="gray", ls=":", alpha=0.5,
                    label="−1 Hz (sign convention)")
    axes[2].set_ylabel("ω (rad/s)")
    axes[2].set_title("EKF cadence estimate — learned over time")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, level_trace, lw=1.2, color="tab:purple", label="cascade")
    ax3b = axes[3].twinx()
    ax3b.plot(t, stride_trace, lw=0.8, color="tab:red", label="strides")
    ax3b.plot(t, template_trace * 100, lw=0.8, color="tab:cyan",
              label="template% × 1")
    axes[3].set_ylabel("L (cascade)", color="tab:purple")
    ax3b.set_ylabel("strides / template fill (%)")
    axes[3].set_xlabel("t (s)")
    axes[3].set_title("Cascade level, stride count, template fill — convergence trace")
    axes[3].set_yticks([1, 2, 3])
    axes[3].grid(alpha=0.3)
    fig.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.00))

    plt.tight_layout()
    fig1_path = out_dir / "ekf_learning_overview.png"
    fig.savefig(fig1_path, dpi=110)
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # ─── Figure 2: omega zoom — first 15 s vs final 15 s ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    mask_start = t <= min(15.0, t[-1])
    mask_end = t >= max(0, t[-1] - 15.0)
    axes[0].plot(t[mask_start], omega_trace[mask_start], lw=0.7, color="tab:orange")
    axes[0].axhline(2 * np.pi, ls="--", color="gray", alpha=0.5)
    axes[0].axhline(-2 * np.pi, ls=":", color="gray", alpha=0.5)
    axes[0].set_title("ω learning — first 15 s (cold-start)")
    axes[0].set_ylabel("ω (rad/s)"); axes[0].set_xlabel("t (s)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t[mask_end], omega_trace[mask_end], lw=0.7, color="tab:orange")
    axes[1].axhline(2 * np.pi, ls="--", color="gray", alpha=0.5)
    axes[1].axhline(-2 * np.pi, ls=":", color="gray", alpha=0.5)
    axes[1].set_title("ω steady-state — last 15 s")
    axes[1].set_xlabel("t (s)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig2_path = out_dir / "ekf_omega_zoom.png"
    fig.savefig(fig2_path, dpi=110)
    plt.close(fig)
    print(f"Saved {fig2_path}")

    # Console summary
    valid_mask = np.isfinite(omega_trace) & (t > 5.0)
    if valid_mask.any():
        omega_steady = omega_trace[valid_mask]
        print()
        print("=== ω learning summary ===")
        print(f"  initial:       {omega_trace[0]:+.3f} rad/s  ({omega_trace[0]/(2*np.pi):+.3f} Hz)")
        print(f"  median (t>5s): {np.median(omega_steady):+.3f} rad/s  ({np.median(omega_steady)/(2*np.pi):+.3f} Hz)")
        print(f"  |ω| median:    {abs(np.median(omega_steady)):.3f} rad/s")
        print(f"  final:         {omega_trace[-1]:+.3f} rad/s")
        print(f"  cascade final: L{level_trace[-1]}")
        print(f"  strides total: {stride_trace[-1]}")
        print(f"  template fill: {template_trace[-1]*100:.1f}%")


if __name__ == "__main__":
    main()
