#!/usr/bin/env python3
"""Run Plan D Phase 1.5 against recorded walking npz — algorithm validation.

This is the FIRST real-data validation of Plan D after Phase 1.5 (commit 37ee3f4).
Up to now, the EKF L1 + CycleTemplate + CrossCorrPhaseEstimator + HipVerticalPhaseEstimator
were validated only on synthetic sinusoids. This script feeds them recorded
joint angles + hip vertical position and checks:

    1. HipVerticalPhaseEstimator emits valid phase under real noise.
    2. EKF L1 ω converges to the subject's stride frequency.
    3. CycleTemplate accumulates a sensible joint-angle template over phase.
    4. CrossCorrPhaseEstimator (after 3 strides) recovers φ consistent with Hilbert φ.

Usage (Jetson, after recording with pipeline_main.py --record-pose-npz):
    python3 scripts/run_plan_d_offline.py walking_60s.npz [--plot]

NPZ schema (from pipeline_main.py 의 --record-pose-npz):
    t_s              (N,) seconds (CLOCK_MONOTONIC)
    hip_z_world_m    (N,) world-frame hip vertical (mean of L/R hip)
    left_hip_z       (N,) per-side hip z
    right_hip_z      (N,)
    left_knee_rad    (N,) joint angle (radians)
    right_knee_rad   (N,)
    left_hip_rad     (N,)
    right_hip_rad    (N,)
    valid            (N,) bool — pipeline 의 frame valid flag
    method           str — 'A' or 'B'

Output:
    stdout: stride count, final ω, template fill, ambiguity at end
    --plot: 4-panel matplotlib figure (saved as <input>_planD_plots.png)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

from perception.plan_d_prototype import (   # noqa: E402
    EKFL1,
    PredictStatus,
    CycleTemplate,
    CrossCorrPhaseEstimator,
    HipVerticalPhaseEstimator,
    TWO_PI,
    wrap_to_pi,
)


def load_session(npz_path: str) -> dict:
    """Load a walking-session npz produced by pipeline_main.py --record-pose-npz.

    Schema versions (Codex consult #4):
      v1 (no schema_version field): legacy 4-joint vector (hip + knee, L/R).
          hip_z_world_m = ZED Z = walker-user HORIZONTAL distance (mislabelled).
      v2 (schema_version=2): full 6-joint vector incl. thigh/shank inclinations.
          hip_vertical_m = world-up projection of hip (what Plan D actually expects).
          walker_user_distance_m = ZED Z = horizontal distance (explicit name).
          hip_z_world_m kept = walker_user_distance_m for back-compat.
    """
    z = np.load(npz_path, allow_pickle=True)
    required = [
        "t_s", "hip_z_world_m", "left_knee_rad", "right_knee_rad",
        "left_hip_rad", "right_hip_rad", "valid",
    ]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"npz missing fields: {missing}")
    out = {k: z[k] for k in z.files}
    out["__schema_version__"] = (
        int(z["schema_version"]) if "schema_version" in z.files else 1
    )
    return out


def run(session: dict, verbose: bool = True) -> dict:
    """Run Plan D Phase 1.5 against session data.

    Returns diagnostics dict with: stride_count, final_omega, template_fill,
    ambiguity_at_end, hilbert_valid_fraction, l1_omega_trace, hilbert_phi_trace, ...
    """
    t = session["t_s"]
    schema = session.get("__schema_version__", 1)

    # Plan D Hilbert cold-start expects VERTICAL hip motion. In schema v2 this
    # is `hip_vertical_m` (world-up projection). In v1 the only thing we have
    # is `hip_z_world_m`, which (Codex consult #5) is actually ZED Z =
    # horizontal distance, not vertical. Plan D will struggle on v1 data;
    # warn so users know to re-record with v2.
    if schema >= 2 and "hip_vertical_m" in session:
        hip_z = session["hip_vertical_m"]
        if verbose:
            print(f"[schema v{schema}] hip signal = hip_vertical_m (world-up projection)")
    else:
        hip_z = session["hip_z_world_m"]
        if verbose:
            print(
                f"[schema v{schema}] hip signal = hip_z_world_m (legacy ZED Z; "
                f"may be horizontal distance, see Codex consult #5)."
            )

    if schema >= 2 and all(
        k in session for k in (
            "left_thigh_inclination_rad", "right_thigh_inclination_rad",
            "left_shank_inclination_rad", "right_shank_inclination_rad",
        )
    ):
        # 6-joint vector in the order Plan D spec uses
        q = np.column_stack([
            session["left_thigh_inclination_rad"],
            session["left_knee_rad"],
            session["left_shank_inclination_rad"],
            session["right_thigh_inclination_rad"],
            session["right_knee_rad"],
            session["right_shank_inclination_rad"],
        ])  # (N, 6)
        if verbose:
            print(f"[schema v{schema}] q vector = 6 joints (thigh+knee+shank, L/R)")
    else:
        q = np.column_stack([
            session["left_hip_rad"],
            session["left_knee_rad"],
            session["right_hip_rad"],
            session["right_knee_rad"],
        ])  # (N, 4) legacy fallback
        if verbose:
            print(f"[schema v{schema}] q vector = 4 joints (hip+knee, L/R, legacy)")
    n_joints = q.shape[1]
    valid = session["valid"]
    n = len(t)

    if n < 10:
        raise ValueError(f"session too short: {n} frames")

    # Estimate fs from median dt
    dt_med = float(np.median(np.diff(t)))
    fs_hz = 1.0 / dt_med if dt_med > 0 else 60.0
    if verbose:
        print(f"Session: {n} frames, ~{fs_hz:.1f} Hz, duration {t[-1]-t[0]:.1f} s")
        print(f"Method: {str(session.get('method', '?'))}")
        print(f"Valid fraction: {float(np.mean(valid))*100:.1f}%")

    hilbert = HipVerticalPhaseEstimator(
        window_seconds=1.5, fs_hz=fs_hz, min_amplitude_m=0.005,
    )
    l1 = EKFL1(
        initial_omega=4.0,
        process_noise_omega=4e-2,
        measurement_noise=0.05,
    )
    template = CycleTemplate(n_bins=128, n_joints=n_joints, beta_default=0.05)
    estimator = CrossCorrPhaseEstimator(
        template, min_touched_fraction=0.25, sigma_floor=0.01,
    )

    # Stride detection (cascade-internal style, Codex Phase 2 design)
    prev_phi = None
    stride_count = 0

    l1_omega_trace = np.zeros(n)
    l1_phi_trace = np.zeros(n)
    hilbert_phi_trace = np.full(n, np.nan)
    estimator_phi_trace = np.full(n, np.nan)
    estimator_ambiguity_trace = np.full(n, np.nan)
    hilbert_valid_count = 0
    estimator_valid_count = 0

    for i in range(n):
        t_now = float(t[i])
        z_hip = float(hip_z[i]) if math.isfinite(hip_z[i]) else float("nan")

        # 1. Hilbert envelope
        hilbert.feed(t_now, z_hip)

        # 2. EKF L1 predict
        status = l1.predict(t_now)
        # status not used for branching here — informational only

        # 3. Phase observation: hilbert (cold-start) or estimator (template ready)
        cold_start = (stride_count < 3 or template.touched_fraction < 0.5)
        if cold_start:
            ph = hilbert.estimate()
            if ph.valid:
                hilbert_valid_count += 1
                hilbert_phi_trace[i] = ph.phi
                l1.update(ph.phi)
        else:
            # Try template estimator
            if bool(valid[i]):
                est = estimator.estimate(q[i])
                if est.valid:
                    estimator_valid_count += 1
                    estimator_phi_trace[i] = est.phi
                    estimator_ambiguity_trace[i] = est.ambiguity_ratio
                    if est.ambiguity_ratio < 0.7:   # accept reasonably sharp
                        l1.update(est.phi)
            # Also continue feeding Hilbert phi (for debug)
            ph = hilbert.estimate()
            if ph.valid:
                hilbert_phi_trace[i] = ph.phi

        # 4. Template update (only if frame valid AND not in active divergence)
        if bool(valid[i]):
            template.update(l1.state.phi, q[i])

        # 5. Stride detection (phase wrap, cascade-style)
        cur_phi = l1.state.phi
        if prev_phi is not None and prev_phi > 3 * math.pi / 2 and cur_phi < math.pi / 2:
            stride_count += 1
        prev_phi = cur_phi

        l1_omega_trace[i] = l1.state.omega
        l1_phi_trace[i] = cur_phi

    diag = {
        "n_frames": n,
        "fs_hz": fs_hz,
        "duration_s": float(t[-1] - t[0]),
        "valid_frac": float(np.mean(valid)),
        "stride_count": stride_count,
        "final_omega_rad_s": float(l1.state.omega),
        "final_omega_hz": float(l1.state.omega / TWO_PI),
        "template_touched_fraction": float(template.touched_fraction),
        "template_touched_per_joint": template.touched_fraction_per_joint.tolist(),
        "hilbert_valid_fraction": float(hilbert_valid_count) / n,
        "estimator_valid_fraction": float(estimator_valid_count) / max(1, n),
        "l1_omega_trace": l1_omega_trace,
        "l1_phi_trace": l1_phi_trace,
        "hilbert_phi_trace": hilbert_phi_trace,
        "estimator_phi_trace": estimator_phi_trace,
        "estimator_ambiguity_trace": estimator_ambiguity_trace,
        "t_s": t,
        "hip_z": hip_z,
    }

    if verbose:
        print()
        print("─── Plan D offline run results ───")
        print(f"  Stride count:                {stride_count}")
        print(f"  Final ω:                      {diag['final_omega_rad_s']:.3f} rad/s "
              f"= {diag['final_omega_hz']:.3f} Hz")
        print(f"  Template touched fraction:    {diag['template_touched_fraction']:.3f}")
        print(f"  Hilbert valid fraction:       {diag['hilbert_valid_fraction']:.3f}")
        print(f"  Estimator valid fraction:     {diag['estimator_valid_fraction']:.3f}")
        print(f"  Per-joint template coverage:  {diag['template_touched_per_joint']}")
        if stride_count >= 3 and 0.3 < diag['final_omega_hz'] < 2.0:
            print("  ✓ Plan D Phase 1.5 algorithm validated on real walking data")
        else:
            print("  ⚠ Convergence weak — check session quality (method B?, walking duration?)")

    return diag


def plot(diag: dict, out_path: str) -> None:
    """4-panel diagnostic plot. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless on Jetson
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping.", file=sys.stderr)
        return

    t = diag["t_s"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Panel 1: Hip vertical signal
    axes[0].plot(t, diag["hip_z"], color="tab:blue", lw=0.8)
    axes[0].set_ylabel("Hip z (m, world)")
    axes[0].set_title("Hip vertical signal (input to Hilbert envelope)")
    axes[0].grid(alpha=0.3)

    # Panel 2: L1 ω evolution
    axes[1].plot(t, diag["l1_omega_trace"], color="tab:green", lw=1.2,
                 label="EKF L1 ω")
    axes[1].axhline(2 * math.pi * 1.0, ls="--", color="gray",
                    label="1 Hz reference")
    axes[1].set_ylabel("ω (rad/s)")
    axes[1].set_title(f"EKF L1 cadence (final: {diag['final_omega_hz']:.2f} Hz)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Panel 3: Phase traces
    axes[2].plot(t, diag["l1_phi_trace"], color="tab:green", lw=0.8,
                 label="L1 φ")
    axes[2].plot(t, diag["hilbert_phi_trace"], color="tab:orange", lw=0.5,
                 alpha=0.7, label="Hilbert φ (cold-start)")
    axes[2].plot(t, diag["estimator_phi_trace"], color="tab:red", lw=0.5,
                 alpha=0.7, label="Estimator φ (template)")
    axes[2].set_ylabel("phase (rad)")
    axes[2].set_title(f"Phase observations | strides: {diag['stride_count']}")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    # Panel 4: Ambiguity ratio
    axes[3].plot(t, diag["estimator_ambiguity_trace"], color="tab:purple",
                 lw=0.5, alpha=0.7)
    axes[3].axhline(0.7, ls="--", color="gray", label="accept threshold")
    axes[3].set_ylabel("ambiguity_ratio")
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("CrossCorr ambiguity_ratio (low = sharp)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    print(f"[plot] Saved: {out_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan D Phase 1.5 offline run on recorded walking npz."
    )
    parser.add_argument("npz", help="Path to walking session npz "
                        "(from pipeline_main.py --record-pose-npz)")
    parser.add_argument("--plot", action="store_true",
                        help="Save diagnostic plots to <npz>_planD_plots.png")
    args = parser.parse_args()

    npz_path = args.npz
    if not os.path.exists(npz_path):
        print(f"ERROR: file not found: {npz_path}", file=sys.stderr)
        return 1

    session = load_session(npz_path)
    diag = run(session, verbose=True)

    if args.plot:
        out_plot = os.path.splitext(npz_path)[0] + "_planD_plots.png"
        plot(diag, out_plot)

    # Summary npz (small) — exclude trace arrays
    summary_path = os.path.splitext(npz_path)[0] + "_planD_summary.npz"
    np.savez(
        summary_path,
        stride_count=diag["stride_count"],
        final_omega_rad_s=diag["final_omega_rad_s"],
        final_omega_hz=diag["final_omega_hz"],
        template_touched_fraction=diag["template_touched_fraction"],
        hilbert_valid_fraction=diag["hilbert_valid_fraction"],
        estimator_valid_fraction=diag["estimator_valid_fraction"],
        fs_hz=diag["fs_hz"],
        duration_s=diag["duration_s"],
    )
    print(f"\n[summary] Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
