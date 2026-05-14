"""Simplest possible Kalman Filter demo — 직접 실행하면서 매 step 봄.

Scenario: 차가 10 m/s 로 직선 운동. GPS 가 매 0.1초 위치를 측정 (±1m 노이즈).
Goal: GPS 측정값 만으로 위치 + 속도 동시 추정.

Run:
    python3 scripts/simple_kf_demo.py            # 매 step 콘솔 출력
    python3 scripts/simple_kf_demo.py --plot     # PNG 저장
    python3 scripts/simple_kf_demo.py --Q-scale 10   # Q 10배 → 더 reactive
    python3 scripts/simple_kf_demo.py --R-scale 10   # R 10배 → 더 smooth

Watch:
  - 'predicted pos' 과 'measured pos' 차이 = innovation y
  - Kalman gain K 의 변화 (처음 큼 → 줄어듦)
  - state covariance P 의 진화 (큰 uncertainty → 작아짐)
  - 추정 velocity가 시간에 따라 10 m/s 로 수렴

이거 이해하면 Plan D EKF 의 본질 80% 이해.
Plan D 는 여기에 추가:
  - 1D → 3D state (phi, omega, alpha 대신 pos, vel만)
  - Nonlinear h(x) → Jacobian H 매 frame 계산
  - 3-stage cascade
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def run_kalman_filter(
    duration_s: float = 10.0,
    dt: float = 0.1,
    true_velocity_mps: float = 10.0,
    measurement_noise_m: float = 1.0,
    q_scale: float = 1.0,
    r_scale: float = 1.0,
    verbose: bool = True,
):
    """Run a 2-state KF and optionally print every step."""
    n = int(duration_s / dt)
    t = np.arange(n) * dt

    # ── Ground truth ──────────────────────────────────────────────────
    truth_pos = true_velocity_mps * t

    # ── Noisy measurements ───────────────────────────────────────────
    rng = np.random.default_rng(0)
    z_obs = truth_pos + rng.normal(0, measurement_noise_m, n)

    # ── KF setup ─────────────────────────────────────────────────────
    # State x = [position, velocity]
    # Process model:  pos_{k+1} = pos_k + vel_k * dt
    #                 vel_{k+1} = vel_k                  (constant velocity)
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])

    # Measurement model: z = pos (we only measure position)
    H = np.array([[1.0, 0.0]])

    # Q (process noise covariance) — how much do we expect velocity to drift
    Q_base = np.array([[1e-4, 0.0],
                       [0.0,  1e-2]])
    Q = Q_base * q_scale

    # R (measurement noise covariance) — how noisy is the GPS
    R_base = np.array([[measurement_noise_m ** 2]])
    R = R_base * r_scale

    # Initial state — we don't know anything, so big uncertainty
    x = np.array([0.0, 0.0])                # initial: position 0, velocity 0
    P = np.array([[10.0,  0.0],             # large covariance
                  [0.0,  10.0]])

    # ── Run ───────────────────────────────────────────────────────────
    history = {
        "t": t, "truth_pos": truth_pos, "truth_vel": np.full(n, true_velocity_mps),
        "z_obs": z_obs,
        "x_pos": np.zeros(n), "x_vel": np.zeros(n),
        "P_pos": np.zeros(n), "P_vel": np.zeros(n),
        "K_pos": np.zeros(n), "K_vel": np.zeros(n),
        "innov": np.zeros(n),
    }

    if verbose:
        print(f"Setup: dt={dt}s, true_vel={true_velocity_mps}m/s, "
              f"meas_noise={measurement_noise_m}m, Q×{q_scale}, R×{r_scale}")
        print(f"\n{'k':>3} {'t':>6} {'meas':>7} {'pred_pos':>9} {'pred_vel':>9} "
              f"{'innov':>7} {'K_pos':>6} {'K_vel':>6} {'post_pos':>9} {'post_vel':>9}")
        print("-" * 110)

    for k in range(n):
        # ── PREDICT step ────────────────────────────────────────────
        x_prior = F @ x
        P_prior = F @ P @ F.T + Q

        # Predicted measurement (model says GPS should read this position)
        z_pred = H @ x_prior

        # ── UPDATE step (vision measurement comes in) ─────────────
        # Innovation: how surprised are we by the measurement?
        y = z_obs[k] - z_pred[0]

        # Innovation covariance + Kalman gain
        S = H @ P_prior @ H.T + R           # scalar (1×1)
        K = P_prior @ H.T @ np.linalg.inv(S)  # 2×1

        # Posterior
        x = x_prior + (K @ np.array([y])).flatten()
        P = (np.eye(2) - K @ H) @ P_prior

        # Record
        history["x_pos"][k] = x[0]; history["x_vel"][k] = x[1]
        history["P_pos"][k] = P[0, 0]; history["P_vel"][k] = P[1, 1]
        history["K_pos"][k] = K[0, 0]; history["K_vel"][k] = K[1, 0]
        history["innov"][k] = y

        if verbose and (k < 20 or k % 20 == 0):
            print(f"{k:>3} {t[k]:>6.2f} {z_obs[k]:>7.2f} "
                  f"{x_prior[0]:>9.2f} {x_prior[1]:>9.2f} "
                  f"{y:>+7.2f} {K[0,0]:>6.3f} {K[1,0]:>6.3f} "
                  f"{x[0]:>9.2f} {x[1]:>9.2f}")

    return history


def render_plot(h: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # 1. Position: truth vs measured vs estimated
    axes[0].plot(h["t"], h["truth_pos"], "k-", lw=2, label="truth", alpha=0.7)
    axes[0].plot(h["t"], h["z_obs"], "x", ms=3, color="tab:red", alpha=0.6,
                 label="GPS measured (noisy)")
    axes[0].plot(h["t"], h["x_pos"], "-", lw=1.2, color="tab:green",
                 label="KF posterior")
    axes[0].set_ylabel("position [m]")
    axes[0].legend(loc="upper left")
    axes[0].set_title("KF estimate vs ground truth (top is what you want)")
    axes[0].grid(alpha=0.3)

    # 2. Velocity: truth vs estimated
    axes[1].axhline(h["truth_vel"][0], color="k", lw=2, alpha=0.7, label="truth")
    axes[1].plot(h["t"], h["x_vel"], "-", lw=1.2, color="tab:green",
                 label="KF posterior")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].legend(loc="lower right")
    axes[1].set_title(
        f"Velocity converges to truth — final {h['x_vel'][-1]:.2f} m/s "
        f"(truth {h['truth_vel'][0]:.0f})"
    )
    axes[1].grid(alpha=0.3)

    # 3. Innovation
    axes[2].plot(h["t"], h["innov"], "-", lw=0.8, color="tab:orange")
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("innovation y [m]")
    axes[2].set_title(
        "Innovation = measurement − prediction (should center on 0)"
    )
    axes[2].grid(alpha=0.3)

    # 4. Kalman gain + covariance
    axes[3].plot(h["t"], h["K_pos"], "-", lw=0.8, color="tab:purple",
                 label="K[0] (position gain)")
    axes[3].plot(h["t"], h["K_vel"], "-", lw=0.8, color="tab:brown",
                 label="K[1] (velocity gain)")
    axes[3].plot(h["t"], h["P_vel"], "--", lw=0.8, color="tab:gray",
                 label="P[1,1] velocity uncertainty")
    axes[3].set_ylabel("Kalman gain / P_vel")
    axes[3].set_xlabel("t [s]")
    axes[3].set_title(
        "Kalman gain drops as P shrinks (KF gets more confident)"
    )
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\nSaved plot: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--true-velocity", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--Q-scale", type=float, default=1.0,
                    help=">1 = process noise larger = KF more reactive")
    ap.add_argument("--R-scale", type=float, default=1.0,
                    help=">1 = measurement noise larger = KF more conservative")
    ap.add_argument("--plot", default=None,
                    help="output PNG path (default: /tmp/simple_kf.png)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    h = run_kalman_filter(
        duration_s=args.duration,
        dt=args.dt,
        true_velocity_mps=args.true_velocity,
        measurement_noise_m=args.noise,
        q_scale=args.Q_scale,
        r_scale=args.R_scale,
        verbose=not args.quiet,
    )

    out = args.plot or "/tmp/simple_kf.png"
    render_plot(h, out)

    print(f"\n=== Summary ===")
    print(f"  final position estimate: {h['x_pos'][-1]:.2f} m  (truth {h['truth_pos'][-1]:.2f})")
    print(f"  final velocity estimate: {h['x_vel'][-1]:.2f} m/s (truth {h['truth_vel'][0]:.2f})")
    print(f"  innovation mean (last 30): {h['innov'][-30:].mean():+.3f} m (should be near 0)")
    print(f"  velocity P (last):         {h['P_vel'][-1]:.4f} (should be small)")
    print(f"\nTry:")
    print(f"  python3 scripts/simple_kf_demo.py --Q-scale 100   # Q crazy big → KF chases noise")
    print(f"  python3 scripts/simple_kf_demo.py --R-scale 0.01  # R tiny → KF trusts measurement blindly")
    print(f"  open {out}")


if __name__ == "__main__":
    main()
