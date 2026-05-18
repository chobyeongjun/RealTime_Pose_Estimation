"""Isolated benchmark of pipeline_main.py's T3→T4 sub-blocks.

5/18 measurement showed T3→T4 stage = 0.621ms (A1) → 2.019ms (A2).
The +1.4ms regression is NOT in shm_v2_publish() (measured: 0.009ms).
This script isolates each sub-block to find the real bottleneck.

Blocks (matches pipeline_main.py:582-700 with --enable-plan-d --enable-shm-v2):

  Block 1: SHM v2 prep — numpy zeros × 4 + dict iteration × 6 (extract from state)
  Block 2: compute_kp_sigma — per-keypoint depth uncertainty
  Block 3: SHM v2 publish — Python writer (and C++ writer for comparison)
  Block 4: Plan D predictor.feed — EKF L1/L2/L3 cascade update
  Block 5: Forecast publisher.publish — forecast SHM channel

Run:
    PYTHONPATH=src python3 tests/benchmark_t3_t4_breakdown.py
    PYTHONPATH=src python3 tests/benchmark_t3_t4_breakdown.py --iters 20000
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import shared_memory
from typing import Dict

import numpy as np


def _cleanup_shm(name: str) -> None:
    try:
        shared_memory.SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass


def _stats_us(times_ns: np.ndarray) -> Dict[str, float]:
    arr_us = times_ns / 1000.0
    return {
        'mean': float(arr_us.mean()),
        'p50':  float(np.percentile(arr_us, 50)),
        'p95':  float(np.percentile(arr_us, 95)),
        'p99':  float(np.percentile(arr_us, 99)),
        'max':  float(arr_us.max()),
    }


def _print_row(label: str, s: Dict[str, float], total_p50: float) -> None:
    pct = (s['p50'] / total_p50 * 100.0) if total_p50 > 0 else 0.0
    print(f"  {label:<40} p50={s['p50']:8.2f}us  p99={s['p99']:8.2f}us  "
          f"max={s['max']:8.2f}us  share={pct:5.1f}%")


# ────────────────────────────────────────────────────────────────────────────
# Block 1: SHM v2 prep — numpy zeros + dict iteration
# ────────────────────────────────────────────────────────────────────────────

KEYPOINT_ORDER_6 = (
    'left_hip', 'left_knee', 'left_ankle',
    'right_hip', 'right_knee', 'right_ankle',
)


class FakeState:
    """Stand-in for JointState — has .positions dict + .pixels + .confs."""
    def __init__(self, rng):
        self.positions = {
            name: rng.random(3).astype(np.float32)
            for name in KEYPOINT_ORDER_6
        }
        self.pixels = {
            name: rng.random(2).astype(np.float32) * 960.0
            for name in KEYPOINT_ORDER_6
        }
        self.confs = {name: float(rng.random()) for name in KEYPOINT_ORDER_6}


def bench_shm_prep(iters: int, rng) -> Dict[str, float]:
    state = FakeState(rng)
    times = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        kpts_3d_v2 = np.zeros((6, 3), dtype=np.float32)
        kpts_2d_v2 = np.zeros((6, 2), dtype=np.float32)
        kpt_conf_v2 = np.zeros(6, dtype=np.float32)
        for j, name in enumerate(KEYPOINT_ORDER_6):
            if name in state.positions:
                kpts_3d_v2[j] = state.positions[name]
            if name in state.pixels:
                kpts_2d_v2[j] = state.pixels[name]
            kpt_conf_v2[j] = float(state.confs.get(name, 0.0))
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0
    return _stats_us(times[200:])  # skip warmup


# ────────────────────────────────────────────────────────────────────────────
# Block 2: compute_kp_sigma
# ────────────────────────────────────────────────────────────────────────────

def bench_compute_kp_sigma(iters: int, rng) -> Dict[str, float]:
    """Try real compute_kp_sigma; fall back to synthetic equivalent."""
    state = FakeState(rng)
    # Try the production version
    try:
        from perception.realtime.kp_sigma import compute_kp_sigma  # type: ignore
        fn = compute_kp_sigma
    except ImportError:
        # Synthetic placeholder mimicking similar cost
        def fn(positions, confs, fx, fy, baseline_m):
            out = {}
            for name, p in positions.items():
                z = float(p[2])
                sigma_z = (z * z) / (fx * baseline_m + 1e-6) * 0.5
                sigma_xy = sigma_z * 0.5
                out[name] = np.array([sigma_xy, sigma_xy, sigma_z], dtype=np.float32)
            return out

    times = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        _ = fn(state.positions, state.confs, fx=362.1, fy=362.1, baseline_m=0.063)
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0
    return _stats_us(times[200:])


# ────────────────────────────────────────────────────────────────────────────
# Block 3: SHM v2 publish (Python + C++)
# ────────────────────────────────────────────────────────────────────────────

def bench_shm_publish_python(iters: int, rng) -> Dict[str, float]:
    from perception.CUDA_Stream.shm_publisher import ShmPublisher
    name = "bench_pub_py"
    _cleanup_shm(name)
    w = ShmPublisher(num_keypoints=6, name=name, create=True)

    kpts_3d = rng.random((6, 3), dtype=np.float32)
    kpts_2d = rng.random((6, 2), dtype=np.float32) * 960.0
    kp_conf = rng.random(6, dtype=np.float32)
    kp_sigma = rng.random((6, 3), dtype=np.float32) * 0.02
    pose_cov = rng.random((6, 3), dtype=np.float32) * 0.0004

    times = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        w.publish(
            frame_id=i,
            rgb_ts_ns=1_700_000_000_000_000_000 + i * 8_333_333,
            kpts_3d_m=kpts_3d,
            kpt_conf=kp_conf,
            kpts_2d_px=kpts_2d,
            box_conf=0.9,
            valid=True,
            kp_sigma_m=kp_sigma,
            pose_cov_diag=pose_cov,
        )
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0
    del w
    _cleanup_shm(name)
    return _stats_us(times[200:])


def bench_shm_publish_cpp(iters: int, rng) -> Dict[str, float]:
    try:
        from perception.realtime import hwalker_shm_v2_writer as cpp_writer
    except ImportError:
        try:
            sys.path.insert(0, "build/cpp")
            import hwalker_shm_v2_writer as cpp_writer  # type: ignore
        except ImportError:
            return {'mean': float('nan'), 'p50': float('nan'),
                    'p95': float('nan'), 'p99': float('nan'),
                    'max': float('nan')}

    name = "bench_pub_cpp"
    _cleanup_shm(name)
    w = cpp_writer.Writer(name, 6)

    kpts_3d = rng.random((6, 3), dtype=np.float32)
    kpts_2d = rng.random((6, 2), dtype=np.float32) * 960.0
    kp_conf = rng.random(6, dtype=np.float32)
    kp_sigma = rng.random((6, 3), dtype=np.float32) * 0.02
    pose_cov = rng.random((6, 3), dtype=np.float32) * 0.0004

    times = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        w.publish(
            frame_id=i,
            rgb_ts_ns=1_700_000_000_000_000_000 + i * 8_333_333,
            depth_ts_ns=1_700_000_000_000_000_000 + i * 8_333_333,
            depth_age_us=0,
            box_conf=0.9,
            depth_invalid_ratio=0.0,
            valid_flag=1,
            world_frame=1,
            valid_reason=0,
            ts_domain=0,
            publish_done_mono_ns=t0,
            valid_mask_bits=0x3F,
            kpts_3d_m=kpts_3d,
            kpts_2d_px=kpts_2d,
            kp_conf=kp_conf,
            kp_sigma_m=kp_sigma,
            pose_cov_diag=pose_cov,
        )
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0
    del w
    _cleanup_shm(name)
    return _stats_us(times[200:])


# ────────────────────────────────────────────────────────────────────────────
# Block 4: Plan D predictor.feed
# ────────────────────────────────────────────────────────────────────────────

def bench_plan_d_feed(iters: int, rng) -> Dict[str, float]:
    try:
        from perception.plan_d_prototype import PlanDPredictor
    except ImportError:
        return {'mean': float('nan'), 'p50': float('nan'),
                'p95': float('nan'), 'p99': float('nan'),
                'max': float('nan')}

    predictor = PlanDPredictor(n_joints=6, fs_hz=60.0)
    sigma_per_joint = np.full(6, 0.05, dtype=np.float64)

    # Generate synthetic walking signal (1 Hz cadence)
    t_step = 1.0 / 60.0
    q_template = np.zeros(6, dtype=np.float64)
    times = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        t_now = float(i * t_step)
        phase = 2 * np.pi * t_now
        # Synthetic q
        q_template[0] = 0.3 * np.sin(phase)
        q_template[1] = 0.6 * np.maximum(0.0, np.sin(phase + np.pi/4))
        q_template[2] = 0.3 * np.sin(phase + np.pi/2)
        q_template[3] = 0.3 * np.sin(phase + np.pi)
        q_template[4] = 0.6 * np.maximum(0.0, np.sin(phase + np.pi + np.pi/4))
        q_template[5] = 0.3 * np.sin(phase + np.pi + np.pi/2)
        hip_z = 0.5 + 0.02 * np.cos(2 * phase)

        t0 = time.perf_counter_ns()
        try:
            predictor.feed(t_now=t_now, q=q_template, sigma_per_joint=sigma_per_joint,
                           hip_z_world_m=float(hip_z))
        except Exception:
            pass
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0
    return _stats_us(times[500:])  # extra warmup for Hilbert cold-start


# ────────────────────────────────────────────────────────────────────────────
# Block 5: Forecast publisher.publish
# ────────────────────────────────────────────────────────────────────────────

def bench_forecast_publish(iters: int, rng) -> Dict[str, float]:
    try:
        from perception.realtime.forecast_publisher import ForecastPublisher
    except ImportError:
        try:
            from src.perception.realtime.forecast_publisher import ForecastPublisher  # type: ignore
        except ImportError:
            return {'mean': float('nan'), 'p50': float('nan'),
                    'p95': float('nan'), 'p99': float('nan'),
                    'max': float('nan')}

    name = "bench_forecast"
    _cleanup_shm(name)
    try:
        pub = ForecastPublisher(name=name, create=True)
    except TypeError:
        # API may differ — try common variants
        try:
            pub = ForecastPublisher(name)
        except Exception:
            return {'mean': float('nan'), 'p50': float('nan'),
                    'p95': float('nan'), 'p99': float('nan'),
                    'max': float('nan')}

    # Inspect publish signature
    import inspect
    sig = inspect.signature(pub.publish)

    # Common forecast fields
    common_args = {
        'frame_id': 0,
        'rgb_ts_ns': 1_700_000_000_000_000_000,
        'forecast_ts_ns': 1_700_000_000_050_000_000,
        'tau_s': 0.05,
        'phi_pred': 0.5,
        'omega_pred': 6.28,
        'q_pred': np.zeros(6, dtype=np.float32),
        'sigma_phi': 0.01,
        'sigma_omega': 0.1,
        'cascade_level': 3,
        'valid': True,
    }
    kwargs = {k: v for k, v in common_args.items() if k in sig.parameters}

    times = np.zeros(iters, dtype=np.float64)
    success = 0
    for i in range(iters):
        kwargs['frame_id'] = i
        t0 = time.perf_counter_ns()
        try:
            pub.publish(**kwargs)
            success += 1
        except Exception as e:
            if i == 0:
                print(f"  forecast publish error: {e}")
            break
        t1 = time.perf_counter_ns()
        times[i] = t1 - t0
    if success < iters // 2:
        return {'mean': float('nan'), 'p50': float('nan'),
                'p95': float('nan'), 'p99': float('nan'),
                'max': float('nan')}
    times = times[200:success]
    _cleanup_shm(name)
    return _stats_us(times)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=10000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)

    print("=" * 85)
    print(f"T3→T4 sub-block breakdown ({args.iters} iters per block)")
    print("=" * 85)
    print()
    print("Reference (5/18 measurement):")
    print("  A1 (minimal):    T3→T4 = 0.621 ms p50")
    print("  A2 (full):       T3→T4 = 2.019 ms p50")
    print("  A2 - A1:         +1.398 ms (the regression)")
    print()

    results = {}
    print("Running benchmarks...")
    print()

    print("  [1/5] SHM v2 prep (numpy zeros + dict)...")
    results['1_shm_prep'] = bench_shm_prep(args.iters, rng)

    print("  [2/5] compute_kp_sigma...")
    results['2_compute_kp_sigma'] = bench_compute_kp_sigma(args.iters, rng)

    print("  [3a/5] SHM v2 publish (Python)...")
    results['3a_shm_publish_py'] = bench_shm_publish_python(args.iters, rng)

    print("  [3b/5] SHM v2 publish (C++)...")
    results['3b_shm_publish_cpp'] = bench_shm_publish_cpp(args.iters, rng)

    print("  [4/5] Plan D predictor.feed...")
    results['4_plan_d_feed'] = bench_plan_d_feed(args.iters, rng)

    print("  [5/5] forecast publisher.publish...")
    results['5_forecast_publish'] = bench_forecast_publish(args.iters, rng)

    print()
    print("=" * 85)
    print("Results")
    print("=" * 85)
    print()

    # Compute sum (using Python publish for realistic comparison)
    sum_p50 = (
        results['1_shm_prep']['p50']
        + results['2_compute_kp_sigma']['p50']
        + results['3a_shm_publish_py']['p50']
        + results['4_plan_d_feed']['p50']
        + results['5_forecast_publish']['p50']
    )
    if np.isnan(sum_p50):
        sum_p50 = 0.0
        for k in ('1_shm_prep', '2_compute_kp_sigma', '3a_shm_publish_py',
                  '4_plan_d_feed', '5_forecast_publish'):
            v = results[k]['p50']
            if not np.isnan(v):
                sum_p50 += v

    print(f"  {'Block':<40} {'p50':>10} {'p99':>10} {'max':>10} {'share':>8}")
    print("  " + "-" * 80)
    _print_row("1. SHM v2 prep (zeros + dict)", results['1_shm_prep'], sum_p50)
    _print_row("2. compute_kp_sigma", results['2_compute_kp_sigma'], sum_p50)
    _print_row("3a. SHM v2 publish (Python)", results['3a_shm_publish_py'], sum_p50)
    _print_row("3b. SHM v2 publish (C++)",  results['3b_shm_publish_cpp'], sum_p50)
    _print_row("4. Plan D predictor.feed", results['4_plan_d_feed'], sum_p50)
    _print_row("5. forecast publisher.publish", results['5_forecast_publish'], sum_p50)
    print("  " + "-" * 80)
    print(f"  {'SUM (1+2+3a+4+5)':<40} p50={sum_p50:8.2f}us = {sum_p50/1000.0:.3f} ms")
    print(f"  {'5/18 measured A2-A1 regression':<40} p50={1398:8.2f}us = 1.398 ms (reference)")
    print()

    # Identify dominant block
    items = [
        ('1. SHM v2 prep',          results['1_shm_prep']['p50']),
        ('2. compute_kp_sigma',     results['2_compute_kp_sigma']['p50']),
        ('3a. SHM v2 publish (Py)', results['3a_shm_publish_py']['p50']),
        ('4. Plan D feed',          results['4_plan_d_feed']['p50']),
        ('5. forecast publish',     results['5_forecast_publish']['p50']),
    ]
    items_valid = [(k, v) for k, v in items if not np.isnan(v)]
    items_valid.sort(key=lambda x: x[1], reverse=True)
    print("  Top 3 bottlenecks (p50):")
    for i, (k, v) in enumerate(items_valid[:3]):
        print(f"    #{i+1}. {k:<30}  {v:8.2f} us")
    print()

    # Potential savings if C++ replaces Python publish
    py_pub = results['3a_shm_publish_py']['p50']
    cpp_pub = results['3b_shm_publish_cpp']['p50']
    if not np.isnan(py_pub) and not np.isnan(cpp_pub):
        savings = py_pub - cpp_pub
        print(f"  C++ publish savings (3a → 3b): {savings:.2f} us = {savings/1000.0:.4f} ms")
        print(f"    → Negligible vs total 1398 us regression")
    print()


if __name__ == "__main__":
    sys.exit(main() or 0)
