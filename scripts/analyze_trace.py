#!/usr/bin/env python3
"""Analyze pipeline_main.py 의 --trace-csv output — 진정 RT analysis.

사용 (Jetson 또는 Mac):
    python3 scripts/analyze_trace.py /tmp/production_trace.csv

Output (stdout + optional plot):
    Per-stage latency: T0-T1, T1-T2, T2-T3, T3-T4 의 p50/p95/p99/max
    Frame interval jitter (T4 - prev_T4): p50/p95/p99/max
    Cumulative vision e2e (T0 → T4) 의 p50/p95/p99/max
    Warmup detection (first N frames vs steady-state)
    Drop detection (gaps in frame_id sequence — 단 현재 frame_id 는 sequential, no gap)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_trace(csv_path: str) -> dict:
    """Load trace CSV. Returns dict of np arrays."""
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty trace: {csv_path}")
    def _opt_float(key):
        if key in rows[0]:
            return np.array([float(r[key]) for r in rows], dtype=np.float64)
        return None

    return {
        'frame_id': np.array([int(r['frame_id']) for r in rows], dtype=np.int64),
        't0_mono_ns': np.array([int(r['t0_mono_ns']) for r in rows], dtype=np.int64),
        't1_fetch_done_perf': np.array([float(r['t1_fetch_done_perf']) for r in rows], dtype=np.float64),
        't2_predict_done': np.array([float(r['t2_predict_done']) for r in rows], dtype=np.float64),
        't3_depth3d_done': np.array([float(r['t3_depth3d_done']) for r in rows], dtype=np.float64),
        't4_publish_done_mono_ns': np.array(
            [int(r['t4_publish_done_mono_ns']) for r in rows], dtype=np.int64
        ),
        'interval_ms': np.array([float(r['interval_ms']) for r in rows], dtype=np.float64),
        'valid': np.array([int(r['valid']) for r in rows], dtype=np.int8),
        # Path B: predict stage profile (optional, added 2026-05-12)
        'predict_preprocess_ms': _opt_float('predict_preprocess_ms'),
        'predict_infer_ms': _opt_float('predict_infer_ms'),
        'predict_postprocess_ms': _opt_float('predict_postprocess_ms'),
    }


def percentiles(arr: np.ndarray, name: str) -> str:
    if len(arr) == 0:
        return f"  {name:<32} (empty)"
    return (
        f"  {name:<32} "
        f"mean {arr.mean():>7.3f}  "
        f"p50 {np.percentile(arr, 50):>7.3f}  "
        f"p95 {np.percentile(arr, 95):>7.3f}  "
        f"p99 {np.percentile(arr, 99):>7.3f}  "
        f"p99.9 {np.percentile(arr, 99.9):>7.3f}  "
        f"max {arr.max():>7.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="trace CSV from pipeline_main.py --trace-csv")
    ap.add_argument("--warmup-frames", type=int, default=200,
                    help="skip first N frames (default 200, ~3s at 67Hz)")
    ap.add_argument("--plot", action="store_true",
                    help="save histogram plots (requires matplotlib)")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 1

    data = load_trace(args.csv)
    n_total = len(data['frame_id'])
    print("=" * 80)
    print(f"  RT Trace Analysis — {args.csv}")
    print("=" * 80)
    print(f"  Total frames: {n_total}")
    print(f"  Warmup (skipped): {args.warmup_frames}")
    print(f"  Steady-state frames: {n_total - args.warmup_frames}")
    print()

    if n_total <= args.warmup_frames:
        print(f"WARN: only {n_total} frames, < warmup ({args.warmup_frames}). "
              "Skipping warmup analysis, using all frames as steady-state.")
        args.warmup_frames = 0

    sl = slice(args.warmup_frames, None)
    skip_warmup_analysis = (args.warmup_frames == 0)

    # Per-stage durations (seconds → ms)
    t1 = data['t1_fetch_done_perf'][sl]
    t2 = data['t2_predict_done'][sl]
    t3 = data['t3_depth3d_done'][sl]
    t0_mono_ns = data['t0_mono_ns'][sl]
    t4_mono_ns = data['t4_publish_done_mono_ns'][sl]

    # Stage breakdown (all in ms)
    # NOTE: t0 of frame = perf_counter at frame start. t1/t2/t3 also perf_counter (s).
    #       t0_mono_ns = int(t0 * 1e9). t4 = monotonic_ns.
    # All durations from differences. We need a consistent reference.
    # Approach: convert t0_mono_ns back to seconds for comparison.
    t0_s = t0_mono_ns / 1e9
    t4_s = t4_mono_ns / 1e9
    # perf_counter and monotonic should be close enough on Linux (same CLOCK_MONOTONIC)

    stage_fetch_ms = (t1 - t0_s) * 1000.0       # T0 → T1: grab + retrieve + release
    stage_predict_ms = (t2 - t1) * 1000.0       # T1 → T2: TRT inference
    stage_depth3d_ms = (t3 - t2) * 1000.0       # T2 → T3: depth 3D + bone_bc
    stage_shm_ms = (t4_s - t3) * 1000.0         # T3 → T4: SHM write
    e2e_ms = (t4_s - t0_s) * 1000.0             # T0 → T4: full vision e2e
    interval_ms = data['interval_ms'][sl]       # T4 - prev_T4

    print("=" * 80)
    print("  Per-stage latency (ms, steady-state)")
    print("=" * 80)
    print(f"  {'stage':<32} {'mean':>7}  {'p50':>7}  {'p95':>7}  "
          f"{'p99':>7}  {'p99.9':>7}  {'max':>7}")
    print(f"  {'-'*32} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    print(percentiles(stage_fetch_ms,    "T0→T1 fetch (RGB+release)"))
    print(percentiles(stage_predict_ms,  "T1→T2 predict (TRT)"))
    print(percentiles(stage_depth3d_ms,  "T2→T3 depth_3d+bone"))
    print(percentiles(stage_shm_ms,      "T3→T4 SHM write"))
    print(f"  {'-'*32}")
    print(percentiles(e2e_ms,            "T0→T4 vision e2e (cumulative)"))
    print()
    print("=" * 80)
    print("  Frame periodicity (interval = T4 - prev_T4)")
    print("=" * 80)
    print(percentiles(interval_ms, "interval (ms)"))
    # Expected: 1/67Hz = 14.9ms
    expected_interval = 1000.0 / 67.0
    jitter_ms = interval_ms - expected_interval
    print(percentiles(np.abs(jitter_ms), "|jitter| (ms, vs 67Hz)"))
    print()

    # Warmup vs steady-state comparison
    print("=" * 80)
    print("  Warmup detection (first 200f vs steady)")
    print("=" * 80)
    if skip_warmup_analysis:
        print("  (skipped — too few frames for warmup separation)")
    else:
        warmup_slice = slice(0, args.warmup_frames)
        warm_t1 = data['t1_fetch_done_perf'][warmup_slice]
        warm_t2 = data['t2_predict_done'][warmup_slice]
        warm_predict = (warm_t2 - warm_t1) * 1000.0
        if len(warm_predict) > 0:
            print(f"  Warmup    predict (T1→T2): mean {warm_predict.mean():.2f}ms  "
                  f"max {warm_predict.max():.2f}ms  std {warm_predict.std():.2f}")
            print(f"  Steady    predict (T1→T2): mean {stage_predict_ms.mean():.2f}ms  "
                  f"max {stage_predict_ms.max():.2f}ms  std {stage_predict_ms.std():.2f}")
            if warm_predict.max() > stage_predict_ms.max() * 1.1:
                print(f"  → Warmup max {warm_predict.max():.2f}ms > steady max "
                      f"{stage_predict_ms.max():.2f}ms (진정 CUDA/cudnn warmup detected)")
        else:
            print("  (empty warmup slice)")
    print()

    # Drop detection (frame_id should be sequential 0, 1, 2, ...)
    print("=" * 80)
    print("  Drop detection (frame_id gaps)")
    print("=" * 80)
    fid = data['frame_id']
    gaps = np.diff(fid)
    n_gaps = int(np.sum(gaps > 1))
    print(f"  Total frames: {n_total}")
    print(f"  Sequential gaps (diff > 1): {n_gaps}")
    if n_gaps > 0:
        print(f"  Max gap: {gaps.max()}")
        print(f"  → 진정 *drop detected* — pipeline 처리 부족")
    else:
        print(f"  → 진정 *no drops in trace* — frame_id sequential ✓")
    print()

    # Valid frames
    n_valid = int(np.sum(data['valid']))
    print(f"  Valid frames: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")
    print()

    # Path B: predict() per-stage profile (if available)
    if data['predict_preprocess_ms'] is not None:
        pp_pre = data['predict_preprocess_ms'][sl]
        pp_inf = data['predict_infer_ms'][sl]
        pp_post = data['predict_postprocess_ms'][sl]
        # Filter zero-rows (production frames without profile)
        nonzero = pp_pre > 0
        if nonzero.any():
            pp_pre = pp_pre[nonzero]
            pp_inf = pp_inf[nonzero]
            pp_post = pp_post[nonzero]
            print("=" * 80)
            print("  Path B: predict() per-stage profile (Python overhead 분해)")
            print("=" * 80)
            print(percentiles(pp_pre,  "  preprocess (numpy→GPU+norm)"))
            print(percentiles(pp_inf,  "  TRT infer (execute+sync)"))
            print(percentiles(pp_post, "  postprocess (parse+scale)"))
            print(f"  {'-'*32}")
            print(percentiles(pp_pre + pp_inf + pp_post, "  predict sum (sanity check)"))
            print()
            print(f"  진정 *trtexec engine floor: 8.48ms")
            print(f"  진정 *현재 TRT infer:        {pp_inf.mean():.2f}ms "
                  f"(overhead = {pp_inf.mean() - 8.48:+.2f}ms)")
            print(f"  진정 *preprocess:            {pp_pre.mean():.2f}ms")
            print(f"  진정 *postprocess:           {pp_post.mean():.2f}ms")
            print()

    # Python overhead estimate (vs trtexec)
    TRTEXEC_FLOOR_MS = 8.48   # from trtexec measurement (commit 5165576)
    python_overhead_ms = stage_predict_ms.mean() - TRTEXEC_FLOOR_MS
    print("=" * 80)
    print("  Python overhead vs trtexec engine floor")
    print("=" * 80)
    print(f"  trtexec engine isolated: 8.48ms (TRT v100300, FP16)")
    print(f"  production predict:      {stage_predict_ms.mean():.2f}ms")
    print(f"  Python overhead:         {python_overhead_ms:+.2f}ms")
    if python_overhead_ms > 5.0:
        print(f"  → 진정 *상당한 Python overhead* (cvt + preprocess + parse?)")
    elif python_overhead_ms > 0:
        print(f"  → 진정 *합리적 Python overhead* (preprocess + parse)")
    print()

    # Optional histogram plots
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0, 0].hist(stage_predict_ms, bins=50, color='tab:blue', alpha=0.7)
            axes[0, 0].set_title(f"predict (mean {stage_predict_ms.mean():.2f}ms)")
            axes[0, 0].set_xlabel("ms")
            axes[0, 1].hist(e2e_ms, bins=50, color='tab:orange', alpha=0.7)
            axes[0, 1].set_title(f"e2e T0→T4 (mean {e2e_ms.mean():.2f}ms)")
            axes[0, 1].set_xlabel("ms")
            axes[1, 0].hist(interval_ms, bins=50, color='tab:green', alpha=0.7)
            axes[1, 0].set_title(f"frame interval (mean {interval_ms.mean():.2f}ms)")
            axes[1, 0].set_xlabel("ms")
            axes[1, 1].plot(jitter_ms, color='tab:red', lw=0.5)
            axes[1, 1].set_title("jitter trace (T4 interval - 14.9ms)")
            axes[1, 1].set_xlabel("frame")
            axes[1, 1].set_ylabel("ms")
            fig.tight_layout()
            out_png = args.csv.rsplit(".", 1)[0] + "_rt_analysis.png"
            fig.savefig(out_png, dpi=100)
            print(f"  Plots saved: {out_png}")
            plt.close(fig)
        except ImportError:
            print("  (matplotlib not installed, skipping plots)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
