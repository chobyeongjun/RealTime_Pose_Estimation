#!/usr/bin/env python3
"""Track A vs Track B comparison analyzer.

Reads trace.csv from each condition under recordings/track_comparison_TS/
and produces a comprehensive stage-by-stage comparison.

Usage:
    python3 scripts/analyze_track_comparison.py recordings/track_comparison_YYYYMMDD_HHMMSS/

Output:
    - Console table (paste-friendly)
    - analysis.md (markdown summary)
    - analysis.json (machine-readable)
    - latency_comparison.png (overlay histograms)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _to_seconds(name: str, raw):
    arr = np.full(len(raw), np.nan, dtype=np.float64)
    is_ns = name.endswith('_ns')
    for i, v in enumerate(raw):
        try:
            x = float(v)
        except (ValueError, TypeError):
            continue
        arr[i] = (x * 1e-9) if is_ns else x
    return arr


def analyze_trace(trace_path: Path) -> dict:
    if not trace_path.exists():
        return {'present': False, 'path': str(trace_path)}
    with open(trace_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'present': True, 'path': str(trace_path), 'empty': True}

    fields = list(rows[0].keys())
    stage_cols = [c for c in fields if re.match(r'^t\d', c, re.IGNORECASE)]
    if not stage_cols:
        return {'present': True, 'path': str(trace_path), 'no_stage_cols': True, 'fields': fields}

    stage_arrays = {c: _to_seconds(c, [r[c] for r in rows]) for c in stage_cols}

    first = stage_cols[0]
    last = stage_cols[-1]
    e2e = (stage_arrays[last] - stage_arrays[first]) * 1000.0  # ms

    e2e_finite = e2e[np.isfinite(e2e)]
    if len(e2e_finite) < 10:
        return {'present': True, 'path': str(trace_path), 'too_few_frames': len(rows)}

    # Skip warmup (first 200 frames or 10%, whichever smaller)
    warmup_n = min(200, max(50, len(e2e_finite) // 10))
    steady = e2e_finite[warmup_n:]

    # Per-stage deltas
    stage_deltas = {}
    for i in range(1, len(stage_cols)):
        delta = (stage_arrays[stage_cols[i]] - stage_arrays[stage_cols[i-1]]) * 1000.0
        d_finite = delta[np.isfinite(delta)][warmup_n:]
        if len(d_finite) > 0:
            stage_deltas[f"{stage_cols[i-1]}→{stage_cols[i]}"] = {
                'mean': float(d_finite.mean()),
                'p50': float(np.percentile(d_finite, 50)),
                'p95': float(np.percentile(d_finite, 95)),
                'p99': float(np.percentile(d_finite, 99)),
                'max': float(d_finite.max()),
            }

    return {
        'present': True,
        'path': str(trace_path),
        'total_frames': len(rows),
        'warmup_skipped': warmup_n,
        'steady_frames': len(steady),
        'e2e_ms': {
            'mean': float(steady.mean()),
            'p50': float(np.percentile(steady, 50)),
            'p95': float(np.percentile(steady, 95)),
            'p99': float(np.percentile(steady, 99)),
            'p99_9': float(np.percentile(steady, 99.9)),
            'max': float(steady.max()),
            'std': float(steady.std()),
        },
        'stage_deltas_ms': stage_deltas,
        'hard_limit_violations': {
            'count': int((steady > 20.0).sum()),
            'rate_pct': float((steady > 20.0).mean() * 100.0),
        },
    }


def find_trace(condition_dir: Path) -> Optional[Path]:
    """Find trace.csv in condition dir (handle multiple naming patterns)."""
    candidates = list(condition_dir.glob('trace*.csv')) + list(condition_dir.glob('*_trace.csv'))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('session_dir', help='recordings/track_comparison_TS/')
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        print(f"ERROR: not a directory: {session_dir}", file=sys.stderr)
        return 1

    conditions = ['A1_track_a_minimal', 'A2_track_a_full',
                  'B1_track_b_minimal', 'B2_track_b_full']

    results = {}
    for cond in conditions:
        cond_dir = session_dir / cond
        trace = find_trace(cond_dir) if cond_dir.is_dir() else None
        if trace is None:
            results[cond] = {'present': False, 'reason': 'no trace file'}
            continue
        results[cond] = analyze_trace(trace)

    # ─── Print comparison table ──────────────────────────────────────────
    print("=" * 100)
    print("Track Comparison Analysis")
    print("=" * 100)
    print(f"  Session dir: {session_dir}")
    print()

    print(f"{'Condition':<28} {'Frames':>8} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'p99.9':>8} {'max':>8} {'>20ms%':>8}")
    print("-" * 100)
    for cond in conditions:
        r = results[cond]
        if r.get('present') and r.get('steady_frames', 0) > 10:
            e = r['e2e_ms']
            v = r['hard_limit_violations']
            print(f"{cond:<28} {r['steady_frames']:>8} "
                  f"{e['mean']:>7.2f}  {e['p50']:>7.2f}  {e['p95']:>7.2f}  "
                  f"{e['p99']:>7.2f}  {e['p99_9']:>7.2f}  {e['max']:>7.2f}  {v['rate_pct']:>7.3f}")
        else:
            reason = r.get('reason', r.get('no_stage_cols', r.get('too_few_frames', 'missing')))
            print(f"{cond:<28} {'-':>8} (no data: {reason})")
    print()

    # ─── Stage-by-stage breakdown ────────────────────────────────────────
    print("=" * 100)
    print("Stage-by-stage breakdown (p50 ms per stage)")
    print("=" * 100)
    for cond in conditions:
        r = results[cond]
        if not r.get('present') or 'stage_deltas_ms' not in r:
            continue
        print(f"\n  {cond}:")
        for stage, stats in r['stage_deltas_ms'].items():
            print(f"    {stage:<35}: p50={stats['p50']:6.3f}ms  p99={stats['p99']:6.3f}ms  max={stats['max']:6.3f}ms")
    print()

    # ─── Comparison: A1 vs A2 (Plan D cost in Track A) ───────────────────
    a1 = results.get('A1_track_a_minimal')
    a2 = results.get('A2_track_a_full')
    if a1 and a2 and a1.get('present') and a2.get('present'):
        print("=" * 100)
        print("Comparison: Plan D + SHM v2 의 cost (Track A)")
        print("=" * 100)
        d_p50 = a2['e2e_ms']['p50'] - a1['e2e_ms']['p50']
        d_p99 = a2['e2e_ms']['p99'] - a1['e2e_ms']['p99']
        print(f"  A1 (minimal)  p50: {a1['e2e_ms']['p50']:.2f}ms  p99: {a1['e2e_ms']['p99']:.2f}ms")
        print(f"  A2 (full)     p50: {a2['e2e_ms']['p50']:.2f}ms  p99: {a2['e2e_ms']['p99']:.2f}ms")
        print(f"  Plan D + SHM v2 cost: p50 {d_p50:+.2f}ms, p99 {d_p99:+.2f}ms")
        print()

    # ─── Comparison: Track A vs Track B (full features) ──────────────────
    b2 = results.get('B2_track_b_full')
    if a2 and b2 and a2.get('present') and b2.get('present'):
        print("=" * 100)
        print("Comparison: Track A vs Track B (full features)")
        print("=" * 100)
        d_p50 = b2['e2e_ms']['p50'] - a2['e2e_ms']['p50']
        d_p99 = b2['e2e_ms']['p99'] - a2['e2e_ms']['p99']
        print(f"  Track A (A2)  p50: {a2['e2e_ms']['p50']:.2f}ms  p99: {a2['e2e_ms']['p99']:.2f}ms")
        print(f"  Track B (B2)  p50: {b2['e2e_ms']['p50']:.2f}ms  p99: {b2['e2e_ms']['p99']:.2f}ms")
        print(f"  Track B vs A diff: p50 {d_p50:+.2f}ms, p99 {d_p99:+.2f}ms")
        if d_p99 < 0:
            print(f"  → Track B is FASTER by {-d_p99:.2f}ms at p99")
        else:
            print(f"  → Track A is FASTER by {d_p99:.2f}ms at p99")
        print()

    # ─── Compare to 5/13 historical baseline ─────────────────────────────
    historical_513 = {
        'condition': '5/13 walking (Track A, full)',
        'p50': 15.93,
        'p99': 16.18,
        'max': 17.34,
        'source': 'recordings/walking_20260513_212441/analyze.txt',
    }
    print("=" * 100)
    print("Historical baseline (5/13 Track A)")
    print("=" * 100)
    print(f"  5/13 Track A p50={historical_513['p50']:.2f}ms  p99={historical_513['p99']:.2f}ms  max={historical_513['max']:.2f}ms")
    if a2 and a2.get('present'):
        d_p50 = a2['e2e_ms']['p50'] - historical_513['p50']
        d_p99 = a2['e2e_ms']['p99'] - historical_513['p99']
        print(f"  Today A2 vs 5/13: p50 {d_p50:+.2f}ms, p99 {d_p99:+.2f}ms")
        if d_p50 > 1.0:
            print(f"  ⚠ Regression from 5/13: +{d_p50:.2f}ms p50")
        elif d_p50 < -1.0:
            print(f"  ✓ Improvement from 5/13: {d_p50:.2f}ms p50")
        else:
            print(f"  ≈ Within ±1ms of 5/13")
    print()

    # ─── Save outputs ────────────────────────────────────────────────────
    json_path = session_dir / 'analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    md_path = session_dir / 'analysis.md'
    with open(md_path, 'w') as f:
        f.write(f"# Track Comparison — {session_dir.name}\n\n")
        f.write("## Summary\n\n")
        f.write("| Condition | Frames | mean | p50 | p95 | p99 | max | >20ms% |\n")
        f.write("|-----------|-------:|-----:|----:|----:|----:|----:|-------:|\n")
        for cond in conditions:
            r = results[cond]
            if r.get('present') and r.get('steady_frames', 0) > 10:
                e = r['e2e_ms']
                v = r['hard_limit_violations']
                f.write(f"| {cond} | {r['steady_frames']} | "
                        f"{e['mean']:.2f} | {e['p50']:.2f} | {e['p95']:.2f} | "
                        f"{e['p99']:.2f} | {e['max']:.2f} | {v['rate_pct']:.3f}% |\n")
            else:
                f.write(f"| {cond} | - | (no data) | | | | | |\n")
        f.write("\n")
    print(f"Saved: {md_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
