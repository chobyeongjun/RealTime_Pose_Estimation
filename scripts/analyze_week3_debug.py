"""Week 3 debug — aggregate results across scenarios × runs.

Reads /tmp/debug_week3_full/{scenario}_run{N}.log + {scenario}_run{N}_trace.csv.
Reports stage-by-stage p50 + variance across runs to distinguish noise from
real regressions.

Output sections:
  1. Per-scenario summary table (mean ± std across runs)
  2. Stream variant comparison (F1 investigation):
        cuda_inline_strm_{trt,null,default} side-by-side
  3. Async variant comparison (F2 investigation):
        cuda_async_strm_{trt,null,default} side-by-side
  4. Verdict per finding (F1, F2, F3) with statistical confidence
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

OUTDIR = Path("/tmp/debug_week3_full")
SCENARIOS = [
    "baseline_inline_torch",
    "baseline_async_torch",
    "cuda_inline_strm_trt",
    "cuda_inline_strm_null",
    "cuda_inline_strm_default",
    "cuda_async_strm_trt",
    "cuda_async_strm_null",
    "cuda_async_strm_default",
]

RE_E2E = re.compile(r"\[e2e lat\]\s+([\d.]+)±([\d.]+)ms")
RE_PROFILE_ROW = re.compile(r"^\s+(\w+)\s+([\d.]+)ms\s+\((\d+)%\)$")
RE_PROFILE_HEADER = re.compile(r"\[PROFILE\]\s+(\d+)f avg")


def parse_log(log_path: Path) -> Dict[str, Optional[float]]:
    """Extract last [e2e lat] mean and last [PROFILE] section ms values."""
    out = {"e2e": None, "fetch": None, "predict": None, "depth_3d": None, "shm": None}
    if not log_path.is_file():
        return out
    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()
    for ln in reversed(lines):
        m = RE_E2E.search(ln)
        if m:
            out["e2e"] = float(m.group(1))
            break
    # Last PROFILE block
    for i in range(len(lines) - 1, -1, -1):
        if RE_PROFILE_HEADER.search(lines[i]):
            for j in range(i + 1, min(i + 8, len(lines))):
                rm = RE_PROFILE_ROW.match(lines[j])
                if rm:
                    sec = rm.group(1)
                    if sec in out:
                        out[sec] = float(rm.group(2))
            break
    return out


def parse_trace(csv_path: Path) -> Dict[str, Optional[float]]:
    """Stage p50 from trace CSV."""
    out = {"fetch": None, "predict": None, "depth": None,
           "preprocess": None, "infer": None, "postprocess": None}
    if not csv_path.is_file():
        return out
    fetch, predict, depth = [], [], []
    pre, inf, post = [], [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t0 = float(row["t0_mono_ns"]) / 1e9
                t1 = float(row["t1_fetch_done_perf"])
                t2 = float(row["t2_predict_done"])
                t3 = float(row["t3_depth3d_done"])
                fetch.append((t1 - t0) * 1000)
                predict.append((t2 - t1) * 1000)
                depth.append((t3 - t2) * 1000)
                pre.append(float(row.get("predict_preprocess_ms", "0") or 0))
                inf.append(float(row.get("predict_infer_ms", "0") or 0))
                post.append(float(row.get("predict_postprocess_ms", "0") or 0))
            except (ValueError, KeyError):
                continue
    if not fetch:
        return out
    skip = min(100, len(fetch) // 4)
    out["fetch"] = float(np.percentile(fetch[skip:], 50))
    out["predict"] = float(np.percentile(predict[skip:], 50))
    out["depth"] = float(np.percentile(depth[skip:], 50))
    out["preprocess"] = float(np.percentile(pre[skip:], 50))
    out["infer"] = float(np.percentile(inf[skip:], 50))
    out["postprocess"] = float(np.percentile(post[skip:], 50))
    return out


def aggregate_scenario(label: str, runs: int = 3) -> Dict[str, Dict]:
    """Aggregate across runs — mean and std."""
    log_vals = {k: [] for k in ["e2e", "fetch", "predict", "depth_3d", "shm"]}
    trace_vals = {k: [] for k in ["fetch", "predict", "depth",
                                    "preprocess", "infer", "postprocess"]}
    n_runs_found = 0
    for r in range(1, runs + 1):
        log = parse_log(OUTDIR / f"{label}_run{r}.log")
        trace = parse_trace(OUTDIR / f"{label}_run{r}_trace.csv")
        if log.get("e2e") is None:
            continue
        n_runs_found += 1
        for k, v in log.items():
            if v is not None:
                log_vals[k].append(v)
        for k, v in trace.items():
            if v is not None:
                trace_vals[k].append(v)
    return {
        "n_runs": n_runs_found,
        "log_mean": {k: (float(np.mean(v)) if v else None) for k, v in log_vals.items()},
        "log_std":  {k: (float(np.std(v)) if v else None) for k, v in log_vals.items()},
        "trace_mean": {k: (float(np.mean(v)) if v else None) for k, v in trace_vals.items()},
        "trace_std":  {k: (float(np.std(v)) if v else None) for k, v in trace_vals.items()},
    }


def fmt(v, w=8):
    if v is None:
        return "?".rjust(w)
    return f"{v:>{w-2}.2f}ms"


def main() -> int:
    print("=" * 78)
    print("Week 3 Debug — aggregate analysis")
    print("=" * 78)

    agg = {s: aggregate_scenario(s) for s in SCENARIOS}

    # ── Per-scenario summary ──────────────────────────────────────────────
    print()
    print("─── E2E latency (mean ± std across runs, from log) ───")
    print(f"{'scenario':<28} {'n':>3} {'e2e':>20}")
    print("─" * 60)
    for s in SCENARIOS:
        a = agg[s]
        n = a["n_runs"]
        if n == 0:
            print(f"{s:<28} {n:>3}  no data")
            continue
        m, sd = a["log_mean"]["e2e"], a["log_std"]["e2e"]
        print(f"{s:<28} {n:>3}  {m:>6.2f} ± {sd:>4.2f} ms")

    # ── Trace CSV stage breakdown ─────────────────────────────────────────
    print()
    print("─── Stage p50 mean (from trace CSV, ms) ───")
    stages = ["fetch", "predict", "depth", "preprocess", "infer", "postprocess"]
    header = f"{'scenario':<28}" + "".join(f" {s:>11}" for s in stages)
    print(header)
    print("─" * len(header))
    for s in SCENARIOS:
        a = agg[s]
        row = f"{s:<28}"
        for stage in stages:
            v = a["trace_mean"].get(stage)
            row += f" {fmt(v, 11)}"
        print(row)

    # ── F1 verdict: stream variant comparison (inline+CUDA) ───────────────
    print()
    print("─── F1: 03_cuda depth jump — stream variant test ───")
    inline_variants = ["cuda_inline_strm_trt", "cuda_inline_strm_null", "cuda_inline_strm_default"]
    print(f"  Baseline depth p50 (inline+torch): "
          f"{agg['baseline_inline_torch']['trace_mean'].get('depth'):.3f} ms")
    print()
    for v in inline_variants:
        a = agg[v]
        d = a["trace_mean"].get("depth")
        p = a["trace_mean"].get("preprocess")
        if d is None:
            print(f"  {v:<28}  depth=? preprocess=?")
            continue
        print(f"  {v:<28}  depth={d:.3f}ms  preprocess={p:.3f}ms")

    # Decide best stream variant for inline+CUDA
    best_v = None
    best_score = float('inf')
    base_depth = agg['baseline_inline_torch']['trace_mean'].get('depth') or 0
    for v in inline_variants:
        d = agg[v]['trace_mean'].get('depth')
        p = agg[v]['trace_mean'].get('preprocess')
        if d is None or p is None:
            continue
        # Score = depth penalty + preprocess (lower = better)
        # depth penalty = max(0, d - base_depth) so we don't reward depth < base
        score = max(0, d - base_depth) + p
        if score < best_score:
            best_score = score
            best_v = v
    if best_v:
        print(f"\n  → Best stream variant: {best_v}  (score={best_score:.3f}ms)")

    # ── F2 verdict: same for async+CUDA ───────────────────────────────────
    print()
    print("─── F2: 04_both infer jump — async + stream variant test ───")
    async_variants = ["cuda_async_strm_trt", "cuda_async_strm_null", "cuda_async_strm_default"]
    for v in async_variants:
        a = agg[v]
        i = a["trace_mean"].get("infer")
        p = a["trace_mean"].get("preprocess")
        if i is None:
            print(f"  {v:<28}  infer=? preprocess=?")
            continue
        print(f"  {v:<28}  infer={i:.3f}ms  preprocess={p:.3f}ms")

    # ── F3 verdict: variance check ────────────────────────────────────────
    print()
    print("─── F3: Phase 1 (async) — variance check ───")
    a_b = agg['baseline_inline_torch']
    a_a = agg['baseline_async_torch']
    if a_b['n_runs'] >= 2 and a_a['n_runs'] >= 2:
        diff = a_a['log_mean']['e2e'] - a_b['log_mean']['e2e']
        # 2σ confidence: if |diff| > 2 * sqrt(σ_b² + σ_a²), real
        noise = 2 * np.sqrt(a_b['log_std']['e2e']**2 + a_a['log_std']['e2e']**2)
        print(f"  inline mean e2e: {a_b['log_mean']['e2e']:.2f} ± {a_b['log_std']['e2e']:.2f} ms (n={a_b['n_runs']})")
        print(f"  async  mean e2e: {a_a['log_mean']['e2e']:.2f} ± {a_a['log_std']['e2e']:.2f} ms (n={a_a['n_runs']})")
        print(f"  diff (async − inline): {diff:+.2f} ms  (2σ noise: ±{noise:.2f} ms)")
        if abs(diff) > noise:
            print(f"  → SIGNIFICANT: async {'slower' if diff > 0 else 'faster'} by {abs(diff):.2f} ms beyond noise")
        else:
            print(f"  → NOT SIGNIFICANT: diff within noise floor — no real Phase 1 gain or regression")

    # ── Final recommendation ──────────────────────────────────────────────
    print()
    print("=" * 78)
    print("RECOMMENDATIONS")
    print("=" * 78)
    if best_v:
        print(f"  - For inline+CUDA, use HWALKER_PREPROC_STREAM={best_v.split('_')[-1]}")
    print(f"  - If F3 not significant, Phase 1 async has no clear gain → keep inline default")
    print(f"  - Baseline (inline+torch) at ~14.4 ms is already sub-15 ms paper target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
