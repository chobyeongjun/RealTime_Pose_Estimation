"""Sprint 1 Phase 2 — ablation result analyzer.

Reads /tmp/sprint1_ablation/{label}.log + {label}_trace.csv from 4 runs.
Reports side-by-side stage breakdown so you can see where each gain comes from.

Usage:
    python3 scripts/analyze_sprint1_ablation.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

OUTDIR = Path("/tmp/sprint1_ablation")
LABELS = ["01_baseline", "02_async", "03_cuda", "04_both"]
DESCRIPTIONS = {
    "01_baseline": "inline + torch    (Sprint 1 start)",
    "02_async":    "async  + torch    (+Phase 1)",
    "03_cuda":     "inline + CUDA     (+Week 3)",
    "04_both":     "async  + CUDA     (+Phase 1 + Week 3)",
}


# ── Log parsing ────────────────────────────────────────────────────────────
RE_E2E = re.compile(r"\[e2e lat\]\s+([\d.]+)±([\d.]+)ms\s+\(min=([\d.]+)\s+max=([\d.]+)\)")
RE_PROFILE_HEADER = re.compile(r"\[PROFILE\]\s+(\d+)f avg \(([\d.]+)ms/frame\)")
RE_PROFILE_ROW = re.compile(r"^\s+(\w+)\s+([\d.]+)ms\s+\((\d+)%\)$")
RE_FPS = re.compile(r"\[FPS\]\s+([\d.]+)\s*Hz")


def parse_log(label: str) -> Dict:
    """Extract last [e2e lat] block + last [PROFILE] block from log."""
    p = OUTDIR / f"{label}.log"
    if not p.is_file():
        return {"exists": False}
    result = {"exists": True, "label": label, "e2e": None, "profile": {},
              "fps": None, "profile_total_ms": None, "profile_frames": None}
    with open(p, "r", errors="replace") as f:
        lines = f.readlines()

    # Last [e2e lat] line
    for ln in reversed(lines):
        m = RE_E2E.search(ln)
        if m:
            result["e2e"] = {
                "mean": float(m.group(1)),
                "std":  float(m.group(2)),
                "min":  float(m.group(3)),
                "max":  float(m.group(4)),
            }
            break

    # Last [FPS] line
    for ln in reversed(lines):
        m = RE_FPS.search(ln)
        if m:
            result["fps"] = float(m.group(1))
            break

    # Last [PROFILE] block (collect 4 rows after the header)
    for i in range(len(lines) - 1, -1, -1):
        m = RE_PROFILE_HEADER.search(lines[i])
        if m:
            result["profile_frames"] = int(m.group(1))
            result["profile_total_ms"] = float(m.group(2))
            for j in range(i + 1, min(i + 8, len(lines))):
                rm = RE_PROFILE_ROW.match(lines[j])
                if rm:
                    name = rm.group(1)
                    ms = float(rm.group(2))
                    pct = int(rm.group(3))
                    result["profile"][name] = (ms, pct)
            break
    return result


# ── Trace CSV parsing ──────────────────────────────────────────────────────
def parse_trace_csv(label: str) -> Optional[Dict]:
    """Stage breakdown from per-frame trace (more precise than profile)."""
    p = OUTDIR / f"{label}_trace.csv"
    if not p.is_file():
        return None
    fetch_ms, predict_ms, depth_ms = [], [], []
    pre_ms, inf_ms, post_ms = [], [], []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t1 = float(row["t1_fetch_done_perf"])
                t2 = float(row["t2_predict_done"])
                t3 = float(row["t3_depth3d_done"])
                # t0 stored as ns (int(t0 * 1e9)) — convert back to seconds
                t0 = float(row["t0_mono_ns"]) / 1e9
                fetch_ms.append((t1 - t0) * 1000.0)
                predict_ms.append((t2 - t1) * 1000.0)
                depth_ms.append((t3 - t2) * 1000.0)
                pre_ms.append(float(row.get("predict_preprocess_ms", "0")))
                inf_ms.append(float(row.get("predict_infer_ms", "0")))
                post_ms.append(float(row.get("predict_postprocess_ms", "0")))
            except (ValueError, KeyError):
                continue
    if not fetch_ms:
        return None
    # Drop first 100 (warmup)
    skip = min(100, len(fetch_ms) // 4)
    def p50(arr): return float(np.percentile(arr[skip:], 50)) if arr[skip:] else 0.0
    return {
        "frames": len(fetch_ms),
        "fetch_p50": p50(fetch_ms),
        "predict_p50": p50(predict_ms),
        "depth_p50": p50(depth_ms),
        "preprocess_p50": p50(pre_ms),
        "infer_p50": p50(inf_ms),
        "postprocess_p50": p50(post_ms),
    }


# ── Report ─────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 78)
    print("Sprint 1 Phase 2 — Ablation Results")
    print("=" * 78)

    results = []
    for lab in LABELS:
        log = parse_log(lab)
        trace = parse_trace_csv(lab)
        results.append((lab, log, trace))

    # Missing runs warning
    missing = [lab for lab, log, _ in results if not log.get("exists")]
    if missing:
        print(f"⚠ Missing log(s): {missing}")
        print(f"  Did you run scripts/run_sprint1_ablation.sh on Jetson?")
        return 1

    # ── e2e summary ────────────────────────────────────────────────────────
    print()
    print("─── E2E latency (from log [e2e lat] last block) ───")
    print(f"{'label':<14} {'description':<38} {'e2e mean±std (min~max)':<30}")
    print("─" * 78)
    e2e_vals = {}
    for lab, log, _ in results:
        e2e = log.get("e2e")
        desc = DESCRIPTIONS[lab]
        if e2e:
            s = f"{e2e['mean']:.2f}±{e2e['std']:.2f}ms ({e2e['min']:.1f}~{e2e['max']:.1f})"
            e2e_vals[lab] = e2e["mean"]
        else:
            s = "(no e2e line — run too short?)"
            e2e_vals[lab] = None
        print(f"{lab:<14} {desc:<38} {s}")

    # ── gain decomposition ─────────────────────────────────────────────────
    if all(e2e_vals.get(k) is not None for k in LABELS):
        print()
        print("─── Gain decomposition ───")
        b, a, c, both = e2e_vals["01_baseline"], e2e_vals["02_async"], \
                         e2e_vals["03_cuda"],     e2e_vals["04_both"]
        phase1_gain = b - a
        week3_gain  = b - c
        combined_gain = b - both
        additive = phase1_gain + week3_gain
        print(f"  Phase 1 gain (01 − 02):     {phase1_gain:+.2f} ms")
        print(f"  Week 3 gain  (01 − 03):     {week3_gain:+.2f} ms")
        print(f"  Combined     (01 − 04):     {combined_gain:+.2f} ms")
        print(f"  Additive expected:          {additive:+.2f} ms")
        residual = combined_gain - additive
        verdict = "INDEPENDENT" if abs(residual) < 0.3 else "INTERACTION"
        print(f"  Residual (combined − sum):  {residual:+.2f} ms  [{verdict}]")
        if abs(residual) >= 0.3:
            print(f"  ⚠ Non-additive: bottleneck shifts between optimizations,")
            print(f"    OR measurement noise > {abs(residual):.1f} ms.")

    # ── per-stage breakdown (from trace CSV) ───────────────────────────────
    print()
    print("─── Stage breakdown (p50 ms, from trace CSV) ───")
    stages = ["fetch", "predict", "depth", "preprocess", "infer", "postprocess"]
    header = f"{'stage':<14}" + "".join(f" {lab:>12}" for lab in LABELS)
    print(header)
    print("─" * len(header))
    for stage in stages:
        row = f"{stage:<14}"
        for lab, _, trace in results:
            if trace is None:
                row += f" {'?':>12}"
            else:
                v = trace.get(f"{stage}_p50", 0.0)
                row += f" {v:>11.3f}m"
        print(row)

    # ── PROFILE 4-section breakdown (matches console output) ──────────────
    print()
    print("─── PROFILE block (last 200-frame avg, from log) ───")
    profile_sections = ["fetch", "predict", "depth_3d", "shm"]
    print(f"{'section':<14}" + "".join(f" {lab:>12}" for lab in LABELS))
    print("─" * 78)
    for sec in profile_sections:
        row = f"{sec:<14}"
        for lab, log, _ in results:
            prof = log.get("profile", {})
            if sec in prof:
                ms, _ = prof[sec]
                row += f" {ms:>11.2f}m"
            else:
                row += f" {'?':>12}"
        print(row)
    # Total row
    row = f"{'total':<14}"
    for lab, log, _ in results:
        t = log.get("profile_total_ms")
        if t:
            row += f" {t:>11.2f}m"
        else:
            row += f" {'?':>12}"
    print(row)

    print()
    print("=" * 78)
    print("Interpretation guide:")
    print("  Phase 1 (async): expect predict↓ (Plan D feed/forecast moved out)")
    print("  Week 3 (CUDA):   expect preprocess↓ (kernel replaces torch ops)")
    print("  Combined:        both deltas should appear (≈ additive)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
