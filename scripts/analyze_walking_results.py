"""Headless walking-session result analyzer.

Walking session 산출물(npz + trace.csv + plan_d.log)을 통합 분석하고
SSH 환경에서 검증 가능한 형태로 결과를 출력:
  - JSON 통계 (console + file)
  - PNG plots (matplotlib agg backend — no display)
  - Markdown 요약 (paste-friendly)

Usage:
    python3 scripts/analyze_walking_results.py recordings/walking_YYYYMMDD_HHMMSS/

또는 명시 path:
    python3 scripts/analyze_walking_results.py \\
        --npz path/to.npz --trace path/to_trace.csv --plan-d path/to_plan_d.log
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Headless matplotlib — must be set before pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = ['L_hip', 'L_knee', 'L_ankle', 'R_hip', 'R_knee', 'R_ankle']


def find_files(session_dir: Path) -> dict:
    """Auto-discover npz/csv/log files in a walking_session output directory."""
    paths = {}
    for key, pattern in [
        ('npz',    'walking_*.npz'),
        ('trace',  'trace_*.csv'),
        ('plan_d', 'plan_d_*.log'),
        ('analyze', 'analyze_*.txt'),
        ('svo',    'walking_*.svo2'),
    ]:
        hits = list(session_dir.glob(pattern))
        paths[key] = hits[0] if hits else None
    return paths


def stats_summary(arr: np.ndarray) -> dict:
    """Common percentile + NaN stats for a numeric array."""
    if arr.size == 0:
        return {"count": 0}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "all_nan": True}
    return {
        "count":     int(arr.size),
        "finite":    int(finite.size),
        "nan_ratio": float((arr.size - finite.size) / arr.size),
        "min":       float(np.min(finite)),
        "p50":       float(np.percentile(finite, 50)),
        "p95":       float(np.percentile(finite, 95)),
        "p99":       float(np.percentile(finite, 99)),
        "max":       float(np.max(finite)),
        "mean":      float(np.mean(finite)),
        "std":       float(np.std(finite)),
    }


def analyze_npz(path: Path, out_dir: Path) -> dict:
    """Pose npz: 6 keypoints × N frames. Compute coverage, σ, joint trajectories."""
    if path is None or not path.exists():
        return {"present": False}
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    result = {"present": True, "path": str(path), "keys": keys}

    # Expected fields (vary by recorder version)
    if 'kpts_3d' in data:
        kpts = data['kpts_3d']   # (N, K, 3)
        N, K, _ = kpts.shape
        result["frames"] = N
        result["K"] = K
        result["valid_per_joint"] = []
        for i in range(K):
            xyz = kpts[:, i, :]
            mask = np.all(np.isfinite(xyz), axis=1)
            result["valid_per_joint"].append({
                "name":  JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"j{i}",
                "valid_ratio": float(mask.sum() / N),
                "z_mean": float(np.nanmean(xyz[:, 2])),
                "z_std":  float(np.nanstd(xyz[:, 2])),
            })

        # Plot z-trajectory per joint
        fig, axes = plt.subplots(K, 1, figsize=(10, 2*K), sharex=True)
        if K == 1:
            axes = [axes]
        for i in range(K):
            z = kpts[:, i, 2]
            axes[i].plot(z, lw=0.6)
            axes[i].set_ylabel(f"{JOINT_NAMES[i] if i < 6 else f'j{i}'}\nZ (m)")
            axes[i].grid(alpha=0.3)
        axes[-1].set_xlabel("frame")
        fig.suptitle(f"Keypoint Z trajectory ({path.name})", fontsize=10)
        plt.tight_layout()
        png = out_dir / "kpts_3d_z.png"
        fig.savefig(png, dpi=100)
        plt.close(fig)
        result["plot_kpts_3d_z"] = str(png)

    return result


def analyze_trace(path: Path, out_dir: Path) -> dict:
    """Trace CSV: per-frame T0~T9 stage timestamps in monotonic_ns.

    Output: latency histogram + per-stage p50/p99.
    """
    if path is None or not path.exists():
        return {"present": False}
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {"present": True, "path": str(path), "rows": 0, "empty": True}

    fields = list(rows[0].keys())
    # Try to find time-stage columns named T0, T1, ... or similar
    t_cols = [c for c in fields if re.match(r"^[Tt]\d", c)]
    result = {"present": True, "path": str(path), "rows": len(rows), "fields": fields}

    if not t_cols:
        return result

    # Convert to arrays of ns
    stages = {}
    for c in t_cols:
        vals = []
        for r in rows:
            try:
                vals.append(int(r[c]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        stages[c] = np.array(vals, dtype=float)

    # Latency = T_last - T_first
    t_keys = sorted(stages.keys(), key=lambda k: int(re.search(r"\d+", k).group()))
    t_first = stages[t_keys[0]]
    t_last  = stages[t_keys[-1]]
    e2e_ms = (t_last - t_first) / 1e6
    result["e2e_ms"] = stats_summary(e2e_ms)

    # Per-stage delta
    deltas = {}
    for i in range(1, len(t_keys)):
        d_ms = (stages[t_keys[i]] - stages[t_keys[i-1]]) / 1e6
        deltas[f"{t_keys[i-1]}_to_{t_keys[i]}"] = stats_summary(d_ms)
    result["stage_deltas_ms"] = deltas

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    finite = e2e_ms[np.isfinite(e2e_ms)]
    if finite.size > 0:
        ax.hist(finite, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.percentile(finite, 50), color='g', ls='--', label=f'p50={np.percentile(finite, 50):.1f}ms')
        ax.axvline(np.percentile(finite, 99), color='r', ls='--', label=f'p99={np.percentile(finite, 99):.1f}ms')
        ax.axvline(20.0, color='k', ls=':', label='HARD LIMIT 20ms')
        ax.set_xlabel("E2E latency (ms)")
        ax.set_ylabel("frames")
        ax.set_title(f"End-to-end latency distribution ({path.name})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        png = out_dir / "e2e_latency_hist.png"
        fig.savefig(png, dpi=100)
        result["plot_latency_hist"] = str(png)
    plt.close(fig)

    return result


def analyze_plan_d(path: Path) -> dict:
    """Parse plan_d.log for cascade transitions + stride count + sigma."""
    if path is None or not path.exists():
        return {"present": False}
    with open(path) as f:
        text = f.read()

    result = {"present": True, "path": str(path), "size_bytes": len(text)}

    # Cascade L1→L2 / L2→L3 transitions
    transitions = re.findall(r"cascade.*L(\d).*L(\d)", text)
    result["cascade_transitions"] = [{"from": int(a), "to": int(b)} for a, b in transitions]

    # Stride count
    strides = re.findall(r"stride[_ ]count[:\s=]+(\d+)", text)
    if strides:
        result["max_stride_count"] = max(int(s) for s in strides)

    # is_ready True ratio (rough)
    ready_true = len(re.findall(r"is_ready.*True", text))
    ready_false = len(re.findall(r"is_ready.*False", text))
    total = ready_true + ready_false
    if total > 0:
        result["is_ready_true_ratio"] = ready_true / total

    # Errors / warnings
    result["error_lines"]   = len(re.findall(r"\b(ERROR|Traceback)\b", text, re.IGNORECASE))
    result["warning_lines"] = len(re.findall(r"\bWARN\b", text, re.IGNORECASE))

    return result


def emit_markdown(stats: dict, out_path: Path) -> str:
    """Generate paste-friendly Markdown summary."""
    lines = []
    lines.append(f"# Walking session analysis — {stats.get('session_dir', '')}\n")

    npz = stats.get("npz", {})
    if npz.get("present"):
        lines.append(f"## Pose npz ({npz.get('frames', '?')} frames × {npz.get('K', '?')} kpts)")
        for j in npz.get("valid_per_joint", []):
            lines.append(f"- {j['name']:8s}  valid={j['valid_ratio']*100:5.1f}%  "
                         f"Z={j['z_mean']:+.2f}±{j['z_std']:.2f} m")
        if "plot_kpts_3d_z" in npz:
            lines.append(f"\nPlot → `{npz['plot_kpts_3d_z']}`")
    else:
        lines.append("## Pose npz — **MISSING** (pipeline did not produce keypoint output)")

    tr = stats.get("trace", {})
    lines.append("\n## RT trace")
    if tr.get("present") and not tr.get("empty"):
        e2e = tr.get("e2e_ms", {})
        lines.append(f"- rows: {tr.get('rows', 0)}")
        lines.append(f"- e2e p50/p95/p99 (ms): {e2e.get('p50', 'NA'):.2f} / "
                     f"{e2e.get('p95', 'NA'):.2f} / {e2e.get('p99', 'NA'):.2f}")
        lines.append(f"- e2e max (ms): {e2e.get('max', 'NA'):.2f}")
        if "plot_latency_hist" in tr:
            lines.append(f"- plot → `{tr['plot_latency_hist']}`")
    else:
        lines.append("- trace CSV **empty or missing**")

    pd_ = stats.get("plan_d", {})
    lines.append("\n## Plan D")
    if pd_.get("present"):
        lines.append(f"- log size: {pd_.get('size_bytes', 0)} bytes")
        lines.append(f"- cascade transitions: {len(pd_.get('cascade_transitions', []))}")
        if pd_.get("max_stride_count") is not None:
            lines.append(f"- max stride_count: {pd_['max_stride_count']}")
        if "is_ready_true_ratio" in pd_:
            lines.append(f"- is_ready=True ratio: {pd_['is_ready_true_ratio']*100:.1f}%")
        lines.append(f"- errors: {pd_.get('error_lines', 0)}   warnings: {pd_.get('warning_lines', 0)}")
    else:
        lines.append("- plan_d log MISSING")

    md = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", default=None,
                    help="walking_YYYYMMDD_HHMMSS directory")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--plan-d", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="Where to write PNG/MD/JSON (default: session_dir)")
    args = ap.parse_args()

    if args.session_dir:
        sd = Path(args.session_dir)
        if not sd.is_dir():
            print(f"ERROR: not a directory: {sd}", file=sys.stderr)
            sys.exit(1)
        files = find_files(sd)
    else:
        sd = Path(args.out_dir or ".")
        files = {
            'npz':    Path(args.npz)   if args.npz   else None,
            'trace':  Path(args.trace) if args.trace else None,
            'plan_d': Path(args.plan_d) if args.plan_d else None,
        }

    out_dir = Path(args.out_dir or sd)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {sd}")
    print(f"  npz:    {files.get('npz')}")
    print(f"  trace:  {files.get('trace')}")
    print(f"  plan_d: {files.get('plan_d')}")
    print()

    stats = {
        "session_dir": str(sd),
        "npz":    analyze_npz(files.get('npz'), out_dir),
        "trace":  analyze_trace(files.get('trace'), out_dir),
        "plan_d": analyze_plan_d(files.get('plan_d')),
    }

    json_path = out_dir / "analysis.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    md = emit_markdown(stats, out_dir / "analysis.md")
    print(md)
    print(f"\n=== Outputs ===")
    print(f"  JSON     : {json_path}")
    print(f"  Markdown : {out_dir / 'analysis.md'}")
    if "plot_kpts_3d_z" in stats["npz"]:
        print(f"  PNG kpts : {stats['npz']['plot_kpts_3d_z']}")
    if "plot_latency_hist" in stats["trace"]:
        print(f"  PNG latcy: {stats['trace']['plot_latency_hist']}")


if __name__ == "__main__":
    main()
