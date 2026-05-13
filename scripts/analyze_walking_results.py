"""Headless walking-session result analyzer.

Walking session 산출물(npz + trace.csv + plan_d.log)을 통합 분석하고
SSH 환경에서 검증 가능한 형태로 결과를 출력:
  - JSON 통계 (console + file)
  - PNG plots (matplotlib agg backend — no display)
  - Markdown 요약 (paste-friendly)

Schema mirrors pipeline_main.py exactly (Codex review 5bd5732 P2-1/2/3):
  NPZ keys: t_s, hip_z_world_m, left_hip_z, right_hip_z,
            left_knee_rad, right_knee_rad, left_hip_rad, right_hip_rad,
            valid, method
  Trace columns (mixed units!):
    frame_id            int
    t0_mono_ns          int    (Python perf_counter * 1e9, ns scale)
    t1_fetch_done_perf  float  (perf_counter seconds)
    t2_predict_done     float  (perf_counter seconds)
    t3_depth3d_done     float  (perf_counter seconds)
    t4_publish_done_mono_ns  int (monotonic_ns)
    interval_ms         float
    valid               int
    preprocess_ms       float
    infer_ms            float
    postprocess_ms      float

Usage:
    python3 scripts/analyze_walking_results.py recordings/walking_YYYYMMDD_HHMMSS/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _is_pose_dump(npz_path: Path) -> bool:
    """Validate the NPZ is a pose dump (has t_s + joint angle keys).

    Codex P2-followup: run_plan_d_offline.py writes *_planD_summary.npz
    in the same directory; it has its own schema and matches the broad
    walking_*.npz glob but lacks t_s, breaking analyze_npz().
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            return 't_s' in data and 'left_hip_rad' in data
    except Exception:
        return False


def find_files(session_dir: Path) -> dict:
    """Auto-discover npz/csv/log files. Codex P2-1 fix: include replay_*."""
    paths = {}
    candidates = {
        'npz': [
            'walking_*.npz',
            'replay_*.npz',
            '*_pose.npz',
        ],
        'trace': [
            'trace_*.csv',
            'replay_*_trace.csv',
            '*_trace.csv',
        ],
        'plan_d': [
            'plan_d_*.log',
            'replay_*.log',
        ],
        'svo': [
            'walking_*.svo2',
        ],
    }
    excludes = {
        'npz': ['*_planD_summary*', '*_analysis*', 'analysis.npz'],
    }
    for key, patterns in candidates.items():
        hits = []
        for p in patterns:
            hits.extend(session_dir.glob(p))
        # Apply per-key exclusion patterns
        for ex in excludes.get(key, []):
            hits = [h for h in hits if not h.match(ex)]
        if not hits:
            paths[key] = None
            continue
        # Sort newest-first
        hits = sorted(set(hits), key=lambda x: x.stat().st_mtime, reverse=True)
        if key == 'npz':
            # Schema-validate: pick the newest NPZ that is actually a pose dump.
            for h in hits:
                if _is_pose_dump(h):
                    paths[key] = h
                    break
            else:
                paths[key] = None
        else:
            paths[key] = hits[0]
    return paths


def stats_summary(arr: np.ndarray) -> dict:
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
    """Pose dump: pipeline_main.py:1081 schema (joint angles, NOT 3D keypoints)."""
    if path is None or not path.exists():
        return {"present": False}
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    result = {"present": True, "path": str(path), "keys": keys}

    if 't_s' not in data:
        result["error"] = f"unexpected schema; keys = {keys}"
        return result

    t_s = data['t_s']
    valid = data['valid'] if 'valid' in data else np.ones_like(t_s, dtype=bool)
    duration_s = float(t_s[-1] - t_s[0]) if t_s.size > 1 else 0.0
    fps = float((t_s.size - 1) / duration_s) if duration_s > 0 else 0.0
    result.update({
        "frames":      int(t_s.size),
        "duration_s":  duration_s,
        "fps":         fps,
        "valid_ratio": float(valid.sum() / max(valid.size, 1)),
        "method":      str(data['method']) if 'method' in data else None,
    })

    joint_fields = ['left_hip_rad', 'right_hip_rad',
                    'left_knee_rad', 'right_knee_rad',
                    'left_hip_z', 'right_hip_z', 'hip_z_world_m']
    result["joints"] = {}
    for name in joint_fields:
        if name in data:
            arr = np.asarray(data[name], dtype=np.float64)
            result["joints"][name] = stats_summary(arr)

    # Plot joint angles
    angle_keys = ['left_hip_rad', 'left_knee_rad', 'right_hip_rad', 'right_knee_rad']
    present = [k for k in angle_keys if k in data]
    if present:
        fig, axes = plt.subplots(len(present), 1, figsize=(11, 2.2*len(present)), sharex=True)
        if len(present) == 1: axes = [axes]
        for ax, k in zip(axes, present):
            y = np.asarray(data[k], dtype=np.float64)
            ax.plot(t_s, np.degrees(y), lw=0.7)
            ax.set_ylabel(f"{k}\n(°)")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("t (s)")
        fig.suptitle(f"Joint angles ({path.name})", fontsize=10)
        plt.tight_layout()
        png = out_dir / "joint_angles_deg.png"
        fig.savefig(png, dpi=100); plt.close(fig)
        result["plot_joint_angles"] = str(png)

    # Plot hip z (vertical) trajectories — gait cycle proxy
    if 'hip_z_world_m' in data:
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(t_s, data['hip_z_world_m'], lw=0.6, label='hip_z_world_m')
        ax.set_xlabel("t (s)"); ax.set_ylabel("hip Z (m)")
        ax.set_title("Hip vertical trajectory — gait cycle proxy")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        png = out_dir / "hip_z_trajectory.png"
        fig.savefig(png, dpi=100); plt.close(fig)
        result["plot_hip_z"] = str(png)

    return result


def _to_seconds(name: str, raw: list) -> np.ndarray:
    """Codex P2-3 fix: pipeline_main writes mixed units in trace CSV.

    *_mono_ns / *_perf_ns / *_ns columns are nanosecond ints.
    *_perf columns (no _ns suffix) are float seconds from perf_counter().
    """
    arr = np.full(len(raw), np.nan, dtype=np.float64)
    is_ns = name.endswith('_ns')
    for i, v in enumerate(raw):
        try:
            x = float(v)
        except (ValueError, TypeError):
            continue
        arr[i] = (x * 1e-9) if is_ns else x
    return arr


def analyze_trace(path: Path, out_dir: Path) -> dict:
    if path is None or not path.exists():
        return {"present": False}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"present": True, "path": str(path), "rows": 0, "empty": True}

    fields = list(rows[0].keys())
    result = {"present": True, "path": str(path), "rows": len(rows), "fields": fields}

    # Identify stage columns by order they appear in fieldnames
    stage_cols = [c for c in fields if re.match(r"^t\d", c, re.IGNORECASE)]
    if not stage_cols:
        return result

    # Normalize to seconds (mixed units!)
    stage_arrays = {c: _to_seconds(c, [r[c] for r in rows]) for c in stage_cols}

    # E2E = last_stage - first_stage (seconds)
    first, last = stage_cols[0], stage_cols[-1]
    e2e_s = stage_arrays[last] - stage_arrays[first]
    result["e2e_ms"] = stats_summary(e2e_s * 1000.0)

    # Per-stage deltas
    deltas = {}
    for i in range(1, len(stage_cols)):
        d_ms = (stage_arrays[stage_cols[i]] - stage_arrays[stage_cols[i-1]]) * 1000.0
        deltas[f"{stage_cols[i-1]} → {stage_cols[i]}"] = stats_summary(d_ms)
    result["stage_deltas_ms"] = deltas

    # interval_ms column (already in ms per pipeline_main)
    if 'interval_ms' in fields:
        intv = np.array([float(r['interval_ms']) if r['interval_ms'] not in ('', None) else np.nan
                         for r in rows])
        result["interval_ms"] = stats_summary(intv)

    # Histogram
    finite = result["e2e_ms"]
    if finite.get("finite", 0) > 0:
        e2e_ms_arr = e2e_s * 1000.0
        finite_arr = e2e_ms_arr[np.isfinite(e2e_ms_arr)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(finite_arr, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.percentile(finite_arr, 50), color='g', ls='--',
                   label=f'p50={np.percentile(finite_arr, 50):.1f}ms')
        ax.axvline(np.percentile(finite_arr, 99), color='r', ls='--',
                   label=f'p99={np.percentile(finite_arr, 99):.1f}ms')
        ax.axvline(20.0, color='k', ls=':', label='HARD LIMIT 20ms')
        ax.set_xlabel("E2E latency (ms)"); ax.set_ylabel("frames")
        ax.set_title(f"E2E latency dist ({path.name})")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        png = out_dir / "e2e_latency_hist.png"
        fig.savefig(png, dpi=100); plt.close(fig)
        result["plot_latency_hist"] = str(png)

    return result


def analyze_plan_d(path: Path) -> dict:
    if path is None or not path.exists():
        return {"present": False}
    with open(path) as f:
        text = f.read()
    result = {"present": True, "path": str(path), "size_bytes": len(text)}
    transitions = re.findall(r"cascade.*L(\d).*L(\d)", text)
    result["cascade_transitions"] = [{"from": int(a), "to": int(b)} for a, b in transitions]
    strides = re.findall(r"stride[_ ]count[:\s=]+(\d+)", text)
    if strides:
        result["max_stride_count"] = max(int(s) for s in strides)
    ready_true = len(re.findall(r"is_ready.*True", text))
    ready_false = len(re.findall(r"is_ready.*False", text))
    total = ready_true + ready_false
    if total > 0:
        result["is_ready_true_ratio"] = ready_true / total
    result["error_lines"]   = len(re.findall(r"\b(ERROR|Traceback)\b", text, re.IGNORECASE))
    result["warning_lines"] = len(re.findall(r"\bWARN\b", text, re.IGNORECASE))
    return result


def emit_markdown(stats: dict, out_path: Path) -> str:
    lines = []
    lines.append(f"# Walking session analysis — {stats.get('session_dir', '')}\n")

    npz = stats.get("npz", {})
    if npz.get("present"):
        if "error" in npz:
            lines.append(f"## Pose npz — schema error: {npz['error']}")
        else:
            lines.append(f"## Pose dump ({npz.get('frames', '?')} frames, "
                         f"{npz.get('duration_s', 0):.1f}s, {npz.get('fps', 0):.1f}Hz)")
            lines.append(f"- valid ratio: {npz.get('valid_ratio', 0)*100:.1f}%")
            lines.append(f"- method: {npz.get('method')}")
            for jname, jstat in npz.get("joints", {}).items():
                if "p50" in jstat:
                    lines.append(f"- {jname:18s}: p50={jstat['p50']:+.3f}  "
                                 f"p99={jstat['p99']:+.3f}  std={jstat['std']:.3f}")
            for k in ("plot_joint_angles", "plot_hip_z"):
                if k in npz:
                    lines.append(f"\n→ `{npz[k]}`")
    else:
        lines.append("## Pose dump — **MISSING** (pipeline did not produce npz)")

    tr = stats.get("trace", {})
    lines.append("\n## RT trace")
    if tr.get("present") and not tr.get("empty"):
        e2e = tr.get("e2e_ms", {})
        if "p50" in e2e:
            lines.append(f"- rows: {tr.get('rows', 0)}")
            lines.append(f"- e2e p50/p95/p99 (ms): {e2e['p50']:.2f} / {e2e['p95']:.2f} / {e2e['p99']:.2f}")
            lines.append(f"- e2e max (ms): {e2e['max']:.2f}")
            lines.append(f"- HARD LIMIT 20ms 위반: {(e2e['max'] > 20)}")
        if "interval_ms" in tr and "p50" in tr["interval_ms"]:
            iv = tr["interval_ms"]
            lines.append(f"- interval p50/p95 (ms): {iv['p50']:.2f} / {iv['p95']:.2f}")
        if "plot_latency_hist" in tr:
            lines.append(f"\n→ `{tr['plot_latency_hist']}`")
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
    out_path.write_text(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--plan-d", default=None)
    ap.add_argument("--out-dir", default=None)
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
    print(f"=== Outputs ===")
    print(f"  JSON     : {json_path}")
    print(f"  Markdown : {out_dir / 'analysis.md'}")
    for k in ("plot_joint_angles", "plot_hip_z"):
        if k in stats["npz"]:
            print(f"  {k:18s}: {stats['npz'][k]}")
    if "plot_latency_hist" in stats["trace"]:
        print(f"  plot_latency_hist : {stats['trace']['plot_latency_hist']}")


if __name__ == "__main__":
    main()
