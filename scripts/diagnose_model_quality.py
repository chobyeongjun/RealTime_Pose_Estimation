"""Model + pipeline-stage diagnosis on an existing SVO.

Splits 'valid_ratio 29.2%' into root causes by re-processing the SVO and
counting frame-level fail categories:

    yolo_no_detection : YOLO predict returned no person box
    conf_low_kp       : at least one of 6 joints has conf < CONF_THRESHOLD
    depth_nan         : 2D keypoint OK but ZED depth NaN at that pixel
    depth_oor         : depth finite but < 0.1m or > 3.0m
    triplet_missing   : left and right both missing a hip/knee/ankle
    ok                : passes compute_joint_state.valid

Also dumps per-keypoint conf distribution (P50/P95) and depth distribution
so we can tell whether the model is fine but the threshold is strict (B),
the model itself is producing low-conf detections (A), or the depth map
is the bottleneck.

PNG output (matplotlib agg) — no GUI needed.

Usage:
    python3 scripts/diagnose_model_quality.py recordings/walking_*/walking_*.svo2
    python3 scripts/diagnose_model_quality.py walking.svo2 --max-frames 1000
    python3 scripts/diagnose_model_quality.py walking.svo2 --dump-frames 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add repo paths
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "perception" / "benchmarks"))
sys.path.insert(0, str(REPO / "src" / "perception" / "realtime"))

JOINT_NAMES_6 = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
                 'left_ankle', 'right_ankle']
CONF_THRESHOLD = 0.5
DEPTH_MIN, DEPTH_MAX = 0.1, 3.0


def categorize_frame(result, depth_at_kp) -> str:
    """Return one of: ok | yolo_no_box | conf_low_kp | depth_nan | depth_oor | triplet_missing

    Codex P2: TRTPoseEngine sets detected=True only when ≥3 keypoints have conf>0.3.
    If detected=False AND confidences/keypoints_2d are non-empty, the model DID
    find a person but the keypoints were low-confidence — that's 'conf_low_kp',
    not 'no detection'. Check confidences first, then fall back to no-box.
    """
    confs = getattr(result, 'confidences', {}) or {}
    kp2d  = getattr(result, 'keypoints_2d', {}) or {}

    # If the engine returned absolutely nothing → no person box at all.
    if not confs and not kp2d:
        return "yolo_no_box"

    # which joints pass conf threshold
    conf_ok = {n for n in JOINT_NAMES_6 if confs.get(n, 0.0) >= CONF_THRESHOLD}
    if len(conf_ok) < 3:
        return "conf_low_kp"

    # depth filter on conf-passing joints
    in_range, nan_or_inf, out_of_range = set(), set(), set()
    for n in conf_ok:
        z = depth_at_kp.get(n, float('nan'))
        if not np.isfinite(z):
            nan_or_inf.add(n)
        elif z < DEPTH_MIN or z > DEPTH_MAX:
            out_of_range.add(n)
        else:
            in_range.add(n)

    # triplet check on in-range joints
    def _triplet(side: str):
        return all(f"{side}_{j}" in in_range for j in ("hip", "knee", "ankle"))
    has_left, has_right = _triplet("left"), _triplet("right")

    if has_left or has_right:
        return "ok"
    if nan_or_inf and not out_of_range:
        return "depth_nan"
    if out_of_range and not nan_or_inf:
        return "depth_oor"
    if nan_or_inf or out_of_range:
        return "depth_nan"  # mixed → call it nan (more common cause)
    return "triplet_missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svo")
    ap.add_argument("--max-frames", type=int, default=300,
                    help="Process at most N frames (300=5s @ 60Hz). Use -1 for all.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write JSON/PNG. Default: alongside SVO.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--engine", default=None,
                    help="TRT engine path; default = src/perception/models/yolo26s-lower6-v2-640.engine")
    ap.add_argument("--dump-frames", type=int, default=0,
                    help="Save first N processed frames as PNG with keypoints overlaid.")
    args = ap.parse_args()

    svo_path = Path(args.svo)
    if not svo_path.is_file():
        print(f"ERROR: SVO not found: {svo_path}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.out_dir or svo_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — Jetson only
    try:
        import pyzed.sl as sl
    except ImportError:
        print("pyzed.sl missing", file=sys.stderr); sys.exit(2)
    try:
        from trt_pose_engine import TRTPoseEngine
    except ImportError as e:
        print(f"TRT engine import failed: {e}", file=sys.stderr); sys.exit(3)

    engine_path = args.engine or str(REPO / "src" / "perception" / "models" /
                                     f"yolo26s-lower6-v2-{args.imgsz}.engine")
    if not Path(engine_path).is_file():
        print(f"ERROR: engine not found: {engine_path}", file=sys.stderr)
        print("       Build first: trtexec --onnx=... --saveEngine=...", file=sys.stderr)
        sys.exit(4)

    print(f"Opening SVO:    {svo_path}")
    print(f"Opening engine: {engine_path}")
    print()

    # Open SVO
    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"open fail: {err}", file=sys.stderr); sys.exit(5)

    total = cam.get_svo_number_of_frames()
    print(f"SVO total frames: {total}")
    max_n = total if args.max_frames < 0 else min(args.max_frames, total)
    print(f"Will process:     {max_n}")

    # Load TRT engine — Codex P1: __init__ only stores config, load() creates
    # CUDA context + tensors. Without this, predict() crashes on _input_tensor.
    model = TRTPoseEngine(engine_path, imgsz=args.imgsz)
    model.load()

    rt = sl.RuntimeParameters()
    img = sl.Mat()
    dep = sl.Mat()

    # Counters + distributions
    categories = Counter()
    per_joint_conf = {n: [] for n in JOINT_NAMES_6}
    per_joint_depth = {n: [] for n in JOINT_NAMES_6}
    # Codex P2: TRTPoseEngine has no result.box_conf field. Use max joint conf
    # as a proxy for 'box presence' confidence.
    max_kp_confs = []

    t0 = time.monotonic()
    dumped_frames = 0
    for fi in range(max_n):
        e = cam.grab(rt)
        if e != sl.ERROR_CODE.SUCCESS:
            break
        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(dep, sl.MEASURE.DEPTH)
        bgra = img.get_data()
        depth_map = dep.get_data()
        import cv2
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        result = model.predict(bgr)
        # Codex P2 fix: derive box-presence proxy from max keypoint conf.
        frame_confs = getattr(result, 'confidences', {}) or {}
        max_kp_confs.append(max(frame_confs.values()) if frame_confs else 0.0)

        # Sample depth at each detected keypoint pixel
        depth_at_kp = {}
        kp2d = getattr(result, 'keypoints_2d', {}) or {}
        confs = getattr(result, 'confidences', {}) or {}
        for n in JOINT_NAMES_6:
            per_joint_conf[n].append(confs.get(n, 0.0))
            if n in kp2d:
                u, v = kp2d[n]
                ui = int(round(u)); vi = int(round(v))
                if 0 <= vi < depth_map.shape[0] and 0 <= ui < depth_map.shape[1]:
                    z = float(depth_map[vi, ui])
                    depth_at_kp[n] = z
                    per_joint_depth[n].append(z)

        cat = categorize_frame(result, depth_at_kp)
        categories[cat] += 1

        if dumped_frames < args.dump_frames:
            vis = bgr.copy()
            for n in JOINT_NAMES_6:
                if n in kp2d:
                    u, v = kp2d[n]
                    c = confs.get(n, 0.0)
                    color = (0, 255, 0) if c >= CONF_THRESHOLD else (0, 0, 255)
                    cv2.circle(vis, (int(u), int(v)), 6, color, 2)
                    cv2.putText(vis, f"{n[:5]}:{c:.2f}", (int(u)+6, int(v)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            max_c = max(confs.values()) if confs else 0.0
            cv2.putText(vis, f"frame {fi} cat={cat} max_kp_conf={max_c:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            png = out_dir / f"diag_frame_{fi:04d}.png"
            cv2.imwrite(str(png), vis)
            dumped_frames += 1

        if (fi+1) % 50 == 0:
            rate = (fi+1) / max(time.monotonic()-t0, 1e-3)
            print(f"  [{fi+1:4d}/{max_n}]  rate={rate:.1f} f/s  cats={dict(categories)}",
                  flush=True)

    cam.close()
    dur = time.monotonic() - t0
    print(f"\nDone {sum(categories.values())} frames in {dur:.1f}s "
          f"= {sum(categories.values())/max(dur,1e-3):.1f} f/s")
    print()

    # Aggregate
    total_n = sum(categories.values())
    cat_pct = {k: 100.0 * v / max(total_n, 1) for k, v in categories.items()}
    print("=== Category breakdown ===")
    for k, v in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}  {v:5d}  {cat_pct[k]:5.1f}%")

    print("\n=== Per-joint confidence (P50 / P95 / mean) ===")
    conf_stats = {}
    for n in JOINT_NAMES_6:
        a = np.array(per_joint_conf[n], dtype=float)
        if a.size == 0: continue
        conf_stats[n] = {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
            "above_threshold_pct": float(np.mean(a >= CONF_THRESHOLD) * 100.0),
        }
        s = conf_stats[n]
        print(f"  {n:14s}  p50={s['p50']:.2f}  p95={s['p95']:.2f}  "
              f"mean={s['mean']:.2f}  >0.5: {s['above_threshold_pct']:5.1f}%")

    print("\n=== Per-joint depth (P50 / P95 / NaN ratio) ===")
    depth_stats = {}
    for n in JOINT_NAMES_6:
        a = np.array(per_joint_depth[n], dtype=float)
        if a.size == 0: continue
        finite = a[np.isfinite(a)]
        depth_stats[n] = {
            "samples": int(a.size),
            "nan_ratio": float((a.size - finite.size) / a.size),
            "p50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
            "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
            "in_range_pct": float(np.mean((finite >= DEPTH_MIN) & (finite <= DEPTH_MAX)) * 100.0) if finite.size else 0.0,
        }
        s = depth_stats[n]
        print(f"  {n:14s}  N={s['samples']:5d}  NaN={s['nan_ratio']*100:4.1f}%  "
              f"p50={s['p50']:.2f}m  p95={s['p95']:.2f}m  in[0.1,3.0]: {s['in_range_pct']:5.1f}%")

    print("\n=== Max-keypoint confidence per frame (box-presence proxy) ===")
    bc = np.array(max_kp_confs)
    print(f"  p50={np.percentile(bc, 50):.2f}  p95={np.percentile(bc, 95):.2f}  "
          f"max={np.max(bc):.2f}  >0.5: {np.mean(bc >= 0.5)*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 4))
    names = sorted(categories.keys(), key=lambda x: -categories[x])
    vals = [categories[n] for n in names]
    bars = ax.bar(names, vals, color=['#2ecc71' if n=='ok' else '#e74c3c' for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v}\n({100*v/total_n:.1f}%)",
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("frames")
    ax.set_title(f"Frame validity categories ({total_n} frames)")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    cat_png = out_dir / "diag_categories.png"
    fig.savefig(cat_png, dpi=100); plt.close(fig)
    print(f"\nCategory plot: {cat_png}")

    # Per-joint conf histogram
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, n in zip(axes.flatten(), JOINT_NAMES_6):
        a = np.array(per_joint_conf[n])
        if a.size == 0: continue
        ax.hist(a, bins=30, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.axvline(CONF_THRESHOLD, color='r', ls='--', label=f'thr={CONF_THRESHOLD}')
        ax.set_title(f"{n}\n>0.5: {np.mean(a>=CONF_THRESHOLD)*100:.1f}%")
        ax.set_xlabel("conf"); ax.legend(fontsize=8)
    fig.suptitle("Per-joint confidence distribution")
    plt.tight_layout()
    conf_png = out_dir / "diag_kp_conf.png"
    fig.savefig(conf_png, dpi=100); plt.close(fig)
    print(f"Conf plot: {conf_png}")

    # Dump JSON
    report = {
        "svo":  str(svo_path),
        "frames_processed": total_n,
        "categories": dict(categories),
        "category_pct": cat_pct,
        "conf_stats": conf_stats,
        "depth_stats": depth_stats,
        "max_kp_conf_summary": {
            "p50": float(np.percentile(bc, 50)),
            "p95": float(np.percentile(bc, 95)),
            "above_threshold_pct": float(np.mean(bc >= 0.5) * 100.0),
        },
    }
    json_path = out_dir / "diag_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"JSON:      {json_path}")
    print()
    print("=== HOW TO READ ===")
    print("  ok                  → valid frame (= pipeline_main valid=True)")
    print("  yolo_no_box         → engine returned no keypoints at all")
    print("  conf_low_kp         → joint conf < 0.5 (env/occlusion → A; or model itself bad → C)")
    print("  depth_nan           → keypoint OK but ZED depth NaN at pixel (env, opaque clothing, distance)")
    print("  depth_oor           → depth finite but outside [0.1, 3.0]m (too close/far)")
    print("  triplet_missing    → neither left nor right side has hip+knee+ankle")
    print()
    print("If 'conf_low_kp' dominates AND per-joint conf p50 < 0.5 →")
    print("  walking is too far / partial occlusion / clothing issue (A).")
    print("If 'conf_low_kp' dominates AND per-joint conf p50 >= 0.5 →")
    print("  model produces inconsistent confidences (C).")
    print("If 'depth_nan' dominates → ZED depth quality issue (lighting, baseline, distance).")


if __name__ == "__main__":
    main()
