"""compare_pose_outputs.py — paired pose 출력 비교 + clinical RMSE pass/fail.

Codex consult Q8 (2026-05-10). Phase B (yolo26n) 진입 시 accuracy 검증 도구.

사용법 (Phase B 진입 시):
    python3 scripts/compare_pose_outputs.py \\
        --reference dumps/yolo26s-lower6/ \\
        --candidate dumps/yolo26n-lower6/ \\
        --output reports/yolo26n_vs_s.csv

Pass criteria (Codex Q3):
    - per-keypoint 2D RMSE max ≤ 6 px
    - 3D RMSE: hip/knee ≤ 15 mm, ankle ≤ 25 mm
    - valid drop ≤ 1 %
    - side bias |L-R| mean ≤ 10 mm
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


KP_NAMES = ["L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]
THRESHOLD_2D_PX = 6.0
THRESHOLD_3D_MM = {"hip": 15.0, "knee": 15.0, "ankle": 25.0}
VALID_DROP_MAX = 0.01
SIDE_BIAS_MM_MAX = 10.0


def load_paired_dumps(ref_dir: Path, cand_dir: Path):
    ref_files = sorted(ref_dir.glob("*.npz"))
    if not ref_files:
        sys.exit(f"ERROR: no .npz in {ref_dir}")
    pairs = []
    for ref_path in ref_files:
        cand_path = cand_dir / ref_path.name
        if not cand_path.exists():
            continue
        ref = np.load(ref_path)
        cand = np.load(cand_path)
        pairs.append((ref, cand))
    return pairs


def compute_rmse(pairs):
    n = len(pairs)
    if n == 0:
        sys.exit("ERROR: no paired frames found")
    diff_2d = np.zeros((n, 6, 2))
    kp3d_ref = np.zeros((n, 6, 3))
    kp3d_cand = np.zeros((n, 6, 3))
    valid_ref = np.zeros((n, 6), dtype=bool)
    valid_cand = np.zeros((n, 6), dtype=bool)

    for i, (ref, cand) in enumerate(pairs):
        diff_2d[i] = ref["kpts_2d"] - cand["kpts_2d"]
        kp3d_ref[i] = ref["kpts_3d_m"]
        kp3d_cand[i] = cand["kpts_3d_m"]
        valid_ref[i] = ref["valid_mask"]
        valid_cand[i] = cand["valid_mask"]

    valid_both = valid_ref & valid_cand
    err_2d = np.linalg.norm(diff_2d, axis=2)
    err_3d_mm = np.linalg.norm((kp3d_ref - kp3d_cand) * 1000.0, axis=2)

    rmse_2d = np.zeros(6)
    rmse_3d = np.zeros(6)
    for k in range(6):
        mask = valid_both[:, k]
        if mask.any():
            rmse_2d[k] = float(np.sqrt(np.mean(err_2d[mask, k] ** 2)))
            rmse_3d[k] = float(np.sqrt(np.mean(err_3d_mm[mask, k] ** 2)))

    valid_rate_ref = float(valid_ref.mean())
    valid_rate_cand = float(valid_cand.mean())
    valid_drop = valid_rate_ref - valid_rate_cand

    return rmse_2d, rmse_3d, valid_drop, valid_rate_ref, valid_rate_cand


def evaluate(rmse_2d, rmse_3d, valid_drop):
    fails = []
    for k, name in enumerate(KP_NAMES):
        if rmse_2d[k] > THRESHOLD_2D_PX:
            fails.append(f"{name} 2D RMSE {rmse_2d[k]:.2f}px > {THRESHOLD_2D_PX}px")
        joint = name.split("_")[1]
        thr = THRESHOLD_3D_MM[joint]
        if rmse_3d[k] > thr:
            fails.append(f"{name} 3D RMSE {rmse_3d[k]:.2f}mm > {thr}mm")
    if valid_drop > VALID_DROP_MAX:
        fails.append(f"valid drop {valid_drop:.4f} > {VALID_DROP_MAX}")
    for joint in ("hip", "knee", "ankle"):
        l_idx = KP_NAMES.index(f"L_{joint}")
        r_idx = KP_NAMES.index(f"R_{joint}")
        bias = abs(rmse_3d[l_idx] - rmse_3d[r_idx])
        if bias > SIDE_BIAS_MM_MAX:
            fails.append(f"{joint} L/R bias {bias:.2f}mm > {SIDE_BIAS_MM_MAX}mm")
    return fails


def write_csv(path: Path, rmse_2d, rmse_3d, valid_drop):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keypoint", "rmse_2d_px", "rmse_3d_mm",
                    "thr_2d_px", "thr_3d_mm", "result"])
        for k, name in enumerate(KP_NAMES):
            joint = name.split("_")[1]
            thr_3d = THRESHOLD_3D_MM[joint]
            ok = rmse_2d[k] <= THRESHOLD_2D_PX and rmse_3d[k] <= thr_3d
            w.writerow([name, f"{rmse_2d[k]:.3f}", f"{rmse_3d[k]:.3f}",
                        THRESHOLD_2D_PX, thr_3d, "PASS" if ok else "FAIL"])
        w.writerow(["valid_drop", "", f"{valid_drop:.4f}", "", VALID_DROP_MAX,
                    "PASS" if valid_drop <= VALID_DROP_MAX else "FAIL"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--output", default=Path("rmse_report.csv"), type=Path)
    args = ap.parse_args()

    pairs = load_paired_dumps(args.reference, args.candidate)
    rmse_2d, rmse_3d, valid_drop, vr_ref, vr_cand = compute_rmse(pairs)
    fails = evaluate(rmse_2d, rmse_3d, valid_drop)

    print(f"Compared {len(pairs)} frame pairs")
    print(f"Valid rate: ref={vr_ref:.4f}, cand={vr_cand:.4f}, drop={valid_drop:.4f}")
    print()
    for k, name in enumerate(KP_NAMES):
        joint = name.split("_")[1]
        thr_3d = THRESHOLD_3D_MM[joint]
        ok_2d = rmse_2d[k] <= THRESHOLD_2D_PX
        ok_3d = rmse_3d[k] <= thr_3d
        m2d = "✓" if ok_2d else "✗"
        m3d = "✓" if ok_3d else "✗"
        print(f"  {name:8s} 2D={rmse_2d[k]:5.2f}px ({m2d}≤{THRESHOLD_2D_PX}) "
              f"3D={rmse_3d[k]:6.2f}mm ({m3d}≤{thr_3d})")

    write_csv(args.output, rmse_2d, rmse_3d, valid_drop)
    print(f"\nReport: {args.output}")

    if fails:
        print("\n=== FAIL ===")
        for f in fails:
            print(f"  ✗ {f}")
        return 1
    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
