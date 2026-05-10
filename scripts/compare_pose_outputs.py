"""compare_pose_outputs.py — paired pose 출력 비교 + clinical RMSE pass/fail.

Codex consult Q8 (2026-05-10) + Codex review f0884e5 (4 P1 fixes 적용).

Phase B (yolo26n) 진입 시 accuracy 검증 도구.

사용법 (Phase B 진입 시):
    python3 scripts/compare_pose_outputs.py \\
        --reference dumps/yolo26s-lower6/ \\
        --candidate dumps/yolo26n-lower6/ \\
        --output reports/yolo26n_vs_s.csv

Pass criteria (Codex Q3):
    - per-keypoint 2D RMSE max ≤ 6 px
    - 3D RMSE: hip/knee ≤ 15 mm, ankle ≤ 25 mm
    - valid drop ≤ 1 %
    - side bias |paired L-R 3D diff| mean ≤ 10 mm

Codex review fixes (P1):
    - paired dump file mismatch 시 fail (--no-strict 로 override)
    - per-keypoint 의 paired valid sample < min_valid_per_kp 면 fail (NaN, no perfect)
    - side bias = paired per-frame (L_diff - R_diff) 의 norm mean
      (기존 abs(rmse_l - rmse_r) 는 systematic bias 통과 가능)
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
MIN_VALID_PER_KP_DEFAULT = 30


def load_paired_dumps(ref_dir: Path, cand_dir: Path, strict: bool = True):
    """Load paired .npz dumps. strict=True 면 file 매칭 mismatch fail."""
    ref_files = sorted(ref_dir.glob("*.npz"))
    cand_files = sorted(cand_dir.glob("*.npz"))
    if not ref_files:
        sys.exit(f"ERROR: no .npz in {ref_dir}")
    if not cand_files:
        sys.exit(f"ERROR: no .npz in {cand_dir}")

    ref_names = {p.name for p in ref_files}
    cand_names = {p.name for p in cand_files}
    only_ref = ref_names - cand_names
    only_cand = cand_names - ref_names

    if strict and (only_ref or only_cand):
        sys.exit(
            f"ERROR: paired dump mismatch — "
            f"only-in-ref={len(only_ref)}, only-in-cand={len(only_cand)}. "
            f"--no-strict 로 부분 overlap 허용."
        )
    if only_ref or only_cand:
        print(
            f"WARN: paired dump mismatch (--no-strict): "
            f"only-in-ref={len(only_ref)}, only-in-cand={len(only_cand)}",
            file=sys.stderr,
        )

    pairs = []
    for ref_path in ref_files:
        cand_path = cand_dir / ref_path.name
        if not cand_path.exists():
            continue
        ref = np.load(ref_path)
        cand = np.load(cand_path)
        pairs.append((ref, cand))
    return pairs


def compute_rmse(pairs, min_valid_per_kp: int):
    """Codex review P1 — zero valid samples 면 NaN (PASS 회피)."""
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

    rmse_2d = np.full(6, np.nan)
    rmse_3d = np.full(6, np.nan)
    valid_per_kp = np.zeros(6, dtype=int)
    for k in range(6):
        mask = valid_both[:, k]
        valid_per_kp[k] = int(mask.sum())
        if valid_per_kp[k] >= min_valid_per_kp:
            rmse_2d[k] = float(np.sqrt(np.mean(err_2d[mask, k] ** 2)))
            rmse_3d[k] = float(np.sqrt(np.mean(err_3d_mm[mask, k] ** 2)))

    valid_rate_ref = float(valid_ref.mean())
    valid_rate_cand = float(valid_cand.mean())
    valid_drop = valid_rate_ref - valid_rate_cand

    # Codex review P1 — paired side bias = mean(|L_diff - R_diff|) over frames where
    # 둘 다 valid. systematic same-side bias 가 scalar RMSE 차이만 보면 통과 가능 →
    # paired per-frame difference vector 의 norm 의 mean 으로 정의.
    side_bias_3d = np.full(3, np.nan)
    joints = ("hip", "knee", "ankle")
    for j_idx, joint in enumerate(joints):
        l = KP_NAMES.index(f"L_{joint}")
        r = KP_NAMES.index(f"R_{joint}")
        joint_mask = valid_both[:, l] & valid_both[:, r]
        if int(joint_mask.sum()) < min_valid_per_kp:
            continue
        l_diff_mm = (kp3d_ref[:, l, :] - kp3d_cand[:, l, :]) * 1000.0
        r_diff_mm = (kp3d_ref[:, r, :] - kp3d_cand[:, r, :]) * 1000.0
        paired = np.linalg.norm(l_diff_mm[joint_mask] - r_diff_mm[joint_mask], axis=1)
        side_bias_3d[j_idx] = float(np.mean(paired))

    return (rmse_2d, rmse_3d, valid_drop, valid_rate_ref, valid_rate_cand,
            valid_per_kp, side_bias_3d)


def evaluate(rmse_2d, rmse_3d, valid_drop, valid_per_kp, side_bias_3d,
             min_valid_per_kp: int):
    fails = []
    for k, name in enumerate(KP_NAMES):
        if valid_per_kp[k] < min_valid_per_kp:
            fails.append(
                f"{name} valid samples {valid_per_kp[k]} < min {min_valid_per_kp}"
            )
            continue  # NaN RMSE — 추가 check 의미 없음
        if rmse_2d[k] > THRESHOLD_2D_PX:
            fails.append(f"{name} 2D RMSE {rmse_2d[k]:.2f}px > {THRESHOLD_2D_PX}px")
        joint = name.split("_")[1]
        thr = THRESHOLD_3D_MM[joint]
        if rmse_3d[k] > thr:
            fails.append(f"{name} 3D RMSE {rmse_3d[k]:.2f}mm > {thr}mm")
    if valid_drop > VALID_DROP_MAX:
        fails.append(f"valid drop {valid_drop:.4f} > {VALID_DROP_MAX}")
    for j_idx, joint in enumerate(("hip", "knee", "ankle")):
        bias = side_bias_3d[j_idx]
        if np.isnan(bias):
            fails.append(f"{joint} side bias undefined (paired valid samples 부족)")
        elif bias > SIDE_BIAS_MM_MAX:
            fails.append(f"{joint} L/R paired bias {bias:.2f}mm > {SIDE_BIAS_MM_MAX}mm")
    return fails


def write_csv(path: Path, rmse_2d, rmse_3d, valid_drop, valid_per_kp, side_bias_3d):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keypoint", "rmse_2d_px", "rmse_3d_mm",
                    "thr_2d_px", "thr_3d_mm", "valid_n", "result"])
        for k, name in enumerate(KP_NAMES):
            joint = name.split("_")[1]
            thr_3d = THRESHOLD_3D_MM[joint]
            r2 = "NaN" if np.isnan(rmse_2d[k]) else f"{rmse_2d[k]:.3f}"
            r3 = "NaN" if np.isnan(rmse_3d[k]) else f"{rmse_3d[k]:.3f}"
            ok = (not np.isnan(rmse_2d[k]) and rmse_2d[k] <= THRESHOLD_2D_PX
                  and not np.isnan(rmse_3d[k]) and rmse_3d[k] <= thr_3d)
            w.writerow([name, r2, r3, THRESHOLD_2D_PX, thr_3d,
                        valid_per_kp[k], "PASS" if ok else "FAIL"])
        w.writerow(["valid_drop", "", f"{valid_drop:.4f}", "", VALID_DROP_MAX, "",
                    "PASS" if valid_drop <= VALID_DROP_MAX else "FAIL"])
        for j_idx, joint in enumerate(("hip", "knee", "ankle")):
            b = side_bias_3d[j_idx]
            br = "NaN" if np.isnan(b) else f"{b:.3f}"
            ok = not np.isnan(b) and b <= SIDE_BIAS_MM_MAX
            w.writerow([f"side_bias_{joint}", "", br, "", SIDE_BIAS_MM_MAX, "",
                        "PASS" if ok else "FAIL"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--output", default=Path("rmse_report.csv"), type=Path)
    ap.add_argument("--no-strict", action="store_true",
                    help="paired dump file 부분 overlap 허용 (default: strict fail)")
    ap.add_argument("--min-valid-per-kp", type=int,
                    default=MIN_VALID_PER_KP_DEFAULT,
                    help=f"per-keypoint 최소 paired valid samples "
                         f"(default {MIN_VALID_PER_KP_DEFAULT})")
    args = ap.parse_args()

    pairs = load_paired_dumps(args.reference, args.candidate, strict=not args.no_strict)
    (rmse_2d, rmse_3d, valid_drop, vr_ref, vr_cand,
     valid_per_kp, side_bias_3d) = compute_rmse(pairs, args.min_valid_per_kp)
    fails = evaluate(rmse_2d, rmse_3d, valid_drop, valid_per_kp,
                     side_bias_3d, args.min_valid_per_kp)

    print(f"Compared {len(pairs)} frame pairs")
    print(f"Valid rate: ref={vr_ref:.4f}, cand={vr_cand:.4f}, drop={valid_drop:.4f}")
    print()
    for k, name in enumerate(KP_NAMES):
        joint = name.split("_")[1]
        thr_3d = THRESHOLD_3D_MM[joint]
        if valid_per_kp[k] < args.min_valid_per_kp:
            print(f"  {name:8s} valid={valid_per_kp[k]:4d} (insufficient — NaN)")
            continue
        ok_2d = rmse_2d[k] <= THRESHOLD_2D_PX
        ok_3d = rmse_3d[k] <= thr_3d
        m2d = "✓" if ok_2d else "✗"
        m3d = "✓" if ok_3d else "✗"
        print(f"  {name:8s} 2D={rmse_2d[k]:5.2f}px ({m2d}≤{THRESHOLD_2D_PX}) "
              f"3D={rmse_3d[k]:6.2f}mm ({m3d}≤{thr_3d}) n={valid_per_kp[k]}")

    print()
    for j_idx, joint in enumerate(("hip", "knee", "ankle")):
        b = side_bias_3d[j_idx]
        if np.isnan(b):
            print(f"  side_bias_{joint}: NaN (paired valid 부족)")
        else:
            ok = b <= SIDE_BIAS_MM_MAX
            mark = "✓" if ok else "✗"
            print(f"  side_bias_{joint}: {b:.2f}mm ({mark}≤{SIDE_BIAS_MM_MAX})")

    write_csv(args.output, rmse_2d, rmse_3d, valid_drop, valid_per_kp, side_bias_3d)
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
