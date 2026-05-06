"""World frame verification — run on Jetson BEFORE experiments.

Usage:
    python3 -m perception.CUDA_Stream.verify_world_frame

Pass conditions:
  1. world_frame_applied = True (IMU warmup succeeded)
  2. Ankle Y > Knee Y > Hip Y (world Y = gravity, down = positive)
  3. Ankle Z ≈ Knee Z (standing straight: same horizontal depth)

All checks print a clear PASS / FAIL with the actual numbers.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .keypoint_config import get_schema
from .shm_publisher import DEFAULT_NAME, ShmReader

SCHEMA = get_schema("lowlimb6")

# Joint indices in lowlimb6
IDX = {name: SCHEMA.index(name) for name in SCHEMA.keypoints}


def _mean_or_none(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def run(n_frames: int = 150) -> bool:
    """Read SHM for n_frames, report gravity alignment and world_frame status."""
    try:
        reader = ShmReader(name=DEFAULT_NAME, expected_k=SCHEMA.num_keypoints)
    except FileNotFoundError:
        print(f"\n[ERROR] SHM /dev/shm/{DEFAULT_NAME} not found.")
        print("  → Start run_stream_demo first:  sudo ./launch_clean.sh 30")
        return False

    print(f"\n{'='*60}")
    print("  World Frame Verification")
    print(f"  Reading {n_frames} frames from /dev/shm/{DEFAULT_NAME}")
    print(f"{'='*60}")

    world_frame_flags = []
    hip_y = []
    knee_y = []
    ankle_y = []
    knee_z = []
    ankle_z = []
    valid_count = 0
    read_count = 0

    t0 = time.monotonic()
    while read_count < n_frames:
        data = reader.read()
        if data is None:
            time.sleep(0.005)
            continue

        # P1: ShmReader returns 12-tuple; this verifier only uses the original 9.
        frame_id, ts_ns, kpts_3d, kpt_conf, kpts_2d, box_conf, valid, depth_inv, world_flag = data[:9]
        read_count += 1

        if not valid:
            continue
        valid_count += 1

        # Only count world_frame_applied on valid frames — zeroed SHM
        # (pipeline warmup / just started) has world_flag=0 but those are
        # not meaningful frames. Counting them gives false 0% rate.
        world_frame_flags.append(int(world_flag))

        # collect Y (height) and Z (depth) for hip/knee/ankle
        # z > 0 guard: ankle_conf_threshold=0.72 zeroes depth when conf is
        # low (0.30–0.72), giving xyz=[0,0,0] which is NOT the ankle position.
        # Including such zero-depth coordinates would make ankle_Y=0 (camera
        # origin), which is ABOVE the knee → false gravity_alignment failure.
        def y(name):
            idx = IDX[name]
            if kpt_conf[idx] <= 0.30:
                return None
            z_val = float(kpts_3d[idx, 2])
            if z_val <= 0.0:   # zero-depth = invalid (ankle zeroed by pipeline)
                return None
            return float(kpts_3d[idx, 1])

        def z(name):
            idx = IDX[name]
            if kpt_conf[idx] <= 0.30:
                return None
            z_val = float(kpts_3d[idx, 2])
            if z_val <= 0.0:
                return None
            return z_val

        lh_y = y("left_hip");  rh_y = y("right_hip")
        lk_y = y("left_knee"); rk_y = y("right_knee")
        la_y = y("left_ankle"); ra_y = y("right_ankle")
        lk_z = z("left_knee"); rk_z = z("right_knee")
        la_z = z("left_ankle"); ra_z = z("right_ankle")

        hip_y.append(_mean_or_none([lh_y, rh_y]))
        knee_y.append(_mean_or_none([lk_y, rk_y]))
        ankle_y.append(_mean_or_none([la_y, ra_y]))
        knee_z.append(_mean_or_none([lk_z, rk_z]))
        ankle_z.append(_mean_or_none([la_z, ra_z]))

        if read_count % 30 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {read_count}/{n_frames} frames  ({elapsed:.1f}s)  valid={valid_count}", end="\r")

    reader.close()
    print()

    if valid_count == 0:
        print(f"\n[ERROR] 0 valid frames in {read_count} reads.")
        print("  Possible causes:")
        print("  1. Pipeline just started — wait 3s for warmup then retry")
        print("  2. No subject in camera view (valid=False from postprocessor)")
        print("  3. SHM is stale from a previous run (pipeline not running)")
        return False

    # ── Check 1: world_frame_applied ──────────────────────────────────
    wf_rate = np.mean(world_frame_flags) if world_frame_flags else 0.0
    wf_ok = wf_rate > 0.95

    print(f"\n{'─'*60}")
    print(f"  Check 1 — world_frame_applied flag")
    print(f"  Frames with world_frame=1: {wf_rate*100:.0f}%")
    if wf_ok:
        print("  → PASS: IMU world frame active")
    else:
        print("  → FAIL: IMU world frame NOT active (sagittal will be distorted)")
        print("    Fix: check Jetson log for 'IMU warmup' / 'No world rotation'")
        print("    Fallback: add --camera-pitch-deg <angle> to launch_clean.sh")

    # ── Check 2: gravity alignment (ankle Y > knee Y > hip Y) ─────────
    def valid_pairs(a, b):
        return [(x, y_) for x, y_ in zip(a, b)
                if x is not None and y_ is not None]

    hip_knee = valid_pairs(hip_y, knee_y)
    knee_ankle = valid_pairs(knee_y, ankle_y)

    hip_below_knee = sum(1 for h, k in hip_knee if k > h) / max(len(hip_knee), 1)
    knee_below_ankle = sum(1 for k, a in knee_ankle if a > k) / max(len(knee_ankle), 1)
    grav_ok = hip_below_knee > 0.85 and knee_below_ankle > 0.85

    mean_hip = np.nanmean([x for x in hip_y if x is not None]) if any(x is not None for x in hip_y) else float('nan')
    mean_knee = np.nanmean([x for x in knee_y if x is not None]) if any(x is not None for x in knee_y) else float('nan')
    mean_ankle = np.nanmean([x for x in ankle_y if x is not None]) if any(x is not None for x in ankle_y) else float('nan')

    print(f"\n{'─'*60}")
    print(f"  Check 2 — Gravity alignment (world Y = down+)")
    print(f"  Mean Y  →  Hip: {mean_hip:+.3f}m  Knee: {mean_knee:+.3f}m  Ankle: {mean_ankle:+.3f}m")
    print(f"  Knee Y > Hip Y  : {hip_below_knee*100:.0f}% of frames")
    print(f"  Ankle Y > Knee Y: {knee_below_ankle*100:.0f}% of frames")

    if grav_ok:
        print("  → PASS: Gravity correctly aligned (ankle is lowest)")
    else:
        print("  → FAIL: Gravity misaligned — camera frame used as world frame")
        print("    Ankle should have largest Y (furthest down). Hip smallest Y.")

    # ── Check 3: horizontal depth consistency ─────────────────────────
    kz_az_pairs = valid_pairs(knee_z, ankle_z)
    depth_diffs = [abs(kz - az) for kz, az in kz_az_pairs]
    mean_diff = np.mean(depth_diffs) if depth_diffs else float('nan')
    depth_ok = mean_diff < 0.15  # < 15cm depth diff when standing straight

    mean_kz = np.nanmean([x for x in knee_z if x is not None]) if any(x is not None for x in knee_z) else float('nan')
    mean_az = np.nanmean([x for x in ankle_z if x is not None]) if any(x is not None for x in ankle_z) else float('nan')

    print(f"\n{'─'*60}")
    print(f"  Check 3 — Horizontal depth (Z) consistency when standing")
    print(f"  Mean Z  →  Knee: {mean_kz:.3f}m  Ankle: {mean_az:.3f}m")
    print(f"  Mean |Z_knee - Z_ankle|: {mean_diff*100:.1f}cm")

    if depth_ok:
        print("  → PASS: Knee and ankle at similar depth (< 15cm difference)")
    else:
        print(f"  → FAIL: {mean_diff*100:.0f}cm depth gap — camera tilt not corrected")
        if not wf_ok:
            print("    (Expected: world frame is off, so this is consistent with Check 1 FAIL)")
        else:
            # Estimate camera pitch from depth difference and height difference
            if not np.isnan(mean_hip) and not np.isnan(mean_ankle) and not np.isnan(mean_kz) and not np.isnan(mean_az):
                vert_diff = abs(mean_ankle - mean_hip)  # height in world frame
                horiz_diff = abs(mean_az - mean_kz)
                if vert_diff > 0.1:
                    est_tilt = np.degrees(np.arctan2(horiz_diff, vert_diff))
                    print(f"    Estimated residual tilt: ~{est_tilt:.1f}°")

    # ── Summary ───────────────────────────────────────────────────────
    all_pass = wf_ok and grav_ok and depth_ok
    print(f"\n{'='*60}")
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED — ready for experiments")
        print(f"  Valid frames collected: {valid_count}/{read_count}")
        if not np.isnan(mean_hip) and not np.isnan(mean_ankle):
            height_m = abs(mean_ankle - mean_hip)
            print(f"  Hip-to-ankle height: {height_m*100:.0f}cm (plausible adult?)")
    else:
        failed = []
        if not wf_ok:   failed.append("world_frame_applied")
        if not grav_ok: failed.append("gravity_alignment")
        if not depth_ok: failed.append("depth_consistency")
        print(f"  RESULT: FAILED — {', '.join(failed)}")
        print()
        print("  Next steps:")
        if not wf_ok:
            print("  1. Check Jetson log: grep 'IMU\\|world' in run_stream_demo output")
            print("  2. If IMU fails: add --camera-pitch-deg <angle> to launch_clean.sh")
            print("     (measure camera angle from vertical, e.g. 35 if tilted 35° down)")
    print(f"{'='*60}\n")

    return all_pass


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
