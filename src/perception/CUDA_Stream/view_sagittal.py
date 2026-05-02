"""Sagittal viewer — mainline draw_sagittal style + sticky display.

Reads ``/dev/shm/hwalker_pose_cuda`` (CUDA_Stream publisher) and renders
the Y-Z (sagittal) plane in a separate process. Mirrors the proven
``verify_geometry.draw_sagittal`` behavior plus fixes for issues found
during 2026-04-21 integration:

  * **All 6 keypoints always shown** (no min_conf gate). Low-confidence
    points just dim — never removed from layout. This eliminates the
    "hip/ankle disappears when arm moves" problem.
  * **Sticky display**: keep last valid frame on screen during a brief
    invalid burst (no flicker / no blank frames).
  * **Left/right separation in sagittal**: in pure Y-Z projection both
    legs overlap. We add a tiny visual X-offset (±2cm screen-space) so
    you can see them apart while still being a true sagittal view.
  * **OpenCV only** — matplotlib backend was 14Hz on Jetson; cv2 hits
    60Hz easily. Single rendering path = simpler.
  * **Auto-fit always uses ALL 6 points** (regardless of conf), so a low
    conf joint never escapes the canvas (root cause of "잘림").

Usage (separate terminal while run_stream_demo is publishing):
    PYTHONPATH=src python3 -m perception.CUDA_Stream.view_sagittal

Controls:
    q / ESC  — quit
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from .keypoint_config import get_schema
from .shm_publisher import DEFAULT_NAME, ShmReader


# Sagittal skeleton: pelvis + L thigh + R thigh + L shank + R shank
SKELETON = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shm", default=DEFAULT_NAME.lstrip("/"))
    ap.add_argument("--schema", default="lowlimb6", choices=["coco17", "lowlimb6"])
    ap.add_argument("--fps", type=float, default=60.0,
                    help="render rate cap (cv2.waitKey is 1ms; rate limited by sleep)")
    ap.add_argument("--zoom", type=float, default=0.85,
                    help="auto-fit padding factor (0.5=tight, 1.0=fills canvas)")
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--low-conf", dest="low_conf", type=float, default=0.30,
                    help="below this conf the keypoint is drawn dim/grey")
    ap.add_argument("--leg-offset-px", dest="leg_offset_px", type=int, default=8,
                    help="visual screen-space X offset between left/right legs (px)")
    ap.add_argument("--calib-frames", dest="calib_frames", type=int, default=30,
                    help="standing-calibration window: first N valid frames to "
                         "learn the human X axis (left_hip→right_hip direction)")
    ap.add_argument("--no-calib", dest="no_calib", action="store_true",
                    help="disable standing calibration; use raw world Y-Z (legacy)")
    ap.add_argument("--display-mode", dest="display_mode", default="offset",
                    choices=["pure", "offset"],
                    help="pure: L/R keypoints overlap exactly (true sagittal, "
                         "colour-only distinction); "
                         "offset: ±leg_offset_px horizontal separation (default)")
    ap.add_argument("--dump-file", dest="dump_file", default=None,
                    help="replay from .npz produced by dump_shm_stream "
                         "instead of reading live SHM. Enables Mac / laptop "
                         "iteration without the Jetson.")
    ap.add_argument("--no-loop", dest="loop", action="store_false",
                    help="stop when dump ends (only with --dump-file; "
                         "default is looping)")
    ap.set_defaults(loop=True)
    ap.add_argument("--no-sanity", dest="sanity", action="store_false",
                    help="disable anatomical sanity gate. By default, frames "
                         "with impossible bone lengths or L/R asymmetry are "
                         "treated as invalid and the sticky display holds.")
    ap.set_defaults(sanity=True)
    ap.add_argument("--sanity-hip", nargs=2, type=float,
                    metavar=("LO", "HI"), default=(0.15, 0.60),
                    help="acceptable hip-width range in meters (default 0.15–0.60)")
    ap.add_argument("--sanity-thigh", nargs=2, type=float,
                    metavar=("LO", "HI"), default=(0.25, 0.60),
                    help="acceptable thigh length in meters (default 0.25–0.60)")
    ap.add_argument("--sanity-shank", nargs=2, type=float,
                    metavar=("LO", "HI"), default=(0.20, 0.55),
                    help="acceptable shank length in meters (default 0.20–0.55)")
    ap.add_argument("--sanity-asym", type=float, default=1.30,
                    help="max L/R asymmetry ratio (default 1.30 = 30%%)")
    ap.add_argument("--sanity-knee-angle", nargs=2, type=float,
                    metavar=("LO", "HI"), default=(30.0, 185.0),
                    help="knee flexion angle range in degrees "
                         "(default 30–185). 180° = straight leg.")
    ap.add_argument("--sanity-ank-above-knee-tol", type=float, default=0.10,
                    help="max metres ankle is allowed to rise ABOVE knee "
                         "before rejecting (default 0.10 m)")
    ap.add_argument("--sanity-knee-above-hip-tol", type=float, default=0.20,
                    help="max metres knee is allowed to rise ABOVE hip "
                         "before rejecting (default 0.20 m — allows moderate "
                         "high-kick / swing phase)")
    return ap.parse_args()


def _build_human_frame(hip_vector_world: np.ndarray) -> Optional[np.ndarray]:
    """Build R_human_from_world: rotates world coords into human-aligned frame.

    Inputs:
      hip_vector_world: (3,) avg of (right_hip - left_hip) in world frame.
                        This is the human's lateral (X) axis after rotation.

    Output:
      R: (3,3) rotation s.t. ``p_human = R @ p_world``
        - human X = pelvis lateral (right side positive)
        - human Y = gravity down (== world Y, unchanged)
        - human Z = X × Y (human forward — perpendicular to pelvis line and
                            to gravity, in the front-of-body direction)

    Returns None if hip_vector is too small (degenerate, e.g. occlusion at
    calibration time). Caller should keep retrying until a clean frame.

    True sagittal projection is then ``(z_human, -y_human)`` per joint —
    independent of how the camera is pointed; the legs always look like a
    side-view skeleton because we project onto the plane PERPENDICULAR to
    the actual pelvis line, not perpendicular to the camera Z.
    """
    n = float(np.linalg.norm(hip_vector_world))
    if n < 0.05:  # < 5cm → bad (e.g. only one hip detected at calib time)
        return None
    x_human = hip_vector_world / n           # (3,)
    y_human = np.array([0.0, 1.0, 0.0])      # world Y is gravity (down)
    # Re-orthogonalize x against y (project out the Y component) so the
    # rotation is well-defined when the pelvis isn't perfectly horizontal.
    x_human = x_human - np.dot(x_human, y_human) * y_human
    nx = float(np.linalg.norm(x_human))
    if nx < 0.05:
        return None
    x_human /= nx
    z_human = np.cross(x_human, y_human)     # forward (right-handed)
    nz = float(np.linalg.norm(z_human))
    if nz < 0.05:
        return None
    z_human /= nz
    # Rows of R are the new basis expressed in old coords.
    R = np.stack([x_human, y_human, z_human], axis=0)  # (3,3)
    return R


def _resolve_index(schema, name: str) -> Optional[int]:
    try:
        return schema.index(name)
    except (ValueError, KeyError):
        return None


def _knee_angle_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> Optional[float]:
    """Interior angle at the knee (degrees).

    Convention: 180° = fully straight leg (hip–knee–ankle collinear),
    90° = right angle, smaller values = more flexion.  Walking range is
    typically 120°–180°; deep squat can reach ~30°.  Anything outside
    [30°, 185°] is physiologically impossible and indicates a bad keypoint.
    """
    v1 = hip - knee
    v2 = ankle - knee
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-3 or n2 < 1e-3:
        return None
    cos_a = float(np.dot(v1, v2)) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


def _check_anatomy(
    kpts_3d: np.ndarray,
    idx_map: dict,
    hip_bounds: Tuple[float, float],
    thigh_bounds: Tuple[float, float],
    shank_bounds: Tuple[float, float],
    asym_max: float,
    knee_angle_bounds: Tuple[float, float] = (30.0, 185.0),
    ankle_above_knee_tol: float = 0.10,
    knee_above_hip_tol: float = 0.20,
) -> Tuple[bool, List[str]]:
    """Biomechanical sanity gate.

    Three families of checks, all read-only (no pose correction — stays
    within CLAUDE.md's ban on keypoint-feedback loops):

      1. **Bone length**   — each segment must fit adult range.
      2. **L/R asymmetry** — left and right equivalents must match within
         ``asym_max`` ratio.
      3. **Anatomical orientation** — ankles must be below knees and knees
         not dramatically above hips; knee flexion angle must fall in a
         physiologically possible range. Catches hyperextension / depth
         mis-sampling that fakes an impossible pose while bone lengths
         still fit individually.

    Returns (is_valid, list_of_failure_reasons).
    """
    reasons: List[str] = []

    def dist_mm(a_name: str, b_name: str) -> Optional[float]:
        ia = idx_map.get(a_name)
        ib = idx_map.get(b_name)
        if ia is None or ib is None:
            return None
        return float(np.linalg.norm(kpts_3d[ia] - kpts_3d[ib]))

    # ── Bone lengths ──────────────────────────────────────────────────────
    hw = dist_mm("left_hip", "right_hip")
    if hw is not None:
        lo, hi = hip_bounds
        if not (lo <= hw <= hi):
            reasons.append(f"hip_w={hw*100:.0f}cm")

    l_thigh = dist_mm("left_hip", "left_knee")
    r_thigh = dist_mm("right_hip", "right_knee")
    l_shank = dist_mm("left_knee", "left_ankle")
    r_shank = dist_mm("right_knee", "right_ankle")

    t_lo, t_hi = thigh_bounds
    for name, val in (("Lt", l_thigh), ("Rt", r_thigh)):
        if val is not None and not (t_lo <= val <= t_hi):
            reasons.append(f"{name}={val*100:.0f}cm")

    s_lo, s_hi = shank_bounds
    for name, val in (("Ls", l_shank), ("Rs", r_shank)):
        if val is not None and not (s_lo <= val <= s_hi):
            reasons.append(f"{name}={val*100:.0f}cm")

    # ── Bilateral asymmetry ───────────────────────────────────────────────
    if l_thigh is not None and r_thigh is not None and min(l_thigh, r_thigh) > 1e-3:
        asym_t = max(l_thigh, r_thigh) / min(l_thigh, r_thigh)
        if asym_t > asym_max:
            reasons.append(f"TA={asym_t:.2f}")

    if l_shank is not None and r_shank is not None and min(l_shank, r_shank) > 1e-3:
        asym_s = max(l_shank, r_shank) / min(l_shank, r_shank)
        if asym_s > asym_max:
            reasons.append(f"SA={asym_s:.2f}")

    # ── Anatomical orientation (Y+ = down) ────────────────────────────────
    # World Y increases downward, so a lower point has LARGER Y. For a
    # standing / walking person ankle_y > knee_y > hip_y. Depth-sampling
    # errors and keypoint swaps frequently invert this.
    ka_min, ka_max = knee_angle_bounds
    for side in ("left", "right"):
        ih = idx_map.get(f"{side}_hip")
        ik = idx_map.get(f"{side}_knee")
        ia = idx_map.get(f"{side}_ankle")
        if ih is None or ik is None or ia is None:
            continue
        hip_pt = kpts_3d[ih]
        knee_pt = kpts_3d[ik]
        ankle_pt = kpts_3d[ia]

        # Ankle must be below knee (allow ``ankle_above_knee_tol`` slack for
        # high-kick / swing-phase frames).
        if ankle_pt[1] < knee_pt[1] - ankle_above_knee_tol:
            reasons.append(f"{side[0].upper()}_ank^knee")

        # Knee must not be far above hip. Brief high lift allowed.
        if knee_pt[1] < hip_pt[1] - knee_above_hip_tol:
            reasons.append(f"{side[0].upper()}_knee^hip")

        # Knee flexion angle in physiological range.
        ang = _knee_angle_deg(hip_pt, knee_pt, ankle_pt)
        if ang is not None and not (ka_min <= ang <= ka_max):
            reasons.append(f"{side[0].upper()}_kang={ang:.0f}")

    return (len(reasons) == 0, reasons)


def main() -> int:
    import cv2

    args = parse_args()
    schema = get_schema(args.schema)

    # Two input modes: live SHM (Jetson) or replay from npz dump (any OS).
    reader: object
    if args.dump_file:
        from .dump_shm_stream import DumpReader
        try:
            reader = DumpReader(args.dump_file,
                                expected_k=schema.num_keypoints,
                                loop=args.loop)
        except FileNotFoundError:
            print(f"dump file not found: {args.dump_file}", file=sys.stderr)
            return 2
        except RuntimeError as err:
            print(f"dump file error: {err}", file=sys.stderr)
            return 2
        print(f"[view] replay mode — {args.dump_file} "
              f"(K={reader.K}, frames={reader.N}, loop={reader.loop})")
    else:
        try:
            reader = ShmReader(name=args.shm, expected_k=schema.num_keypoints)
        except FileNotFoundError:
            print(f"SHM '/dev/shm/{args.shm}' not found. Is run_stream_demo "
                  f"running? (Or pass --dump-file for offline replay.)",
                  file=sys.stderr)
            return 2

    # Resolve skeleton indices once
    skel_idx: List[Tuple[int, int]] = []
    for a, b in SKELETON:
        ia = _resolve_index(schema, a)
        ib = _resolve_index(schema, b)
        if ia is not None and ib is not None:
            skel_idx.append((ia, ib))

    # left/right per-keypoint (for color + offset)
    is_left = np.array(["left" in n for n in schema.keypoints], dtype=bool)
    is_right = np.array(["right" in n for n in schema.keypoints], dtype=bool)

    W, H = args.width, args.height
    margin = 50
    sleep_s = 1.0 / max(args.fps, 1.0)

    # Sticky state — keep last valid frame so transient invalid runs don't blank.
    last_kpts_3d: Optional[np.ndarray] = None
    last_kpt_conf: Optional[np.ndarray] = None
    last_box_conf = 0.0
    last_dir = 0.0
    last_frame_id = -1
    sticky_age_frames = 0

    # Resolve hip indices for calibration (need both for hip vector).
    lh_idx = _resolve_index(schema, "left_hip")
    rh_idx = _resolve_index(schema, "right_hip")

    # Index map for sanity gate (name → array index).
    kp_idx_map = {n: _resolve_index(schema, n) for n in schema.keypoints}

    # Counters for the sanity gate so the header can summarise rejections.
    sanity_rejects = 0
    sanity_accepted = 0
    last_sanity_reasons: List[str] = []

    # Standing-calibration buffer: collect first N valid hip vectors
    # (in world frame), average → human X axis. Then build R that
    # rotates world coords so the human's lateral X is aligned with
    # screen X — equivalently, world Z gets rotated to be the
    # walking-forward direction. The user sees a true side view of
    # the gait regardless of camera yaw.
    R_view: Optional[np.ndarray] = None
    calib_buf: List[np.ndarray] = []
    calib_done = False
    calib_failed = False

    # FPS counter
    fps_t0 = time.monotonic()
    fps_n = 0
    fps_show = 0.0

    try:
        while True:
            tick_start = time.monotonic()
            img = np.full((H, W, 3), 30, dtype=np.uint8)

            # ── Read latest SHM frame ──────────────────────────────────────
            data = reader.read()
            cur_valid = False
            cur_frame_id = -1
            if data is not None:
                cur_frame_id, _ts, kpts_3d, kpt_conf, _kpts_2d, box_conf, valid, dir_, world_frame_applied = data
                # Geometric sanity gate — catches YOLO failure modes (self-occlusion,
                # L/R swap, treadmill-belt confusion) that produce impossible poses
                # while still reporting valid=True. Gate is read-only; rejected
                # frames are held with the existing sticky display.
                if valid and args.sanity:
                    raw_3d = np.asarray(kpts_3d)
                    sane, reasons = _check_anatomy(
                        raw_3d, kp_idx_map,
                        tuple(args.sanity_hip),
                        tuple(args.sanity_thigh),
                        tuple(args.sanity_shank),
                        args.sanity_asym,
                        knee_angle_bounds=tuple(args.sanity_knee_angle),
                        ankle_above_knee_tol=args.sanity_ank_above_knee_tol,
                        knee_above_hip_tol=args.sanity_knee_above_hip_tol,
                    )
                    if not sane:
                        sanity_rejects += 1
                        last_sanity_reasons = reasons
                        valid = False
                    else:
                        sanity_accepted += 1
                        last_sanity_reasons = []

                if valid:
                    last_kpts_3d = np.asarray(kpts_3d).copy()
                    last_kpt_conf = np.asarray(kpt_conf).copy()
                    last_box_conf = float(box_conf)
                    last_dir = float(dir_)
                    last_frame_id = cur_frame_id
                    sticky_age_frames = 0
                    cur_valid = True

                    # ── Standing calibration (first N valid frames) ──
                    # Collect right_hip - left_hip in WORLD frame, average,
                    # use that as the "lateral" axis. Then world Z gets
                    # rotated to be the human walking-forward direction.
                    if (not args.no_calib and not calib_done
                            and lh_idx is not None and rh_idx is not None):
                        if (last_kpt_conf[lh_idx] >= 0.5
                                and last_kpt_conf[rh_idx] >= 0.5):
                            hip_vec = (last_kpts_3d[rh_idx]
                                       - last_kpts_3d[lh_idx])
                            calib_buf.append(hip_vec)
                            if len(calib_buf) >= args.calib_frames:
                                avg = np.mean(np.stack(calib_buf, axis=0), axis=0)
                                R_view = _build_human_frame(avg)
                                if R_view is None:
                                    calib_failed = True
                                    calib_done = True  # stop trying
                                else:
                                    calib_done = True
                else:
                    sticky_age_frames += 1

            # ── Apply walking-direction rotation (after calibration) ──────
            # Without this, world Z is just camera-horizontal-forward and the
            # legs look 'foreshortened' if the user isn't perfectly facing the
            # camera. With R_view, world Z rotates to align with the user's
            # actual walking direction → legs always look full-length in side view.
            display_kpts: Optional[np.ndarray] = None
            if last_kpts_3d is not None:
                if R_view is not None:
                    display_kpts = (R_view @ last_kpts_3d.T).T
                else:
                    display_kpts = last_kpts_3d

            # ── Layout: hip-center as fixed origin, all 6 keypoints ─────
            if display_kpts is not None:
                # C2: hip midpoint is the projection origin (in display frame).
                # Both L-hip and R-hip project to this point in pure sagittal.
                if lh_idx is not None and rh_idx is not None:
                    hip_ctr = 0.5 * (display_kpts[lh_idx] + display_kpts[rh_idx])
                else:
                    hip_ctr = display_kpts.mean(axis=0)

                # C4: Sx = −Z  (forward → left on screen)
                #     Sy = +Y  (world Y+ = down → screen bottom)
                # No sign flip needed for Y — direct mapping to gravity.
                z_all = display_kpts[:, 2]   # forward (m), world Z
                y_all = display_kpts[:, 1]   # down    (m), world Y

                # Auto-scale: distances from hip_ctr set the viewport.
                rel_z = z_all - hip_ctr[2]
                rel_y = y_all - hip_ctr[1]
                z_half  = max(float(np.abs(rel_z).max()), 0.15)   # symmetric ±Z
                y_below = max(float(rel_y.max()), 0.15)            # legs below hip
                y_above = max(float(-rel_y.min()), 0.05)           # head-room above

                s_z = (W / 2 - margin) / z_half
                s_y = (H - 3 * margin) / max(y_below + y_above, 0.30)
                scale = min(s_z, s_y) * args.zoom

                # Hip midpoint is anchored at upper 25 % of canvas.
                anchor_x = W // 2
                anchor_y = margin + int(H * 0.12)

                def to_screen(z: float, y: float, side_offset: int = 0) -> Tuple[int, int]:
                    sx = int(anchor_x - (z - hip_ctr[2]) * scale + side_offset)
                    sy = int(anchor_y + (y - hip_ctr[1]) * scale)
                    return sx, sy

                # Grid: vertical lines every 0.25 m along Z, centred on hip_ctr.
                tick = 0.25
                z_half_world = (W / 2 - margin) / scale
                z_low  = hip_ctr[2] - z_half_world
                z_high = hip_ctr[2] + z_half_world
                z_t = np.ceil(z_low / tick) * tick
                while z_t <= z_high:
                    sx, _ = to_screen(z_t, hip_ctr[1])
                    if margin < sx < W - margin:
                        cv2.line(img, (sx, margin), (sx, H - margin), (55, 55, 55), 1)
                        if abs(z_t - round(z_t)) < 1e-3:
                            cv2.putText(img, f"{z_t:.0f}m", (sx - 12, H - margin + 18),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (140, 140, 140), 1, cv2.LINE_AA)
                    z_t += tick

                # Gravity arrow (top-right)
                ax, ay0, ay1 = W - margin - 10, margin + 16, margin + 60
                cv2.arrowedLine(img, (ax, ay0), (ax, ay1), (200, 200, 200), 2, tipLength=0.4)
                cv2.putText(img, "g", (ax - 14, ay1 - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # ── Compute per-side screen points ────────────────────────
                # C5: pure mode → no offset (true sagittal, colour-only);
                #     offset mode → ±leg_offset_px visual separation.
                offset = args.leg_offset_px if args.display_mode == "offset" else 0
                pts: List[Optional[Tuple[int, int]]] = [None] * len(z_all)
                for i in range(len(z_all)):
                    side_off = -offset if is_left[i] else (+offset if is_right[i] else 0)
                    pts[i] = to_screen(float(z_all[i]), float(y_all[i]), side_off)

                # ── Draw bones (under joints) ─────────────────────────────
                # Left = blue, right = red. Pelvis = grey.
                left_bone = (255, 130, 80)   # cv2 BGR — blue-ish for "left"
                right_bone = (80, 80, 255)   # red for "right"
                pelvis_bone = (160, 160, 160)
                for (a, b) in skel_idx:
                    pa, pb = pts[a], pts[b]
                    if pa is None or pb is None:
                        continue
                    if is_left[a] and is_left[b]:
                        col = left_bone
                    elif is_right[a] and is_right[b]:
                        col = right_bone
                    else:
                        col = pelvis_bone
                    cv2.line(img, pa, pb, col, 3, cv2.LINE_AA)

                # ── Draw joints (always shown; dim if low conf) ───────────
                for i, p in enumerate(pts):
                    if p is None:
                        continue
                    conf = float(last_kpt_conf[i]) if last_kpt_conf is not None else 1.0
                    dim = conf < args.low_conf
                    if is_left[i]:
                        fill = (255, 130, 80) if not dim else (120, 90, 70)
                    elif is_right[i]:
                        fill = (80, 80, 255) if not dim else (70, 70, 120)
                    else:
                        fill = (200, 200, 200)
                    cv2.circle(img, p, 7, fill, -1, cv2.LINE_AA)
                    cv2.circle(img, p, 7, (255, 255, 255), 1, cv2.LINE_AA)

                # ── Bone length panel (lower-left) — uses RAW world coords
                #    (not display_kpts) so cm reads the actual physical
                #    length, independent of the view rotation. ─────────────
                panel_y = H - margin + 38
                kp_name = {n: i for i, n in enumerate(schema.keypoints)}
                for side, scol in (("left", (255, 130, 80)), ("right", (80, 80, 255))):
                    for a, b in (("hip", "knee"), ("knee", "ankle")):
                        an, bn = f"{side}_{a}", f"{side}_{b}"
                        if an in kp_name and bn in kp_name:
                            ia, ib = kp_name[an], kp_name[bn]
                            length_cm = float(np.linalg.norm(
                                last_kpts_3d[ia] - last_kpts_3d[ib])) * 100
                            cv2.putText(
                                img,
                                f"{side[0].upper()}-{a}{b}: {length_cm:5.1f}cm",
                                (10, panel_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, scol, 1, cv2.LINE_AA,
                            )
                            panel_y += 16
            else:
                cv2.putText(img, "Sagittal — waiting for first valid frame",
                            (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (180, 180, 180), 1, cv2.LINE_AA)

            # ── Header ──────────────────────────────────────────────────
            fps_n += 1
            now = time.monotonic()
            if now - fps_t0 > 1.0:
                fps_show = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            valid_tag = "VALID" if cur_valid else f"sticky+{sticky_age_frames}"
            wf_tag = "WF:IMU" if (data is not None and world_frame_applied) else "WF:CAM"
            if args.no_calib:
                calib_tag = "calib OFF"
            elif calib_failed:
                calib_tag = "calib FAILED — raw view"
            elif calib_done:
                calib_tag = "calibrated"
            else:
                calib_tag = f"calibrating {len(calib_buf)}/{args.calib_frames}"
            if args.sanity:
                total_checked = sanity_rejects + sanity_accepted
                reject_pct = (sanity_rejects / total_checked * 100.0
                              if total_checked else 0.0)
                sanity_tag = f"sanity:{reject_pct:.0f}%rej"
            else:
                sanity_tag = "sanity:off"
            header = (f"frame {last_frame_id}  conf {last_box_conf:.2f}  "
                      f"depth_inv {last_dir:.0%}  viewer {fps_show:.0f}Hz  "
                      f"{valid_tag}  [{calib_tag}]  mode:{args.display_mode}  "
                      f"{sanity_tag}  {wf_tag}")
            cv2.putText(img, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # When the current frame was rejected, show the reasons in red
            # just below the header — makes it obvious which check failed.
            if (not cur_valid) and last_sanity_reasons:
                msg = "REJECT: " + " ".join(last_sanity_reasons[:4])
                cv2.putText(img, msg, (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (80, 80, 255), 1, cv2.LINE_AA)

            # Axis labels: Z increases going LEFT, Y increases going DOWN.
            cv2.putText(img, "<- Z (forward)", (10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
            cv2.putText(img, "+Y (down)", (6, anchor_y - 6 if display_kpts is not None else 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)

            cv2.imshow("sagittal", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

            # Rate cap (don't burn CPU)
            elapsed = time.monotonic() - tick_start
            if elapsed < sleep_s:
                time.sleep(sleep_s - elapsed)
    finally:
        reader.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
