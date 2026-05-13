"""SVO → mp4 변환 (Mac 에서 볼 수 있게).

Usage:
    python3 scripts/svo_to_mp4.py walking.svo2                  # → walking.mp4 (LEFT view, 30fps)
    python3 scripts/svo_to_mp4.py walking.svo2 -o out.mp4 --fps 60
    python3 scripts/svo_to_mp4.py walking.svo2 --view DEPTH     # depth map
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import pyzed.sl as sl
except ImportError:
    print("pyzed.sl missing", file=sys.stderr); sys.exit(2)
try:
    import cv2
except ImportError:
    print("opencv-python missing — pip install opencv-python", file=sys.stderr); sys.exit(2)

VIEW_MAP = {
    "LEFT":  sl.VIEW.LEFT,
    "RIGHT": sl.VIEW.RIGHT,
    "DEPTH": sl.VIEW.DEPTH,
    "CONFIDENCE": sl.VIEW.CONFIDENCE,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svo")
    ap.add_argument("-o", "--out", default=None,
                    help="output .mp4 path; default = same dir, same name, .mp4")
    ap.add_argument("--view", default="LEFT", choices=list(VIEW_MAP.keys()))
    ap.add_argument("--fps", type=int, default=30,
                    help="output fps (subsample SVO 120fps to 30fps by default)")
    ap.add_argument("--max-frames", type=int, default=-1)
    args = ap.parse_args()

    svo = Path(args.svo)
    if not svo.is_file():
        print(f"SVO not found: {svo}", file=sys.stderr); sys.exit(1)
    out = Path(args.out) if args.out else svo.with_suffix(".mp4")

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"open failed: {err}", file=sys.stderr); sys.exit(3)

    info = cam.get_camera_information()
    cfg = info.camera_configuration
    src_fps = cfg.fps
    w = cfg.resolution.width
    h = cfg.resolution.height
    total = cam.get_svo_number_of_frames()
    print(f"SVO: {w}x{h} @ {src_fps}fps × {total} frames")

    stride = max(1, int(round(src_fps / args.fps)))
    print(f"Subsampling stride: every {stride}-th frame → ~{src_fps/stride:.1f}fps output")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(args.fps), (w, h))

    img = sl.Mat()
    rt = sl.RuntimeParameters()
    view = VIEW_MAP[args.view]

    t0 = time.monotonic()
    grabs = 0
    written = 0
    target = total if args.max_frames < 0 else min(args.max_frames, total)
    for i in range(target):
        e = cam.grab(rt)
        if e == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if e != sl.ERROR_CODE.SUCCESS:
            continue
        grabs += 1
        if i % stride != 0:
            continue
        cam.retrieve_image(img, view)
        bgra = img.get_data()
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        writer.write(bgr)
        written += 1
        if written % 30 == 0:
            now = time.monotonic() - t0
            print(f"  [{i+1:5d}/{target}]  wrote={written}  in {now:.1f}s", flush=True)

    writer.release()
    cam.close()

    sz = out.stat().st_size / 1024 / 1024
    print(f"\nDone. {written} frames → {out} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
