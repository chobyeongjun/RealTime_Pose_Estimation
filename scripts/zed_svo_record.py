"""Headless SVO recorder — ZED Python SDK, no GUI needed.

ZED_Explorer requires X display, which Jetson over SSH doesn't have.
This script opens the camera, enables SVO recording, runs for N seconds,
then closes cleanly. Designed for walking_session.sh.

Usage:
    python3 scripts/zed_svo_record.py walking.svo2 --duration 60
    python3 scripts/zed_svo_record.py walking.svo2 --duration 60 --resolution SVGA --fps 120
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import pyzed.sl as sl
except ImportError:
    print("pyzed.sl not found — ZED SDK Python wrapper missing.", file=sys.stderr)
    print("Install: cd /usr/local/zed && python3 get_python_api.py", file=sys.stderr)
    sys.exit(2)

RES_MAP = {
    "VGA":   sl.RESOLUTION.VGA,
    "SVGA":  sl.RESOLUTION.SVGA,
    "HD720": sl.RESOLUTION.HD720,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD2K":  sl.RESOLUTION.HD2K,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Output .svo2 file path")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--resolution", choices=list(RES_MAP.keys()), default="SVGA")
    ap.add_argument("--fps", type=int, default=120)
    args = ap.parse_args()

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RES_MAP[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"open failed: {err}", file=sys.stderr)
        sys.exit(1)

    rec_params = sl.RecordingParameters(args.path, sl.SVO_COMPRESSION_MODE.H264)
    err = cam.enable_recording(rec_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"enable_recording failed: {err}", file=sys.stderr)
        cam.close()
        sys.exit(1)

    print(f"Recording → {args.path} ({args.resolution} @ {args.fps}fps, {args.duration}s)")
    print("Press Ctrl+C to stop early.")

    rt = sl.RuntimeParameters()
    t0 = time.monotonic()
    grabs = 0
    last_print = 0.0
    try:
        while (time.monotonic() - t0) < args.duration:
            if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                grabs += 1
            now = time.monotonic() - t0
            if (now - last_print) >= 1.0:
                last_print = now
                fps = grabs / max(now, 1e-3)
                print(f"  [{now:5.1f}s / {args.duration:.0f}s]  frames={grabs}  avg_fps={fps:.1f}",
                      flush=True)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        cam.disable_recording()
        cam.close()

    dur = time.monotonic() - t0
    print(f"\nDone. frames={grabs}  duration={dur:.1f}s  avg_fps={grabs/max(dur,1e-3):.1f}")
    print(f"SVO: {args.path}")


if __name__ == "__main__":
    main()
