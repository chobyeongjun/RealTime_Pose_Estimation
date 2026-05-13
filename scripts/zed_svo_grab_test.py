"""Minimal SVO grab test — ZED SDK only, NO pipeline_main, NO TRT, NO Plan D.

If this fails, the SVO file or ZED SDK is broken.
If this passes, the bug is in pipeline_main's SVO handling.
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import pyzed.sl as sl
except ImportError:
    print("pyzed.sl missing", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svo")
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--depth-mode", default="PERFORMANCE",
                    choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"])
    args = ap.parse_args()

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False                      # full speed
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER

    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"OPEN FAIL: {err}", file=sys.stderr)
        sys.exit(1)

    info = cam.get_camera_information()
    cfg = info.camera_configuration
    total = cam.get_svo_number_of_frames()
    print(f"Opened {args.svo}")
    print(f"  resolution: {cfg.resolution.width}x{cfg.resolution.height}")
    print(f"  fps:        {cfg.fps}")
    print(f"  total:      {total} frames")
    print(f"  depth_mode: {args.depth_mode}")
    print()

    rt = sl.RuntimeParameters()
    img = sl.Mat()
    dep = sl.Mat()

    t0 = time.monotonic()
    ok = 0
    eof = 0
    err_other = 0
    last_print = 0.0
    last_err = None

    target = min(args.max_frames, total)
    for i in range(target):
        e = cam.grab(rt)
        if e == sl.ERROR_CODE.SUCCESS:
            ok += 1
            cam.retrieve_image(img, sl.VIEW.LEFT)
            if args.depth_mode != "NONE":
                cam.retrieve_measure(dep, sl.MEASURE.DEPTH)
        elif e == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            eof += 1
            break
        else:
            err_other += 1
            last_err = e
        now = time.monotonic() - t0
        if (now - last_print) >= 0.5:
            last_print = now
            rate = (i+1) / max(now, 1e-3)
            print(f"  [{i+1:4d}/{target}]  rate={rate:6.1f} grabs/s  ok={ok}  eof={eof}  err={err_other}",
                  flush=True)

    dur = time.monotonic() - t0
    print()
    print(f"=== {ok}/{target} grabs OK in {dur:.2f}s = {ok/max(dur,1e-3):.1f} grabs/s ===")
    print(f"  EOF reached: {eof}")
    print(f"  other errs:  {err_other}  last={last_err}")

    cam.close()
    sys.exit(0 if ok > 0 else 3)


if __name__ == "__main__":
    main()
