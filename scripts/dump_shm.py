"""Dump /hwalker_pose_v2 + /hwalker_forecast SHM packets — Jetson verification tool.

Reads both SHMs once (or continuously with --watch), prints header fields, and
sanity-checks: version, seq parity (even=stable), K, packet size, NaN counts.

Usage:
    python3 scripts/dump_shm.py                  # one shot, both SHMs
    python3 scripts/dump_shm.py --watch 5        # 5s continuous @ 10Hz
    python3 scripts/dump_shm.py --pose-only
    python3 scripts/dump_shm.py --forecast-only
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shm_to_teensy_bridge import (  # noqa: E402
    ForecastReader, FORECAST_NAME, FORECAST_VERSION, N_JOINTS,
)


# ─── /hwalker_pose_v2 layout (mirrors CUDA_Stream/shm_publisher.py) ──────
POSE_NAMES = ["hwalker_pose_v2", "hwalker_pose_cuda"]    # try both
POSE_HEADER_SIZE = 64
POSE_VERSION = 2

POSE_SEQ_OFF              = 0
POSE_VERSION_OFF          = 4
POSE_K_OFF                = 8
POSE_FRAME_ID_OFF         = 12
POSE_RGB_TS_OFF           = 16
POSE_DEPTH_TS_OFF         = 24
POSE_DEPTH_AGE_OFF        = 32
POSE_BOX_CONF_OFF         = 36
POSE_DEPTH_INVALID_OFF    = 40
POSE_VALID_FLAG_OFF       = 44
POSE_WORLD_FRAME_OFF      = 45
POSE_VALID_REASON_OFF     = 46
POSE_PUBLISH_DONE_OFF     = 48
POSE_VALID_MASK_BITS_OFF  = 56


def _open(name):
    try:
        return shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return None


def dump_pose_v2():
    print("\n── /hwalker_pose_v2 ────────────────────────────────────────────")
    shm = None
    for n in POSE_NAMES:
        shm = _open(n)
        if shm is not None:
            print(f"  found: /{n} (size={shm.size}B)")
            break
    if shm is None:
        print(f"  ✗ NOT FOUND (tried: {POSE_NAMES})")
        print("    Hint: run `python3 src/perception/realtime/pipeline_main.py --enable-shm-v2` first.")
        return False

    buf = shm.buf
    try:
        seq = struct.unpack_from("<I", buf, POSE_SEQ_OFF)[0]
        ver = struct.unpack_from("<I", buf, POSE_VERSION_OFF)[0]
        K = struct.unpack_from("<I", buf, POSE_K_OFF)[0]
        frame_id = struct.unpack_from("<I", buf, POSE_FRAME_ID_OFF)[0]
        rgb_ts = struct.unpack_from("<Q", buf, POSE_RGB_TS_OFF)[0]
        depth_age_us = struct.unpack_from("<I", buf, POSE_DEPTH_AGE_OFF)[0]
        box_conf = struct.unpack_from("<f", buf, POSE_BOX_CONF_OFF)[0]
        depth_inv = struct.unpack_from("<f", buf, POSE_DEPTH_INVALID_OFF)[0]
        valid_flag = buf[POSE_VALID_FLAG_OFF]
        world_frame = buf[POSE_WORLD_FRAME_OFF]
        valid_reason = buf[POSE_VALID_REASON_OFF]
        pub_done_ns = struct.unpack_from("<Q", buf, POSE_PUBLISH_DONE_OFF)[0]
        valid_mask = struct.unpack_from("<Q", buf, POSE_VALID_MASK_BITS_OFF)[0]

        print(f"  seq={seq}  parity={'EVEN(stable)' if seq%2==0 else 'ODD(writing!)'}")
        print(f"  version={ver}   (expect {POSE_VERSION})")
        print(f"  K={K}")
        print(f"  frame_id={frame_id}")
        print(f"  rgb_ts_ns={rgb_ts}  ({(time.time_ns()-rgb_ts)/1e6:+.1f}ms vs wall — only valid if same clock)")
        print(f"  depth_age_us={depth_age_us}")
        print(f"  box_conf={box_conf:.3f}")
        print(f"  depth_invalid_ratio={depth_inv:.3f}")
        print(f"  valid_flag={valid_flag}  world_frame={world_frame}  valid_reason={valid_reason}")
        print(f"  publish_done_mono_ns={pub_done_ns}")
        print(f"  valid_mask_bits=0x{valid_mask:016x}  ({bin(valid_mask)[2:].zfill(8)})")

        # Decode keypoints
        if 1 <= K <= 16:
            print(f"  keypoints (K={K}):")
            kpts_3d_off = POSE_HEADER_SIZE
            kpts_2d_off = kpts_3d_off + K*12
            conf_off = kpts_2d_off + K*8
            sigma_off = conf_off + K*4
            cov_off = sigma_off + K*12
            joint_names = ['L_hip', 'L_knee', 'L_ankle', 'R_hip', 'R_knee', 'R_ankle']
            for i in range(min(K, 6)):
                x, y, z = struct.unpack_from("<3f", buf, kpts_3d_off + i*12)
                u, v = struct.unpack_from("<2f", buf, kpts_2d_off + i*8)
                c = struct.unpack_from("<f", buf, conf_off + i*4)[0]
                sx, sy, sz = struct.unpack_from("<3f", buf, sigma_off + i*12)
                valid = (valid_mask >> i) & 1
                print(f"    [{i}] {joint_names[i]:8s}  3D=({x:+6.3f},{y:+6.3f},{z:+6.3f})m"
                      f"  2D=({u:6.1f},{v:6.1f})px  conf={c:.2f}  σ=({sx:.3f},{sy:.3f},{sz:.3f})  valid={valid}"
                      .replace('nan', ' NaN'))
        else:
            print(f"  ⚠️ K={K} out of expected range — skipping keypoint decode")

        ok = (ver == POSE_VERSION) and (seq % 2 == 0) and (K == 6)
        print(f"  → {'PASS' if ok else 'FAIL'}: layout sanity")
        return ok
    finally:
        try:
            buf.release()
        except Exception:
            pass
        shm.close()


def dump_forecast():
    print("\n── /hwalker_forecast ───────────────────────────────────────────")
    try:
        reader = ForecastReader()
    except FileNotFoundError:
        print("  ✗ NOT FOUND")
        print("    Hint: run pipeline with --enable-plan-d first.")
        return False
    except Exception as e:
        print(f"  ✗ open error: {e}")
        return False

    try:
        fc = reader.read()
        if fc is None:
            print("  ✗ seqlock read failed (writer always odd?) — pipeline may be paused")
            return False
        joint_names = ['L_hip', 'L_knee', 'L_ankle', 'R_hip', 'R_knee', 'R_ankle']
        print(f"  seq={fc.seq}  parity={'EVEN(stable)' if fc.seq%2==0 else 'ODD'}")
        print(f"  version={fc.version}   (expect {FORECAST_VERSION})")
        print(f"  frame_id={fc.frame_id}")
        print(f"  publish_done_mono_ns={fc.publish_done_mono_ns}")
        print(f"  tau_lookahead_s={fc.tau_lookahead_s:.4f}")
        print(f"  phi={fc.phi:+.3f}rad  phi_sigma={fc.phi_sigma:.3f}")
        print(f"  omega={fc.omega:+.3f}rad/s  ({fc.omega/(2*3.14159):+.2f}Hz)")
        print(f"  cascade_level={fc.cascade_level}   is_ready={fc.is_ready}")
        print(f"  stride_count={fc.stride_count}  template_frac={fc.template_frac:.2f}")
        print(f"  q_pred (rad) by joint:")
        for i, (q, s) in enumerate(zip(fc.q_pred, fc.q_pred_sigma)):
            print(f"    [{i}] {joint_names[i]:8s}  q={q:+.3f}  σ={s:+.3f}")
        print(f"  HS_L t={fc.t_hs_L_s:+.3f}s  HS_R t={fc.t_hs_R_s:+.3f}s")
        ok = (fc.version == FORECAST_VERSION) and (fc.seq % 2 == 0)
        valid = fc.valid_for_control()
        print(f"  → layout {'PASS' if ok else 'FAIL'}, valid_for_control={'YES' if valid else 'NO'}")
        return ok
    finally:
        reader.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0.0, help="seconds; 0 = one-shot")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--pose-only", action="store_true")
    ap.add_argument("--forecast-only", action="store_true")
    args = ap.parse_args()

    def one_pass():
        ok_pose = True
        ok_fc = True
        if not args.forecast_only:
            ok_pose = dump_pose_v2()
        if not args.pose_only:
            ok_fc = dump_forecast()
        return ok_pose and ok_fc

    if args.watch <= 0:
        sys.exit(0 if one_pass() else 1)

    period = 1.0 / max(args.rate, 0.1)
    t0 = time.monotonic()
    iters = 0
    pass_count = 0
    while (time.monotonic() - t0) < args.watch:
        iters += 1
        if one_pass():
            pass_count += 1
        time.sleep(period)
    print(f"\n=== Watch summary: {pass_count}/{iters} passes over {args.watch:.1f}s ===")
    sys.exit(0 if pass_count == iters else 1)


if __name__ == "__main__":
    main()
