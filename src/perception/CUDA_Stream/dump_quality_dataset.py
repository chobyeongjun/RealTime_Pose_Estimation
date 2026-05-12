"""Quality dataset dumper — recorded session 의 raw RGB + depth + pose + calib + timestamps + valid_mask.

Codex orchestration `bvfvkxo1m` Q2(a) spec. Plan D EKF + V4L2 baseline + Mocap RMSE prerequisite.

사용 (Jetson):
    sudo PYTHONPATH=/home/chobb0/.local/lib/python3.10/site-packages \\
        python3 -m perception.CUDA_Stream.dump_quality_dataset \\
        --output dumps/session_001 \\
        --duration 60 --every 5

출력:
    dumps/session_001/
        session_calib.json          (1회, ZED self-calib disabled + snapshot)
        frame_000000.npz
        frame_000001.npz
        ... (~720 files for 60s × 12fps)

⚠️ Production hot path 영향 X — *별도 entry*. production pipeline 동시 실행 X.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .quality_dataset_io import QualityFrame, save_frame_npz
from .zed_calib_load import disable_self_calib_and_snapshot

LOGGER = logging.getLogger(__name__)


# Disk space safety threshold — 100 MB minimum free before each dump.
MIN_FREE_BYTES = 100 * 1024 * 1024


def _check_disk_space(output_dir: Path, required: int = MIN_FREE_BYTES) -> bool:
    """Disk free space check. Returns False if too low."""
    try:
        stat = shutil.disk_usage(output_dir)
        return stat.free > required
    except OSError:
        return True   # best-effort: skip if disk_usage fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path,
                    help="출력 directory (frame_NNNNNN.npz + session_calib.json)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="dump 시간 초 (default 60)")
    ap.add_argument("--every", type=int, default=5,
                    help="N frame 마다 dump (default 5 = 12fps at 60Hz)")
    ap.add_argument("--include-right", action="store_true",
                    help="right RGB 도 archive (V4L2 sparse stereo baseline)")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="JPEG quality (default 90)")
    ap.add_argument("--calib-only", action="store_true",
                    help="session_calib.json 만 dump + exit")
    ap.add_argument("--force", action="store_true",
                    help="output dir 이미 존재 시 overwrite")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        stream=sys.stdout)

    output_dir: Path = args.output.resolve()
    if output_dir.exists() and not args.force:
        if any(output_dir.iterdir()):
            LOGGER.error(
                "output dir %s not empty — use --force 또는 다른 경로", output_dir
            )
            return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _check_disk_space(output_dir):
        LOGGER.error("disk space < %d MB — abort", MIN_FREE_BYTES // 1024 // 1024)
        return 1

    # ─── ZED open with self-calib disabled ────────────────────────────────
    try:
        import pyzed.sl as sl
    except ImportError:
        LOGGER.error("pyzed not available — Jetson 에서만 실행 가능")
        return 1

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 120
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True   # ★ Codex Q8

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        LOGGER.error("ZED open failed: %s", status)
        return 1

    try:
        # session_calib.json
        calib_path = output_dir / "session_calib.json"
        calib = disable_self_calib_and_snapshot(zed, calib_path)
        LOGGER.info(
            "session: zed_serial=%d baseline=%.2fmm fx=%.2f cx=%.2f",
            calib["zed_serial"], calib["baseline_mm"],
            calib["left_cam"]["fx"], calib["left_cam"]["cx"]
        )

        if args.calib_only:
            LOGGER.info("--calib-only — exiting after calib snapshot")
            return 0

        # ─── ZED main loop (production pipeline 외, minimal capture) ──────
        rt = sl.RuntimeParameters()
        image_mat = sl.Mat()
        image_right_mat = sl.Mat() if args.include_right else None
        depth_mat = sl.Mat()

        stop_flag = {"stop": False}

        def _on_signal(signum, _frame):
            LOGGER.info("signal %d — graceful stop", signum)
            stop_flag["stop"] = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        LOGGER.info(
            "dumping → %s for %.1fs (every %d frames, JPEG q=%d, include_right=%s)",
            output_dir, args.duration, args.every,
            args.jpeg_quality, args.include_right,
        )
        t_start = time.time()
        grab_count = 0
        saved_count = 0
        total_bytes = 0

        while not stop_flag["stop"] and time.time() - t_start < args.duration:
            if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.001)
                continue
            grab_count += 1

            # Sample every N frames
            if grab_count % args.every != 0:
                continue

            # Disk space sanity (every 100 saves)
            if saved_count % 100 == 0 and not _check_disk_space(output_dir):
                LOGGER.error("disk low — graceful stop at frame %d", saved_count)
                break

            # ZED capture
            rgb_ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            rgb_bgra = image_mat.get_data(deep_copy=True)   # (H, W, 4) uint8

            rgb_right_bgra = None
            if args.include_right:
                zed.retrieve_image(image_right_mat, sl.VIEW.RIGHT)
                rgb_right_bgra = image_right_mat.get_data(deep_copy=True)

            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_m = depth_mat.get_data(deep_copy=True).astype(np.float32, copy=False)
            # ZED depth 의 valid 픽셀 비율 (NaN/inf 제외)
            depth_invalid_ratio = float(
                np.mean(~np.isfinite(depth_m) | (depth_m <= 0.0))
            )

            # Pose: dump_quality_dataset 는 *raw frame 만* archive.
            # Plan D EKF 가 *별도 분석* 으로 pose 계산. 단 placeholder fields.
            K = 6
            kpts_2d = np.zeros((K, 2), dtype=np.float32)
            kpts_3d = np.zeros((K, 3), dtype=np.float32)
            kp_conf = np.zeros((K,), dtype=np.float32)
            kp_sigma_m = np.full((K, 3), 0.015, dtype=np.float32)
            pose_cov_diag = (kp_sigma_m ** 2).astype(np.float32)

            frame = QualityFrame(
                frame_id=saved_count,
                rgb_ts_ns=rgb_ts_ns,
                depth_ts_ns=rgb_ts_ns,            # same-frame
                depth_age_us=0,
                publish_done_mono_ns=time.monotonic_ns(),
                valid_mask_bits=0,                 # pose 미계산 → all invalid
                valid_reason=0,
                world_frame_applied=False,
                box_conf=0.0,
                depth_invalid_ratio=depth_invalid_ratio,
                kpts_2d_px=kpts_2d,
                kpts_3d_m=kpts_3d,
                kp_conf=kp_conf,
                kp_sigma_m=kp_sigma_m,
                pose_cov_diag=pose_cov_diag,
                rgb_bgra=rgb_bgra,
                depth_m=depth_m,
                rgb_right_bgra=rgb_right_bgra,
            )

            out_path = output_dir / f"frame_{saved_count:06d}.npz"
            bytes_written = save_frame_npz(
                out_path, frame,
                jpeg_quality=args.jpeg_quality,
                compress=True,
            )
            total_bytes += bytes_written
            saved_count += 1

            if saved_count % 50 == 0:
                elapsed = time.time() - t_start
                LOGGER.info(
                    "saved %d / target %d (elapsed %.1fs, %.1f MB)",
                    saved_count,
                    int(args.duration * (60 / args.every)),
                    elapsed,
                    total_bytes / 1024 / 1024,
                )

        elapsed = time.time() - t_start
        LOGGER.info(
            "DONE: saved=%d grab=%d elapsed=%.1fs total=%.1f MB",
            saved_count, grab_count, elapsed, total_bytes / 1024 / 1024,
        )
        return 0
    finally:
        zed.close()


if __name__ == "__main__":
    raise SystemExit(main())
