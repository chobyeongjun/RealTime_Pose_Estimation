"""Batch pose computation — offline YOLO TRT inference on dumped quality dataset.

Codex orchestration Q3 + 사용자 의지 (모두 implement).

Input:  dumps/session_*/frame_NNNNNN.npz (rgb_bgra_jpeg + depth_m + placeholder pose)
Output: ★ Codex Q3 권유 — new posed dir (raw 보존):
        dumps/session_*_pose/frame_NNNNNN.npz (pose attached)

Workflow:
    1. Load frame.npz (rgb_bgra_jpeg → decode → BGRA)
    2. Preprocess (BGRA → BGR → RGB normalize → letterbox to 640×640)
    3. YOLO TRT inference (Jetson only — engine load)
    4. Post-decode (argmax + un-letterbox + 3D lift via depth_m)
    5. Update QualityFrame:
         kpts_2d_px, kpts_3d_m, kp_conf, kp_sigma_m, pose_cov_diag
         box_conf, valid, valid_reason, valid_mask_bits
    6. Re-save (atomic, in-place)

사용 (Jetson):
    sudo PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PWD/src" \\
        python3 scripts/batch_pose_compute.py \\
        --input-dir /tmp/dump_test \\
        --engine src/perception/CUDA_Stream/yolo26s-lower6-v2.engine \\
        --schema lowlimb6

Mac 에서 (no CUDA): synthetic 검증 만:
    python3 scripts/batch_pose_compute.py --self-test

⚠️ Production hot path 영향 X. Jetson 에서 *별도 process* (dumps 의 offline 분석).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# repo src/ 자동
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from perception.CUDA_Stream.quality_dataset_io import (
    QualityFrame, save_frame_npz, load_frame_npz, _decode_rgb_jpeg,
)
from perception.CUDA_Stream.shm_publisher import (
    VALID_OK, INVALID_NO_DETECTION, TS_DOMAIN_EPOCH,
)

LOGGER = logging.getLogger("batch_pose")

# Default fallback sigma (used if real estimation unavailable)
DEFAULT_SIGMA_M_FALLBACK_VALUE = 0.015   # 15mm uniform

# kp_conf threshold for per-kp validity derive (matches gpu_postprocess)
KP_CONF_THRESHOLD = 0.5


def _self_test() -> int:
    """Mac executable — synthetic frame 의 round-trip 검증 (no CUDA).

    실제 YOLO 안 함. pose placeholder 채우기 path 만 검증.
    """
    LOGGER.info("=== self-test (Mac executable) ===")
    import tempfile
    K, H, W = 6, 600, 960
    rng = np.random.default_rng(42)

    # Natural-like RGB
    rgb_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(H):
        rgb_bgra[y, :, 0] = (y * 200) // H
        rgb_bgra[y, :, 1] = ((H - y) * 200) // H
    rgb_bgra[:, :, 3] = 255

    depth_m = (rng.random((H, W)).astype(np.float32) * 2.0) + 0.5

    frame = QualityFrame(
        frame_id=0, rgb_ts_ns=10**9, depth_ts_ns=10**9, depth_age_us=0,
        publish_done_mono_ns=10**9 + 1000,
        valid_mask_bits=0, valid_reason=INVALID_NO_DETECTION,
        ts_domain=TS_DOMAIN_EPOCH, valid=False,
        world_frame_applied=False,
        box_conf=0.0, depth_invalid_ratio=0.0,
        kpts_2d_px=np.zeros((K, 2), dtype=np.float32),
        kpts_3d_m=np.zeros((K, 3), dtype=np.float32),
        kp_conf=np.zeros((K,), dtype=np.float32),
        kp_sigma_m=np.full((K, 3), 0.015, dtype=np.float32),
        pose_cov_diag=np.full((K, 3), 0.015 ** 2, dtype=np.float32),
        rgb_bgra=rgb_bgra, depth_m=depth_m,
    )

    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "frame_000000.npz"
        save_frame_npz(npz_path, frame)
        LOGGER.info(f"saved synthetic frame: {npz_path.stat().st_size / 1024:.1f} KB")

        # Synthetic "pose detection result" (would come from YOLO TRT)
        fake_kpts_2d = np.array([
            [W // 2 - 20, H // 2 - 100],
            [W // 2 + 20, H // 2 - 100],
            [W // 2 - 30, H // 2],
            [W // 2 + 30, H // 2],
            [W // 2 - 25, H // 2 + 100],
            [W // 2 + 25, H // 2 + 100],
        ], dtype=np.float32)
        fake_kp_conf = np.array([0.95, 0.92, 0.88, 0.85, 0.80, 0.75], dtype=np.float32)
        fake_box_conf = 0.9

        # Lift to 3D via depth (synthetic camera intrinsics)
        fx, fy, cx, cy = 480.0, 480.0, W / 2, H / 2
        kpts_3d = np.zeros((K, 3), dtype=np.float32)
        for i in range(K):
            u, v = int(fake_kpts_2d[i, 0]), int(fake_kpts_2d[i, 1])
            z = depth_m[v, u]   # 1 px sample (real impl = 3x3 patch median)
            if not np.isfinite(z) or z <= 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            kpts_3d[i] = [x, y, z]

        # Per-kp validity derive
        z_valid = np.isfinite(kpts_3d[:, 2]) & (kpts_3d[:, 2] > 0)
        conf_valid = fake_kp_conf >= KP_CONF_THRESHOLD
        bit_valid = z_valid & conf_valid
        valid_mask = 0
        for i in range(K):
            if bit_valid[i]:
                valid_mask |= (1 << i)

        # Update + atomic re-save
        updated = QualityFrame(
            frame_id=frame.frame_id,
            rgb_ts_ns=frame.rgb_ts_ns,
            depth_ts_ns=frame.depth_ts_ns,
            depth_age_us=frame.depth_age_us,
            publish_done_mono_ns=time.monotonic_ns(),
            valid_mask_bits=valid_mask,
            valid_reason=VALID_OK if valid_mask != 0 else INVALID_NO_DETECTION,
            ts_domain=TS_DOMAIN_EPOCH,
            valid=valid_mask != 0,
            world_frame_applied=False,
            box_conf=fake_box_conf,
            depth_invalid_ratio=frame.depth_invalid_ratio,
            kpts_2d_px=fake_kpts_2d,
            kpts_3d_m=kpts_3d,
            kp_conf=fake_kp_conf,
            kp_sigma_m=np.full((K, 3), DEFAULT_SIGMA_M_FALLBACK_VALUE, dtype=np.float32),
            pose_cov_diag=np.full((K, 3), DEFAULT_SIGMA_M_FALLBACK_VALUE ** 2, dtype=np.float32),
            rgb_bgra=frame.rgb_bgra,
            depth_m=frame.depth_m,
            rgb_right_bgra=frame.rgb_right_bgra,
        )
        save_frame_npz(npz_path, updated)
        loaded = load_frame_npz(npz_path)

        # 검증
        assert loaded.valid_mask_bits == valid_mask
        assert loaded.valid == (valid_mask != 0)
        assert np.array_equal(loaded.kpts_3d_m, kpts_3d)
        LOGGER.info(f"✓ self-test PASS: mask=0b{valid_mask:06b}, valid={loaded.valid}")
    return 0


def _process_one_frame_jetson(
    npz_path: Path,
    output_path: Path,
    runner: Any,
    pre: Any,
    post: Any,
    sm: Any,
    schema: Any,
    calib: dict,
) -> Optional[dict]:
    """Jetson — one frame 의 npz → pose update → save to output_path.

    Production pipeline 의 *single-frame mode*. Bridge 우회 — npz 의 RGB + depth 만 사용.

    Args:
        npz_path: input frame_NNNNNN.npz (raw)
        output_path: output frame_NNNNNN.npz (posed, raw 보존)
        runner: TRTRunner
        pre: GpuPreprocessor
        post: GpuPostprocessor
        sm: StreamManager
        schema: KeypointSchema
        calib: dict (left_cam.fx, .fy, .cx, .cy, baseline_mm)

    Returns:
        Stats dict (box_conf, valid_mask_bits, n_valid_kp, infer_ms).
    """
    import torch
    import time as _time

    frame = load_frame_npz(npz_path)
    H, W = frame.depth_m.shape
    K = schema.num_keypoints

    # BGRA → BGR (drop alpha)
    bgr_np = frame.rgb_bgra[:, :, :3]   # (H, W, 3) uint8

    # Upload to GPU
    device = torch.device("cuda:0")
    bgr_gpu = torch.from_numpy(bgr_np).to(device, non_blocking=False)   # (H, W, 3) uint8
    depth_gpu = torch.from_numpy(frame.depth_m).to(device, non_blocking=False)   # (H, W) float32

    # GpuPreprocessor 의 API — production pipeline 동일 (BGRA / BGR 의 input)
    # GpuPreprocessor.preprocess(bgra_gpu) — 단 우리 = bgr.
    # 단순화: GpuPreprocessor 가 BGR 직접 받기 어려우면 BGRA 로 변환.
    bgra_gpu = torch.cat([bgr_gpu, torch.full_like(bgr_gpu[..., :1], 255)], dim=-1)

    # Preprocess (BGRA → letterbox 640×640 normalized)
    with torch.cuda.stream(sm.streams["pre"].stream):
        try:
            pre_out, lb_params = pre.preprocess(bgra_gpu)
        except Exception as e:
            LOGGER.error("preprocess failed at %s: %s", npz_path.name, e)
            return None

    sm.streams["pre"].stream.synchronize()

    # TRT inference
    t0 = _time.perf_counter()
    with torch.cuda.stream(sm.streams["infer"].stream):
        try:
            raw_output = runner.infer(pre_out, stream=sm.streams["infer"].stream)
        except Exception as e:
            LOGGER.error("infer failed at %s: %s", npz_path.name, e)
            return None
    sm.streams["infer"].stream.synchronize()
    infer_ms = (_time.perf_counter() - t0) * 1e3

    # Postprocess (un-letterbox + 3D lift)
    calibration_dict = {
        "fx": float(calib["left_cam"]["fx"]),
        "fy": float(calib["left_cam"]["fy"]),
        "cx": float(calib["left_cam"]["cx"]),
        "cy": float(calib["left_cam"]["cy"]),
        "R_world_from_cam": None,
    }
    with torch.cuda.stream(sm.streams["post"].stream):
        try:
            pose_result = post.post(
                raw_output=raw_output,
                lb_params=lb_params,
                depth_hw=depth_gpu,
                calibration=calibration_dict,
                stream=sm.streams["post"].stream,
                ts_s=frame.rgb_ts_ns / 1e9,
                scalar_host=None,
            )
        except Exception as e:
            LOGGER.error("post failed at %s: %s", npz_path.name, e)
            return None
    sm.streams["post"].stream.synchronize()

    # PoseResult → numpy
    kpts_3d_np = pose_result.kpts_3d_m.cpu().numpy().astype(np.float32)
    kpts_2d_np = pose_result.kpts_2d_px.cpu().numpy().astype(np.float32)
    kp_conf_np = pose_result.kpt_conf.cpu().numpy().astype(np.float32)

    # Per-kp validity derive (production 동일 logic)
    z_valid = np.isfinite(kpts_3d_np[:, 2]) & (kpts_3d_np[:, 2] > 0)
    conf_valid = kp_conf_np >= KP_CONF_THRESHOLD
    bit_valid = z_valid & conf_valid
    valid_mask_bits = 0
    for i in range(K):
        if bit_valid[i]:
            valid_mask_bits |= (1 << i)

    is_valid_frame = (
        valid_mask_bits != 0
        and pose_result.box_conf >= 0.25   # detection threshold
    )

    # kp_sigma_m (현재 default uniform — production 의 진정 추정 의무, Codex P2-4)
    kp_sigma_m = np.full((K, 3), 0.015, dtype=np.float32)
    pose_cov_diag = (kp_sigma_m ** 2).astype(np.float32)

    # Update QualityFrame + atomic save
    updated = QualityFrame(
        frame_id=frame.frame_id,
        rgb_ts_ns=frame.rgb_ts_ns,
        depth_ts_ns=frame.depth_ts_ns,
        depth_age_us=frame.depth_age_us,
        publish_done_mono_ns=time.monotonic_ns(),
        valid_mask_bits=valid_mask_bits,
        valid_reason=VALID_OK if is_valid_frame else INVALID_NO_DETECTION,
        ts_domain=TS_DOMAIN_EPOCH,
        valid=is_valid_frame,
        world_frame_applied=False,
        box_conf=float(pose_result.box_conf),
        depth_invalid_ratio=frame.depth_invalid_ratio,
        kpts_2d_px=kpts_2d_np,
        kpts_3d_m=kpts_3d_np,
        kp_conf=kp_conf_np,
        kp_sigma_m=kp_sigma_m,
        pose_cov_diag=pose_cov_diag,
        rgb_bgra=frame.rgb_bgra,
        depth_m=frame.depth_m,
        rgb_right_bgra=frame.rgb_right_bgra,
    )
    save_frame_npz(output_path, updated)

    return {
        "frame_id": frame.frame_id,
        "valid": is_valid_frame,
        "valid_mask_bits": valid_mask_bits,
        "n_valid_kp": int(bin(valid_mask_bits).count("1")),
        "box_conf": float(pose_result.box_conf),
        "infer_ms": infer_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path,
                    help="dump_quality_dataset 의 output dir (frame_*.npz 포함)")
    ap.add_argument("--engine", type=Path,
                    help="TRT engine path (Jetson)")
    ap.add_argument("--schema", default="lowlimb6",
                    choices=["coco17", "lowlimb6"])
    ap.add_argument("--calib", type=Path,
                    help="session_calib.json (default: --input-dir/session_calib.json)")
    ap.add_argument("--output-dir", type=Path,
                    help="output dir (default = <input-dir>_pose). raw 보존.")
    ap.add_argument("--overwrite", action="store_true",
                    help="DEPRECATED — Codex Q3 권유: raw 보존 + new posed dir. "
                         "true 시 input dir 의 frame 을 overwrite.")
    ap.add_argument("--self-test", action="store_true",
                    help="Mac executable — synthetic round-trip 만 검증")
    ap.add_argument("--limit", type=int, default=0,
                    help="N frames 만 처리 (0 = 모두)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        stream=sys.stdout)

    if args.self_test:
        return _self_test()

    if not args.input_dir or not args.engine:
        LOGGER.error("--input-dir + --engine 의무 (또는 --self-test)")
        return 1

    if not args.input_dir.is_dir():
        LOGGER.error("input-dir not found: %s", args.input_dir)
        return 1

    calib_path = args.calib or (args.input_dir / "session_calib.json")
    if not calib_path.exists():
        LOGGER.error("session_calib.json not found: %s", calib_path)
        return 1

    with open(calib_path, "r") as f:
        calib = json.load(f)

    # Jetson-only path: import perception.* 의 production pipeline 모듈.
    try:
        import torch
        from perception.CUDA_Stream.trt_runner import TRTRunner
        from perception.CUDA_Stream.gpu_preprocess import GpuPreprocessor
        from perception.CUDA_Stream.gpu_postprocess import GpuPostprocessor
        from perception.CUDA_Stream.stream_manager import StreamManager
        from perception.CUDA_Stream.keypoint_config import get_schema
    except ImportError as e:
        LOGGER.error("Jetson-only import failed: %s", e)
        return 1

    if not torch.cuda.is_available():
        LOGGER.error("CUDA 부재 — Jetson 에서만 실행 가능")
        return 1

    schema = get_schema(args.schema)
    device = torch.device("cuda:0")
    sm = StreamManager(device=device, high_priority_stages=None)   # all-high
    runner = TRTRunner(str(args.engine), device=device)
    pre = GpuPreprocessor(imgsz=640, device=device)
    post = GpuPostprocessor(schema=schema, device=device, use_filter=False)

    # ★ Codex Q3 fix: new posed dir (raw 보존)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_dir.parent / (args.input_dir.name + "_pose")
    if args.overwrite:
        output_dir = args.input_dir   # backward compat
        LOGGER.warning("--overwrite — in-place update (raw lost)")
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("posed output → %s", output_dir)

    # Copy session_calib.json to output (raw 보존, 단 posed dir 도 동일 calib)
    calib_dst = output_dir / "session_calib.json"
    if not calib_dst.exists() and not args.overwrite:
        import shutil
        shutil.copy2(calib_path, calib_dst)
        LOGGER.info("copied session_calib.json → %s", calib_dst)

    npz_files = sorted(args.input_dir.glob("frame_*.npz"))
    if args.limit > 0:
        npz_files = npz_files[:args.limit]

    LOGGER.info("processing %d frames from %s → %s",
                len(npz_files), args.input_dir, output_dir)
    LOGGER.info("calib: fx=%.2f, baseline=%.2fmm",
                calib["left_cam"]["fx"], calib["baseline_mm"])

    t_start = time.time()
    processed = 0
    valid_count = 0
    infer_ms_list = []
    for npz_path in npz_files:
        output_path = output_dir / npz_path.name
        stats = _process_one_frame_jetson(
            npz_path, output_path,
            runner, pre, post, sm, schema, calib,
        )
        if stats is not None:
            if stats.get("valid", False):
                valid_count += 1
            infer_ms_list.append(stats.get("infer_ms", 0))
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            LOGGER.info("processed %d/%d (elapsed %.1fs)",
                        processed, len(npz_files), elapsed)

    elapsed = time.time() - t_start
    avg_infer = float(np.mean(infer_ms_list)) if infer_ms_list else 0.0
    LOGGER.info("DONE: processed=%d valid=%d elapsed=%.1fs (%.1f fps, avg infer=%.2fms)",
                processed, valid_count, elapsed,
                processed / max(elapsed, 0.001), avg_infer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
