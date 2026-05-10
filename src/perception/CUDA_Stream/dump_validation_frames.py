"""dump_validation_frames.py — paired RGB/depth/pose 덤프 (Phase B accuracy 검증 prerequisite).

Codex consult Q8 (2026-05-10).

목적:
    YOLO26s-lower6 → YOLO26n-lower6 (또는 다른 model) 비교의 ground truth 생성.
    동일 RGB/depth 입력에 대해 frame_id 별 .npz 저장 → 오프라인 RMSE 비교
    (scripts/compare_pose_outputs.py).

사용법 (Phase B 진입 시):
    sudo python3 -m perception.CUDA_Stream.dump_validation_frames \\
        --output dumps/yolo26s-lower6/ --count 500

Phase B Pass criteria (Codex Q3):
    - per-keypoint 2D RMSE max ≤ 6 px
    - 3D RMSE: hip/knee ≤ 15 mm, ankle ≤ 25 mm
    - valid drop ≤ 1 %
    - side bias |L-R| mean ≤ 10 mm

⚠️ NOTE — Phase B 진입 시점에 wire-up 채움.
    이 skeleton 은 인프라 prep 만 (사용자 D 결정 — A measurement 후 B 진입 결정).
    Jetson 측정 시점에 ZEDGpuBridge.latest() + GpuPostprocessor.post() 의
    minimal loop 로 채움. hot path 절대 변경 X — 별도 entry point.

출력 (per frame, .npz):
    rgb_bgra  : (H, W, 4) uint8
    depth_m   : (H, W) float32
    calib     : json string (fx, fy, cx, cy, baseline)
    ts_ns     : int (capture timestamp)
    kpts_2d   : (6, 2) float32
    kpts_3d_m : (6, 3) float32
    kp_conf   : (6,) float32
    valid_mask: (6,) bool
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--count", type=int, default=500,
                    help="frames to dump")
    ap.add_argument("--every", type=int, default=2,
                    help="sample 1 in N valid frames")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "dump_validation_frames — skeleton (Phase B 진입 시 wire-up). "
        "output=%s count=%d every=%d imgsz=%d",
        args.output, args.count, args.every, args.imgsz,
    )
    LOGGER.warning(
        "현재는 Phase A 인프라 prep — A 측정 부족 시 B 진입할 때 wire-up. "
        "Jetson 에서 ZEDGpuBridge + GpuPostprocessor 직접 호출하는 minimal loop "
        "구현 후 동일 RGB/depth 의 (.npz) 덤프 시작. NEXT PR 에서 채움."
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
