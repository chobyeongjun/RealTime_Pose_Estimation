"""SHM → npz dumper + replay reader.

Two components share this file so the writer and reader layouts stay in
sync:

  * ``main()``: Jetson-side dumper. Runs next to the Track B publisher,
    attaches to ``/dev/shm/hwalker_pose_cuda``, and records every unique
    SHM snapshot to a compact ``.npz`` for off-Jetson replay.

  * ``DumpReader``: drop-in replacement for :class:`ShmReader`. Reads from
    the dumped ``.npz`` and exposes the same ``.read() → tuple`` /
    ``.close()`` interface. Lets ``view_sagittal.py`` render on any
    machine (Mac, laptop) with just ``numpy`` and ``opencv-python`` —
    no ZED SDK, no CUDA, no TensorRT, no live camera.

Workflow::

    Jetson ─ record_svo ──────────────▶ walk.svo2
    Jetson ─ launch_clean.sh (live) ───┐
                                       ├──▶ /dev/shm/hwalker_pose_cuda
    Jetson ─ dump_shm_stream --out  ───┘       │
                                               ▼
                                          walk_pose.npz   (scp to Mac)
                                               │
                                               ▼
    Mac    ─ view_sagittal --dump-file walk_pose.npz ─ iterate on viewer

``.npz`` layout (N frames, K keypoints):

    frame_id         (N,)       int64
    ts_ns            (N,)       uint64
    kpts_3d          (N, K, 3)  float32   — world frame, meters
    kpt_conf         (N, K)     float32
    kpts_2d          (N, K, 2)  float32
    box_conf         (N,)       float32
    valid            (N,)       bool
    depth_inv_ratio  (N,)       float32
    meta             () object  — JSON string: schema, K, duration, ...

Usage (Jetson)::

    # Start the production pipeline in another terminal first.
    PYTHONPATH=src python3 -m perception.CUDA_Stream.dump_shm_stream \\
        --out data/recordings/$(date +%Y-%m-%d)/walk_01_pose.npz \\
        --duration 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .keypoint_config import get_schema
from .shm_publisher import DEFAULT_NAME, ShmReader


# ---------------------------------------------------------------------------
# DumpReader — used by view_sagittal.py when --dump-file is given.
# ---------------------------------------------------------------------------

class DumpReader:
    """Replay an .npz produced by :func:`main` as if it were a live SHM."""

    def __init__(
        self,
        path: str,
        expected_k: Optional[int] = None,
        loop: bool = True,
    ) -> None:
        data = np.load(path, allow_pickle=True)
        self.frame_id = data["frame_id"]
        self.ts_ns = data["ts_ns"]
        self.kpts_3d = data["kpts_3d"]
        self.kpt_conf = data["kpt_conf"]
        self.kpts_2d = data["kpts_2d"]
        self.box_conf = data["box_conf"]
        self.valid = data["valid"]
        self.depth_inv_ratio = data["depth_inv_ratio"]
        # world_frame_applied was added later — old dumps default to False
        self.world_frame_applied = (
            data["world_frame_applied"].astype(bool)
            if "world_frame_applied" in data else
            np.zeros(self.frame_id.shape[0], dtype=bool)
        )
        try:
            self.meta = json.loads(str(data["meta"]))
        except Exception:
            self.meta = {}

        self.N = int(self.frame_id.shape[0])
        if self.N == 0:
            raise RuntimeError(f"dump '{path}' contains 0 frames")

        self.K = int(self.kpts_3d.shape[1])
        if expected_k is not None and expected_k != self.K:
            raise RuntimeError(
                f"dump keypoint count mismatch: file K={self.K}, "
                f"expected K={expected_k}"
            )

        self.loop = bool(loop)
        self.idx = 0

    def read(self) -> Optional[Tuple]:
        if self.idx >= self.N:
            if not self.loop:
                return None
            self.idx = 0
        i = self.idx
        self.idx += 1
        # v2 (2026-05-11): 17-tuple matching live ShmReader (Codex orchestration
        # bvfvkxo1m). Legacy npz files (v1) 는 rgb_ts == ts_ns 로 fallback,
        # depth_ts == rgb_ts (same-frame), 새 fields default 값.
        K = int(self.kpts_3d.shape[1]) if self.kpts_3d.ndim == 3 else 6
        ts_ns = int(self.ts_ns[i])
        kpts_3d = self.kpts_3d[i].astype(np.float32, copy=False)
        kp_conf = self.kpt_conf[i].astype(np.float32, copy=False)
        valid = bool(self.valid[i])
        return (
            int(self.frame_id[i]),
            ts_ns,                                          # rgb_ts_ns
            ts_ns,                                          # depth_ts_ns (same-frame fallback)
            0,                                              # depth_age_us
            kpts_3d,
            kp_conf,
            self.kpts_2d[i].astype(np.float32, copy=False),
            np.full((K, 3), 0.015, dtype=np.float32),       # kp_sigma_m default
            np.full((K, 3), 0.015 ** 2, dtype=np.float32),  # pose_cov_diag default
            float(self.box_conf[i]),
            valid,
            float(self.depth_inv_ratio[i]),
            bool(self.world_frame_applied[i]),
            0,                                              # publish_done_mono_ns
            0,                                              # valid_reason
            0,                                              # ts_domain
            ((1 << K) - 1) if valid else 0,                # valid_mask_bits
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Dumper CLI — Jetson only (needs the live SHM publisher running).
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--shm", default=DEFAULT_NAME.lstrip("/"),
                    help="SHM segment name (default: hwalker_pose_cuda)")
    ap.add_argument("--schema", default="lowlimb6",
                    choices=["coco17", "lowlimb6"])
    ap.add_argument("--out", required=True,
                    help="output .npz path (e.g. data/recordings/.../walk_01_pose.npz)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds to dump (default 30)")
    ap.add_argument("--skip-invalid", dest="skip_invalid", action="store_true",
                    help="skip frames where valid=False (smaller file)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    schema = get_schema(args.schema)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
        print(f"[dump] note: forced .npz extension → {out_path}")

    try:
        reader = ShmReader(name=args.shm, expected_k=schema.num_keypoints)
    except FileNotFoundError:
        print(f"SHM '/dev/shm/{args.shm}' not found. Is the Track B "
              f"pipeline (run_stream_demo) running?", file=sys.stderr)
        return 2

    frame_ids = []
    ts_list = []
    kpts_3d_list = []
    kpt_conf_list = []
    kpts_2d_list = []
    box_conf_list = []
    valid_list = []
    depth_inv_list = []
    world_frame_list = []

    last_frame_id = -1
    dup_skipped = 0
    invalid_skipped = 0

    print(f"[dump] schema={args.schema} K={schema.num_keypoints}")
    print(f"[dump] duration={args.duration}s  skip_invalid={args.skip_invalid}")
    print(f"[dump] out={out_path}")
    print(f"[dump] Ctrl-C to stop early")

    t_start = time.monotonic()
    try:
        while time.monotonic() - t_start < args.duration:
            data = reader.read()
            if data is None:
                time.sleep(0.001)
                continue
            # v2 (2026-05-11): ShmReader returns 17-tuple. dumper 는 npz format
            # 호환 위해 *legacy fields 만* archive (rgb_ts → ts_ns).
            # 진정 quality dataset (kp_sigma_m / pose_cov_diag) 는
            # dump_quality_dataset.py (Week 0 다음 작업) 가 처리.
            (frame_id, rgb_ts_ns, _depth_ts_ns, _depth_age_us,
             kpts_3d, kpt_conf, kpts_2d, _kp_sigma, _pose_cov,
             box_conf, valid, depth_inv, world_frame_applied,
             _publish_done, _valid_reason, _ts_domain,
             _valid_mask_bits) = data
            ts_ns = rgb_ts_ns   # legacy field name 유지
            if frame_id == last_frame_id:
                dup_skipped += 1
                time.sleep(0.001)
                continue
            last_frame_id = frame_id
            if args.skip_invalid and not valid:
                invalid_skipped += 1
                continue
            frame_ids.append(frame_id)
            ts_list.append(ts_ns)
            kpts_3d_list.append(kpts_3d)
            kpt_conf_list.append(kpt_conf)
            kpts_2d_list.append(kpts_2d)
            box_conf_list.append(box_conf)
            valid_list.append(valid)
            depth_inv_list.append(depth_inv)
            world_frame_list.append(world_frame_applied)
            n = len(frame_ids)
            if n % 60 == 0:
                elapsed = time.monotonic() - t_start
                print(f"[dump]  t={elapsed:5.1f}s  kept={n}  "
                      f"dup_skip={dup_skipped}  "
                      f"invalid_skip={invalid_skipped}", flush=True)
    except KeyboardInterrupt:
        print("\n[dump] Ctrl-C received")
    finally:
        reader.close()

    N = len(frame_ids)
    if N == 0:
        print("[dump] ERROR: 0 frames captured. Is valid=True being published?",
              file=sys.stderr)
        return 3

    duration_s = (ts_list[-1] - ts_list[0]) / 1e9 if N >= 2 else 0.0
    meta = {
        "schema": args.schema,
        "K": schema.num_keypoints,
        "keypoint_names": list(schema.keypoints),
        "first_ts_ns": int(ts_list[0]),
        "last_ts_ns": int(ts_list[-1]),
        "duration_s": duration_s,
        "skip_invalid": args.skip_invalid,
        "dup_skipped": dup_skipped,
        "invalid_skipped": invalid_skipped,
    }

    np.savez_compressed(
        out_path,
        frame_id=np.asarray(frame_ids, dtype=np.int64),
        ts_ns=np.asarray(ts_list, dtype=np.uint64),
        kpts_3d=np.stack(kpts_3d_list, axis=0).astype(np.float32),
        kpt_conf=np.stack(kpt_conf_list, axis=0).astype(np.float32),
        kpts_2d=np.stack(kpts_2d_list, axis=0).astype(np.float32),
        box_conf=np.asarray(box_conf_list, dtype=np.float32),
        valid=np.asarray(valid_list, dtype=bool),
        depth_inv_ratio=np.asarray(depth_inv_list, dtype=np.float32),
        world_frame_applied=np.asarray(world_frame_list, dtype=bool),
        meta=np.asarray(json.dumps(meta), dtype=object),
    )

    size_mb = out_path.stat().st_size / 1e6
    fps = N / duration_s if duration_s > 0 else 0.0
    print(f"[dump] DONE")
    print(f"[dump]   frames         : {N}")
    print(f"[dump]   duration       : {duration_s:.1f}s")
    print(f"[dump]   avg fps        : {fps:.1f}")
    print(f"[dump]   dup_skipped    : {dup_skipped}")
    print(f"[dump]   invalid_skipped: {invalid_skipped}")
    print(f"[dump]   size           : {size_mb:.2f} MB")
    print(f"[dump]   file           : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
