"""Quality dataset I/O — save/load/verify (Mac executable, no CUDA).

작성: 2026-05-12. Codex orchestration `bvfvkxo1m` Q2(a) spec.

Quality dataset = recorded session 의 *raw RGB + depth + pose + calib + timestamps + valid_mask*.
Plan D EKF (사용자 control repo) + V4L2 sparse stereo baseline + Mocap RMSE 의 prerequisite.

Schema (per-frame npz):
    frame_id            uint32
    rgb_ts_ns           uint64    (T_N, RGB capture)
    depth_ts_ns         uint64    (T_{N-1} if 1-frame-late, else T_N)
    depth_age_us        uint32    ((rgb_ts - depth_ts) / 1000)
    publish_done_mono_ns uint64    (CLOCK_MONOTONIC, post pipeline)
    rgb_bgra_jpeg       bytes     (JPEG encoded, quality=90)
    rgb_right_bgra_jpeg bytes     (optional, --include-right)
    depth_m             float32 (H, W)   (raw, np.savez_compressed)
    kpts_2d_px          float32 (K, 2)
    kpts_3d_m           float32 (K, 3)
    kp_conf             float32 (K,)
    kp_sigma_m          float32 (K, 3)    (SHM v2 — depth uncertainty)
    pose_cov_diag       float32 (K, 3)    (SHM v2 — pose covariance)
    valid_mask_bits     uint64    (per-kp validity)
    valid_reason        uint8     (VALID_REASON_*)
    world_frame_applied bool
    box_conf            float32
    depth_invalid_ratio float32

이 file 은 *Mac + Jetson 모두 import 가능* (no CUDA/ZED dependency).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# Schema definition — 필수 fields (필수) + optional fields
# (사용자 control repo 와 동기 보장).
SCHEMA_VERSION = 1

REQUIRED_SCALAR_FIELDS = {
    "frame_id": (np.uint32, ()),
    "rgb_ts_ns": (np.uint64, ()),
    "depth_ts_ns": (np.uint64, ()),
    "depth_age_us": (np.uint32, ()),
    "publish_done_mono_ns": (np.uint64, ()),
    "valid_mask_bits": (np.uint64, ()),
    "valid_reason": (np.uint8, ()),
    "world_frame_applied": (np.bool_, ()),
    "box_conf": (np.float32, ()),
    "depth_invalid_ratio": (np.float32, ()),
}

REQUIRED_ARRAY_FIELDS_DYNAMIC = (
    # field_name, dtype, axis-name pairs (K-dependent)
    ("kpts_2d_px", np.float32, ("K", 2)),
    ("kpts_3d_m", np.float32, ("K", 3)),
    ("kp_conf", np.float32, ("K",)),
    ("kp_sigma_m", np.float32, ("K", 3)),
    ("pose_cov_diag", np.float32, ("K", 3)),
)

REQUIRED_BLOB_FIELDS = ("rgb_bgra_jpeg", "depth_m")
OPTIONAL_BLOB_FIELDS = ("rgb_right_bgra_jpeg",)


@dataclass
class QualityFrame:
    """Per-frame quality dataset entry (Plan D EKF + V4L2 baseline compat)."""

    # Scalars
    frame_id: int
    rgb_ts_ns: int
    depth_ts_ns: int
    depth_age_us: int
    publish_done_mono_ns: int
    valid_mask_bits: int
    valid_reason: int
    world_frame_applied: bool
    box_conf: float
    depth_invalid_ratio: float

    # Arrays
    kpts_2d_px: np.ndarray         # (K, 2) float32
    kpts_3d_m: np.ndarray          # (K, 3) float32
    kp_conf: np.ndarray            # (K,) float32
    kp_sigma_m: np.ndarray         # (K, 3) float32
    pose_cov_diag: np.ndarray      # (K, 3) float32

    # Image blobs
    rgb_bgra: np.ndarray           # (H, W, 4) uint8 — encoded to JPEG on save
    depth_m: np.ndarray            # (H, W) float32 — raw
    rgb_right_bgra: Optional[np.ndarray] = None    # (H, W, 4) uint8 optional


def _encode_rgb_jpeg(rgb_bgra: np.ndarray, quality: int = 90) -> bytes:
    """BGRA → RGB JPEG bytes. PIL 또는 cv2 활용."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PIL (Pillow) required for JPEG encoding. pip install Pillow"
        ) from exc

    if rgb_bgra.ndim != 3 or rgb_bgra.shape[2] != 4:
        raise ValueError(
            f"rgb_bgra must be (H, W, 4) BGRA, got {rgb_bgra.shape}"
        )
    if rgb_bgra.dtype != np.uint8:
        raise ValueError(f"rgb_bgra must be uint8, got {rgb_bgra.dtype}")

    # BGRA → RGB (drop alpha + swap)
    rgb = rgb_bgra[:, :, [2, 1, 0]]    # B, G, R → R, G, B
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode_rgb_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes → BGRA (H, W, 4) uint8."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL required") from exc

    img = Image.open(io.BytesIO(jpeg_bytes))
    rgb = np.array(img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"decoded JPEG must be (H, W, 3), got {rgb.shape}")
    # RGB → BGRA (add alpha=255)
    h, w = rgb.shape[:2]
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[:, :, 0] = rgb[:, :, 2]   # B
    bgra[:, :, 1] = rgb[:, :, 1]   # G
    bgra[:, :, 2] = rgb[:, :, 0]   # R
    bgra[:, :, 3] = 255            # A
    return bgra


def save_frame_npz(
    output_path: Path,
    frame: QualityFrame,
    jpeg_quality: int = 90,
    compress: bool = True,
) -> int:
    """Save QualityFrame to .npz.

    Returns:
        Total bytes written (approximate).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode RGB to JPEG
    rgb_jpeg = _encode_rgb_jpeg(frame.rgb_bgra, quality=jpeg_quality)
    rgb_right_jpeg = None
    if frame.rgb_right_bgra is not None:
        rgb_right_jpeg = _encode_rgb_jpeg(frame.rgb_right_bgra, quality=jpeg_quality)

    # Build save dict
    data: Dict[str, Any] = {
        # scalars (as 0-d np arrays for type preservation)
        "frame_id": np.uint32(frame.frame_id),
        "rgb_ts_ns": np.uint64(frame.rgb_ts_ns),
        "depth_ts_ns": np.uint64(frame.depth_ts_ns),
        "depth_age_us": np.uint32(frame.depth_age_us),
        "publish_done_mono_ns": np.uint64(frame.publish_done_mono_ns),
        "valid_mask_bits": np.uint64(frame.valid_mask_bits),
        "valid_reason": np.uint8(frame.valid_reason),
        "world_frame_applied": np.bool_(frame.world_frame_applied),
        "box_conf": np.float32(frame.box_conf),
        "depth_invalid_ratio": np.float32(frame.depth_invalid_ratio),
        # arrays
        "kpts_2d_px": frame.kpts_2d_px.astype(np.float32, copy=False),
        "kpts_3d_m": frame.kpts_3d_m.astype(np.float32, copy=False),
        "kp_conf": frame.kp_conf.astype(np.float32, copy=False),
        "kp_sigma_m": frame.kp_sigma_m.astype(np.float32, copy=False),
        "pose_cov_diag": frame.pose_cov_diag.astype(np.float32, copy=False),
        # blobs (JPEG bytes + raw depth)
        "rgb_bgra_jpeg": np.frombuffer(rgb_jpeg, dtype=np.uint8),
        "depth_m": frame.depth_m.astype(np.float32, copy=False),
        # schema version
        "schema_version": np.uint32(SCHEMA_VERSION),
    }
    if rgb_right_jpeg is not None:
        data["rgb_right_bgra_jpeg"] = np.frombuffer(rgb_right_jpeg, dtype=np.uint8)

    if compress:
        np.savez_compressed(output_path, **data)
    else:
        np.savez(output_path, **data)

    return output_path.stat().st_size


def load_frame_npz(input_path: Path) -> QualityFrame:
    """Load QualityFrame from .npz. Reverse of save_frame_npz."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"npz not found: {input_path}")

    with np.load(input_path, allow_pickle=False) as npz:
        # Schema version check
        sv = int(npz["schema_version"])
        if sv != SCHEMA_VERSION:
            raise RuntimeError(
                f"schema version mismatch: file={sv}, expected={SCHEMA_VERSION}"
            )

        # Decode JPEG
        rgb_jpeg_bytes = bytes(npz["rgb_bgra_jpeg"])
        rgb_bgra = _decode_rgb_jpeg(rgb_jpeg_bytes)

        rgb_right_bgra = None
        if "rgb_right_bgra_jpeg" in npz.files:
            rgb_right_jpeg_bytes = bytes(npz["rgb_right_bgra_jpeg"])
            rgb_right_bgra = _decode_rgb_jpeg(rgb_right_jpeg_bytes)

        return QualityFrame(
            frame_id=int(npz["frame_id"]),
            rgb_ts_ns=int(npz["rgb_ts_ns"]),
            depth_ts_ns=int(npz["depth_ts_ns"]),
            depth_age_us=int(npz["depth_age_us"]),
            publish_done_mono_ns=int(npz["publish_done_mono_ns"]),
            valid_mask_bits=int(npz["valid_mask_bits"]),
            valid_reason=int(npz["valid_reason"]),
            world_frame_applied=bool(npz["world_frame_applied"]),
            box_conf=float(npz["box_conf"]),
            depth_invalid_ratio=float(npz["depth_invalid_ratio"]),
            kpts_2d_px=npz["kpts_2d_px"].copy(),
            kpts_3d_m=npz["kpts_3d_m"].copy(),
            kp_conf=npz["kp_conf"].copy(),
            kp_sigma_m=npz["kp_sigma_m"].copy(),
            pose_cov_diag=npz["pose_cov_diag"].copy(),
            rgb_bgra=rgb_bgra,
            depth_m=npz["depth_m"].copy(),
            rgb_right_bgra=rgb_right_bgra,
        )


def verify_frame_schema(npz_path: Path, expected_k: Optional[int] = None) -> None:
    """Verify .npz file의 schema 정확성. raise on failure."""
    with np.load(npz_path, allow_pickle=False) as npz:
        files = set(npz.files)

        # Required scalars
        for fname, (dtype, shape) in REQUIRED_SCALAR_FIELDS.items():
            if fname not in files:
                raise ValueError(f"missing required scalar field: {fname}")
            arr = npz[fname]
            if arr.shape != shape:
                raise ValueError(
                    f"{fname}: shape mismatch (expected {shape}, got {arr.shape})"
                )

        # Required arrays
        for fname, dtype, shape_spec in REQUIRED_ARRAY_FIELDS_DYNAMIC:
            if fname not in files:
                raise ValueError(f"missing required array field: {fname}")
            arr = npz[fname]
            if arr.dtype != dtype:
                raise ValueError(
                    f"{fname}: dtype mismatch (expected {dtype}, got {arr.dtype})"
                )
            # K consistency check (later)

        # Required blobs
        for fname in REQUIRED_BLOB_FIELDS:
            if fname not in files:
                raise ValueError(f"missing required blob field: {fname}")

        # K consistency: kpts_3d_m / kpts_2d_px / kp_conf / kp_sigma_m / pose_cov_diag
        ks = []
        ks.append(npz["kpts_3d_m"].shape[0])
        ks.append(npz["kpts_2d_px"].shape[0])
        ks.append(npz["kp_conf"].shape[0])
        ks.append(npz["kp_sigma_m"].shape[0])
        ks.append(npz["pose_cov_diag"].shape[0])
        if len(set(ks)) != 1:
            raise ValueError(f"K mismatch across array fields: {ks}")
        if expected_k is not None and ks[0] != expected_k:
            raise ValueError(f"K={ks[0]}, expected={expected_k}")


def save_session_calib(
    output_path: Path,
    calib: Dict[str, Any],
) -> None:
    """Save session calibration JSON. 1회 per session (start).

    Required fields (Codex Q3):
        version, session_start_ns, session_start_mono_ns,
        zed_serial, zed_sdk_version,
        resolution_width, resolution_height, fps, depth_mode,
        self_calibration_disabled (bool),
        left_cam = {fx, fy, cx, cy, disto[]},
        right_cam = {fx, fy, cx, cy, disto[]},
        baseline_mm, stereo_transform (4x4 list).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = (
        "version", "session_start_ns", "session_start_mono_ns",
        "zed_serial", "zed_sdk_version",
        "resolution_width", "resolution_height", "fps", "depth_mode",
        "self_calibration_disabled",
        "left_cam", "right_cam",
        "baseline_mm",
    )
    for k in required:
        if k not in calib:
            raise ValueError(f"session calib missing field: {k}")

    # nested
    for cam in ("left_cam", "right_cam"):
        cam_dict = calib[cam]
        for k in ("fx", "fy", "cx", "cy"):
            if k not in cam_dict:
                raise ValueError(f"{cam} missing intrinsic: {k}")

    with open(output_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_session_calib(input_path: Path) -> Dict[str, Any]:
    """Load + verify session calibration JSON."""
    with open(input_path, "r") as f:
        calib = json.load(f)

    # re-verify by save semantics
    if "version" not in calib or calib["version"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"calib version mismatch: file={calib.get('version')}, "
            f"expected={SCHEMA_VERSION}"
        )

    return calib


__all__ = [
    "SCHEMA_VERSION",
    "QualityFrame",
    "save_frame_npz",
    "load_frame_npz",
    "verify_frame_schema",
    "save_session_calib",
    "load_session_calib",
]
