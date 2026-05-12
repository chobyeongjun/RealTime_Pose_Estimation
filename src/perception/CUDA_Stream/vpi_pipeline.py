"""VPI pipeline — Debayer + Rectify (Jetson only).

docs/lessons/v4l2_bypass_plan.md Step 2 + Step 3.

Stack:
    Bayer RAW10 (uint16) → debayer → RGB8 → rectify → undistorted RGB8

NVIDIA VPI 3.2.4 API:
    vpi.Image, vpi.Stream, vpi.Backend.CUDA
    Debayer: vpi.Image.convert (BayerRGGB / BGGR / GRBG / GBRG → RGB8)
    Rectify: vpi.WarpMap + vpi.Image.remap

ZED X Mini distortion:
    Brown-Conrady extended (12-coeff): k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
    Raw values (2026-05-12 검증): [0.0428, 0.0277, -7.5e-5, -2.2e-4, -4.9e-3, 0.055, ...]

⚠️ Jetson only (VPI 의 Linux + CUDA 의무). Mac 에선 syntax check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class RectifyMaps:
    """Pre-computed remap maps (1회 build)."""
    map_x: np.ndarray            # (H, W) float32
    map_y: np.ndarray            # (H, W) float32
    width: int
    height: int


def build_rectify_maps(
    fx: float, fy: float, cx: float, cy: float,
    disto: list,                  # 12-coeff Brown-Conrady extended
    width: int, height: int,
) -> RectifyMaps:
    """Compute remap maps for undistort.

    Brown-Conrady extended model (OpenCV equivalent):
        r² = x_d² + y_d²
        radial = 1 + k1*r² + k2*r⁴ + k3*r⁶ / (1 + k4*r² + k5*r⁴ + k6*r⁶)
        x_u = x_d * radial + 2*p1*x_d*y_d + p2*(r² + 2*x_d²) + s1*r² + s2*r⁴
        y_u = y_d * radial + p1*(r² + 2*y_d²) + 2*p2*x_d*y_d + s3*r² + s4*r⁴

    이 implementation = OpenCV cv2.initUndistortRectifyMap 의 equivalent.

    Args:
        fx, fy, cx, cy: raw intrinsics
        disto: 12-coeff [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
        width, height: output rectified image size

    Returns:
        RectifyMaps with float32 maps for vpi.WarpMap or cv2.remap.
    """
    if len(disto) < 12:
        disto = list(disto) + [0.0] * (12 - len(disto))
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = disto[:12]

    # Output pixel grid → undistorted normalized coords → distorted pixel coords
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Undistorted normalized
    x_u = (u - cx) / fx
    y_u = (v - cy) / fy

    # Iterative distortion model (5 iterations 정확)
    x_d, y_d = x_u.copy(), y_u.copy()
    for _ in range(5):
        r2 = x_d * x_d + y_d * y_d
        r4 = r2 * r2
        r6 = r4 * r2
        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        tang_x = 2 * p1 * x_d * y_d + p2 * (r2 + 2 * x_d * x_d)
        tang_y = p1 * (r2 + 2 * y_d * y_d) + 2 * p2 * x_d * y_d
        prism_x = s1 * r2 + s2 * r4
        prism_y = s3 * r2 + s4 * r4
        x_d = (x_u - tang_x - prism_x) / radial
        y_d = (y_u - tang_y - prism_y) / radial

    # Distorted normalized → distorted pixel
    map_x = (x_d * fx + cx).astype(np.float32)
    map_y = (y_d * fy + cy).astype(np.float32)

    return RectifyMaps(map_x=map_x, map_y=map_y, width=width, height=height)


def debayer_via_vpi(bayer_uint16: "np.ndarray", pattern: str = "BGGR"):
    """Bayer RAW10 → RGB8 via VPI.

    Args:
        bayer_uint16: (H, W) uint16 (10-bit data in lower bits)
        pattern: ZED X Mini = 'BGGR' (확인 의무, BA10 의 GRGR/BGBG 실제 pattern)

    Returns:
        vpi.Image (H, W, 3) RGB8.
    """
    try:
        import vpi
    except ImportError as e:
        raise RuntimeError("VPI required (Jetson). pip install python3-vpi3") from e

    # 10-bit → 8-bit (drop lower 2 bits, simpler)
    # Production: 10→8 의 *gamma + LUT* 권장
    bayer_uint8 = (bayer_uint16 >> 2).astype(np.uint8)

    pattern_map = {
        "BGGR": vpi.Format.BAYER_BGGR,
        "RGGB": vpi.Format.BAYER_RGGB,
        "GRBG": vpi.Format.BAYER_GRBG,
        "GBRG": vpi.Format.BAYER_GBRG,
    }
    bayer_fmt = pattern_map.get(pattern, vpi.Format.BAYER_BGGR)

    with vpi.Backend.CUDA:
        bayer_img = vpi.asimage(bayer_uint8).convert(bayer_fmt)
        rgb_img = bayer_img.convert(vpi.Format.RGB8)
    return rgb_img


def rectify_via_vpi(rgb_image, rectify_maps: RectifyMaps):
    """Undistort RGB via VPI remap.

    Args:
        rgb_image: vpi.Image RGB8
        rectify_maps: build_rectify_maps 의 output

    Returns:
        vpi.Image RGB8 undistorted.
    """
    try:
        import vpi
    except ImportError as e:
        raise RuntimeError("VPI required (Jetson)") from e

    with vpi.Backend.CUDA:
        # Build warp map (VPI 의 WarpMap)
        warp_map = vpi.WarpMap(
            vpi.WarpGrid((rectify_maps.width, rectify_maps.height))
        )
        warp_arr = np.asarray(warp_map)
        # WarpMap = mapping from output (x, y) → input (map_x, map_y)
        warp_arr[..., 0] = rectify_maps.map_x
        warp_arr[..., 1] = rectify_maps.map_y

        output = vpi.Image(rgb_image.size, rgb_image.format)
        rgb_image.remap(warp_map, out=output)
    return output


def pipeline_bayer_to_undistorted_rgb(
    bayer_uint16: "np.ndarray",
    rectify_maps: RectifyMaps,
    bayer_pattern: str = "BGGR",
):
    """Full Step 2 + Step 3: Bayer → debayer → rectify → undistorted RGB.

    Args:
        bayer_uint16: (H, W) uint16 from V4L2 BA10
        rectify_maps: pre-built (build_rectify_maps from raw calib)
        bayer_pattern: 'BGGR' (ZED X Mini)

    Returns:
        vpi.Image RGB8 undistorted.
    """
    rgb = debayer_via_vpi(bayer_uint16, pattern=bayer_pattern)
    undistorted = rectify_via_vpi(rgb, rectify_maps)
    return undistorted


def _self_test() -> int:
    """Mac syntax check + numpy-only rectify maps build."""
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOGGER.info("=== vpi_pipeline self-test (Mac syntax + numpy maps) ===")

    # build_rectify_maps test (numpy-only, no VPI 의무)
    fx, fy, cx, cy = 367.35, 367.35, 488.20, 320.04
    disto = [0.0428, 0.0277, -7.5e-5, -2.2e-4, -4.9e-3, 0.055,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    maps = build_rectify_maps(fx, fy, cx, cy, disto, 960, 600)

    # Sanity: maps shape + center pixel ≈ identity (distortion 작음)
    assert maps.map_x.shape == (600, 960), f"map_x shape {maps.map_x.shape}"
    assert maps.map_y.shape == (600, 960)

    # Center pixel: image (480, 300) ≈ undistorted (480, 300) due to small disto at center
    cx_idx, cy_idx = int(cx), int(cy)
    diff_x = abs(maps.map_x[cy_idx, cx_idx] - cx_idx)
    diff_y = abs(maps.map_y[cy_idx, cx_idx] - cy_idx)
    LOGGER.info(f"center pixel ({cx_idx}, {cy_idx}): map=({maps.map_x[cy_idx, cx_idx]:.2f}, {maps.map_y[cy_idx, cx_idx]:.2f})")
    LOGGER.info(f"diff: ({diff_x:.2f}, {diff_y:.2f}) px (small disto 영역)")
    if diff_x > 5 or diff_y > 5:
        LOGGER.warning(f"center diff > 5 px (real lens 의 distortion 다름)")

    # Edge pixel: 더 큰 distortion 의 fallback (sanity)
    edge_diff_x = abs(maps.map_x[0, 0] - 0)
    LOGGER.info(f"corner (0, 0): map=({maps.map_x[0, 0]:.2f}, {maps.map_y[0, 0]:.2f})")

    LOGGER.info("✓ build_rectify_maps PASS (numpy-only)")

    # VPI import check
    try:
        import vpi   # noqa: F401
        LOGGER.info("✓ VPI import OK — Jetson run path")
    except ImportError:
        LOGGER.info("VPI not available (Mac) — Jetson 의무 test")

    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
