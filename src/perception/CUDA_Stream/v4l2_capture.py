"""V4L2 raw bayer capture — ZED X Mini bypass path (Linux only).

⚠️ ARCHIVED (2026-05-12 사용자 결정 C) ⚠️

Jetson run #2 (commit dbbc83a) 결과:
    struct size 정확 (204/88/20) + IOCTL number 정확 (0xC0CC5605)
    단 tegra-capture-vi driver 가 VIDIOC_S_FMT 거부 (Tegra-level quirk)
    G_FMT (read-only) PASS, S_FMT (set) ENOTTY

진정 path = C++ libargus direct (수개월 effort, Python 만으론 어려움).
사용자 결정 = V4L2 우회 abandon, Plan D EKF 집중 (control repo, phase-locked 예측 -50ms).

코드 유지 이유:
    1. 학습 (V4L2 ABI, ctypes struct alignment, Tegra quirk 의 진정 발견)
    2. Future C++ libargus prototype 의 reference (struct definitions)
    3. Paper의 "investigated but abandoned" engineering decision 의 정직 기록

원본 docstring:
    사용자 의지 (정확 + 속도): 4-9주 effort 진행.
    docs/lessons/v4l2_bypass_plan.md Step 1.

    Format (2026-05-12 검증):
        /dev/video0,1 = 'BA10' (10-bit Bayer GRGR/BGBG)
        960×600 @ 120fps (우리 SVGA path)

    Implementation:
        - Python ctypes + fcntl + mmap (stdlib only, no v4l2py 의존)
        - Direct V4L2 IOCTL (open → set format → request buffers → mmap → stream)
        - 10-bit Bayer unpack (BA10 = 5 bytes / 4 pixels)
        - L/R sync via frame timestamps

⚠️ Linux only. Mac 에선 syntax check 만. Jetson 의무 test (ARCHIVED).
"""
from __future__ import annotations

import ctypes
import fcntl
import logging
import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ─── V4L2 constants (linux/videodev2.h) ──────────────────────────────────

V4L2_PIX_FMT_SRGGB10 = 0x30314752    # 'BA10' (10-bit Bayer)
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FIELD_NONE = 1
V4L2_MEMORY_MMAP = 1

# IOCTL codes (computed from linux/videodev2.h)
# _IORW for query types, _IOW for set, _IO for stream control
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(d, t, nr, size):
    return (d << _IOC_DIRSHIFT) | (t << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)


def _IOR(t, nr, size):
    return _IOC(_IOC_READ, t, nr, size)


def _IOW(t, nr, size):
    return _IOC(_IOC_WRITE, t, nr, size)


def _IOWR(t, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, t, nr, size)


# ─── V4L2 struct (linux/videodev2.h) ─────────────────────────────────────

class V4L2Format(ctypes.Structure):
    """struct v4l2_format with v4l2_pix_format union member.

    ★ Codex Jetson 2026-05-12 발견: ctypes default alignment ≠ kernel packed.
    _pack_ = 1 명시 + _padding 정확 사이즈 (200 - 48 = 152, 이전 156 = wrong).

    Layout (linux/videodev2.h):
        struct v4l2_format {
            __u32 type;             // 4 bytes
            union { ... } fmt;      // 200 bytes max (v4l2_pix_format = 48 + padding 152)
        };
    """
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_uint32),
        # fmt.pix (v4l2_pix_format) — 48 bytes
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
        # fmt union 의 나머지 padding (200 - 48 = 152)
        ("_padding", ctypes.c_uint8 * 152),
    ]


class V4L2RequestBuffers(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class V4L2Buffer(ctypes.Structure):
    """struct v4l2_buffer.

    ★ Codex Jetson 2026-05-12 발견: m union = 8 bytes (pointer 또는 offset+padding).
    _pack_ = 1 + m_offset/m_pad 의무.
    """
    class Timestamp(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("tv_sec", ctypes.c_long),
            ("tv_usec", ctypes.c_long),
        ]

    class Timecode(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("type", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("frames", ctypes.c_uint8),
            ("seconds", ctypes.c_uint8),
            ("minutes", ctypes.c_uint8),
            ("hours", ctypes.c_uint8),
            ("userbits", ctypes.c_uint8 * 4),
        ]

    _pack_ = 1
    _fields_ = [
        ("index", ctypes.c_uint32),         # offset 0
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),         # offset 16
        ("timestamp", Timestamp),            # 16 bytes (offset 20-36, 64-bit timeval)
        ("timecode", Timecode),              # 16 bytes (offset 36-52)
        ("sequence", ctypes.c_uint32),       # offset 52
        ("memory", ctypes.c_uint32),         # offset 56
        ("_pre_m_pad", ctypes.c_uint32),    # ★ 8-byte alignment of m union (60-64)
        # m union — 8 bytes (offset uint32 OR userptr uint64 OR planes pointer)
        ("offset", ctypes.c_uint32),         # offset 64 (m.offset OR m.userptr low)
        ("m_pad", ctypes.c_uint32),          # offset 68 (m.userptr high if pointer)
        ("length", ctypes.c_uint32),         # offset 72
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),      # v4l2 Linux >= 4.20 추가
        ("_tail_pad", ctypes.c_uint32),     # ★ struct 8-byte alignment (84-88)
    ]


# IOCTL codes
VIDIOC_S_FMT = _IOWR(ord('V'), 5, ctypes.sizeof(V4L2Format))
VIDIOC_REQBUFS = _IOWR(ord('V'), 8, ctypes.sizeof(V4L2RequestBuffers))
VIDIOC_QUERYBUF = _IOWR(ord('V'), 9, ctypes.sizeof(V4L2Buffer))
VIDIOC_QBUF = _IOWR(ord('V'), 15, ctypes.sizeof(V4L2Buffer))
VIDIOC_DQBUF = _IOWR(ord('V'), 17, ctypes.sizeof(V4L2Buffer))
VIDIOC_STREAMON = _IOW(ord('V'), 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _IOW(ord('V'), 19, ctypes.sizeof(ctypes.c_int))


# ─── High-level interface ────────────────────────────────────────────────

@dataclass
class V4L2CaptureHandle:
    """Open + streaming V4L2 device handle."""
    fd: int
    width: int
    height: int
    pixelformat: int
    num_buffers: int
    buffers: List[mmap.mmap]
    buffer_lengths: List[int]
    device_path: str


def open_v4l2_bayer_capture(
    device: str = "/dev/video0",
    width: int = 960,
    height: int = 600,
    num_buffers: int = 4,
) -> V4L2CaptureHandle:
    """Open V4L2 device + set Bayer RAW10 format + mmap + stream on.

    Args:
        device: /dev/video0 (left) or /dev/video1 (right)
        width / height: 960×600 (SVGA path) — must match v4l2-ctl 의 supported
        num_buffers: ring buffer size (4 = balanced)

    Returns:
        V4L2CaptureHandle (capture_frame 의 input).

    Raises:
        RuntimeError on V4L2 fail. Caller 의 try/finally + close_v4l2.
    """
    fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    try:
        # 1. Set format (Bayer RAW10)
        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.width = width
        fmt.height = height
        fmt.pixelformat = V4L2_PIX_FMT_SRGGB10
        fmt.field = V4L2_FIELD_NONE
        fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)

        actual_pix = fmt.pixelformat
        if actual_pix != V4L2_PIX_FMT_SRGGB10:
            raise RuntimeError(
                f"V4L2 set format failed: requested 0x{V4L2_PIX_FMT_SRGGB10:08x} "
                f"(BA10), got 0x{actual_pix:08x}"
            )

        # 2. Request buffers
        req = V4L2RequestBuffers()
        req.count = num_buffers
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
        if req.count < num_buffers:
            raise RuntimeError(
                f"V4L2 REQBUFS: requested {num_buffers}, got {req.count}"
            )

        # 3. mmap + queue each buffer
        buffers = []
        lengths = []
        for i in range(req.count):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = i
            fcntl.ioctl(fd, VIDIOC_QUERYBUF, buf)
            mapped = mmap.mmap(fd, buf.length, mmap.MAP_SHARED,
                                mmap.PROT_READ | mmap.PROT_WRITE,
                                offset=buf.offset)
            buffers.append(mapped)
            lengths.append(buf.length)
            # Initial queue
            fcntl.ioctl(fd, VIDIOC_QBUF, buf)

        # 4. Stream on
        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(fd, VIDIOC_STREAMON, buf_type)

        LOGGER.info(
            "V4L2 capture opened: %s %dx%d, %d buffers (BA10 Bayer RAW10)",
            device, width, height, req.count,
        )
        return V4L2CaptureHandle(
            fd=fd, width=width, height=height,
            pixelformat=V4L2_PIX_FMT_SRGGB10,
            num_buffers=req.count, buffers=buffers,
            buffer_lengths=lengths, device_path=device,
        )
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def capture_frame_bayer(handle: V4L2CaptureHandle) -> Tuple[np.ndarray, int]:
    """Dequeue → bayer raw + ts.

    Returns:
        (bayer_raw, ts_ns):
            bayer_raw: (H, W) uint16 unpacked from BA10
            ts_ns: V4L2 buffer timestamp (CLOCK_MONOTONIC ns)

    Raises:
        BlockingIOError if no buffer ready (non-blocking).
    """
    buf = V4L2Buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_MMAP
    fcntl.ioctl(handle.fd, VIDIOC_DQBUF, buf)

    idx = buf.index
    bytesused = buf.bytesused
    ts_ns = buf.timestamp.tv_sec * 1_000_000_000 + buf.timestamp.tv_usec * 1_000

    raw_bytes = bytes(handle.buffers[idx][:bytesused])
    bayer_raw = _unpack_ba10(raw_bytes, handle.width, handle.height)

    # Re-queue buffer
    fcntl.ioctl(handle.fd, VIDIOC_QBUF, buf)

    return bayer_raw, ts_ns


def _unpack_ba10(raw_bytes: bytes, width: int, height: int) -> np.ndarray:
    """BA10 (10-bit packed Bayer) → uint16 (H, W).

    BA10 layout: 5 bytes for 4 pixels (40 bits = 4 × 10 bits).
    Each row's bytes-per-line should be 5 × width / 4 (or 4 × width if 4-byte align).

    Simplified: each 10-bit sample = upper byte's 8 bits + lower 2 bits from packed byte.
    실제 NVIDIA Tegra 의 BA10 = *16-bit aligned* (10 bits in lower, upper 6 zeros).
    그러므로 stride = 2 × width.
    """
    # Tegra V4L2 BA10 (2-byte aligned) — uint16 little-endian, 10 bits in lower
    arr = np.frombuffer(raw_bytes, dtype=np.uint16)
    expected_pixels = width * height
    if arr.size < expected_pixels:
        raise RuntimeError(
            f"_unpack_ba10: buffer has {arr.size} pixels, expected {expected_pixels}"
        )
    bayer = arr[:expected_pixels].reshape(height, width)
    # Mask to 10 bits (upper 6 should be 0)
    bayer = bayer & 0x03FF
    return bayer


def close_v4l2(handle: V4L2CaptureHandle) -> None:
    """Stream off + unmap + close fd."""
    try:
        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(handle.fd, VIDIOC_STREAMOFF, buf_type)
    except OSError as e:
        LOGGER.warning("STREAMOFF fail: %s", e)
    for mm in handle.buffers:
        try:
            mm.close()
        except Exception:
            pass
    try:
        os.close(handle.fd)
    except OSError:
        pass


# ─── Stereo capture (synchronized L + R) ─────────────────────────────────

@dataclass
class StereoFrame:
    left_bayer: np.ndarray      # (H, W) uint16 BA10
    right_bayer: np.ndarray
    left_ts_ns: int
    right_ts_ns: int
    sync_diff_us: float          # |L_ts - R_ts| in microseconds


def open_stereo_capture(
    left_device: str = "/dev/video0",
    right_device: str = "/dev/video1",
    width: int = 960, height: int = 600,
) -> Tuple[V4L2CaptureHandle, V4L2CaptureHandle]:
    """Open L + R V4L2 captures."""
    left = open_v4l2_bayer_capture(left_device, width, height)
    try:
        right = open_v4l2_bayer_capture(right_device, width, height)
    except Exception:
        close_v4l2(left)
        raise
    return left, right


def capture_stereo_frame(
    left_handle: V4L2CaptureHandle,
    right_handle: V4L2CaptureHandle,
    max_sync_diff_us: float = 1000.0,
) -> Optional[StereoFrame]:
    """Capture L + R frame with sw sync check.

    Returns:
        StereoFrame or None if sync diff > max_sync_diff_us.
    """
    left_bayer, left_ts = capture_frame_bayer(left_handle)
    right_bayer, right_ts = capture_frame_bayer(right_handle)
    diff_us = abs(left_ts - right_ts) / 1000.0
    if diff_us > max_sync_diff_us:
        LOGGER.warning("L/R sync diff %.1f us > %.1f", diff_us, max_sync_diff_us)
        return None
    return StereoFrame(
        left_bayer=left_bayer, right_bayer=right_bayer,
        left_ts_ns=left_ts, right_ts_ns=right_ts,
        sync_diff_us=diff_us,
    )


def main() -> int:
    """CLI smoke test (Jetson only)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--n-frames", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    handle = open_v4l2_bayer_capture(args.device, args.width, args.height)
    try:
        captured = 0
        t_start = time.time()
        while captured < args.n_frames and time.time() - t_start < 10:
            try:
                bayer, ts = capture_frame_bayer(handle)
                captured += 1
                if captured <= 3 or captured % 10 == 0:
                    LOGGER.info("frame %d: shape=%s, ts=%d, max=%d, min=%d",
                                captured, bayer.shape, ts, bayer.max(), bayer.min())
            except BlockingIOError:
                time.sleep(0.001)
        elapsed = time.time() - t_start
        LOGGER.info("captured %d frames in %.2fs (%.1f fps)",
                    captured, elapsed, captured / elapsed)
    finally:
        close_v4l2(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
