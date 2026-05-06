"""SHM publisher with variable keypoint count + seqlock.

Binary layout (little-endian, all fields):

  [ 0:   4]  uint32   seq             (even = stable, odd = write in progress)
  [ 4:   8]  uint32   version         (= 1)
  [ 8:  12]  uint32   num_keypoints   (K)
  [12:  16]  uint32   frame_id
  [16:  24]  uint64   ts_ns           (★ semantics below)
  [24:  28]  float32  box_conf
  [28:  32]  uint32   valid_flag      (0/1)
  [32:  36]  float32  depth_invalid_ratio
  [36:  37]  uint8    world_frame_applied  (0=camera frame, 1=world frame via IMU/pitch)
  [37:  40]  reserved (3 bytes)
  [40:  48]  uint64   publish_done_mono_ns  (★ P1: 2026-05-06)
  [48:  49]  uint8    valid_reason          (★ P1: enum VALID_REASON_*)
  [49:  50]  uint8    ts_domain             (★ P1: 0=CLOCK_REALTIME / epoch_ns)
  [50:  64]  padding (aligned to 64B for cache line)
  [64:  64+K*12]       float32×K×3   kpts_3d_m
  [ . :  . +K*4]       float32×K     kpt_conf
  [ . :  . +K*8]       float32×K×2   kpts_2d_px
  [end, aligned 64B]

ts_ns semantics
---------------
``ts_ns`` is **ZED SDK ``get_timestamp(TIME_REFERENCE.IMAGE)``**. Per
Stereolabs SDK 5.x docs, this corresponds to the GMSL2 deserializer
buffer "fully available" point — *not* photon arrival time, *not* grab
call return time. Clock domain is **CLOCK_REALTIME (epoch nanoseconds)**.
``ts_domain`` field encodes this (0 = epoch).

publish_done_mono_ns semantics (P1, 2026-05-06)
-----------------------------------------------
Monotonic-ns timestamp captured *just before* the seqlock write closes
(seq → even). C++ reader can compute:
    publish_to_read_gap_ns = clock_gettime(MONOTONIC) - publish_done_mono_ns
to detect Python publish → C++ read scheduling delay independently of
wall-clock skew.

valid_reason semantics (P1, 2026-05-06)
---------------------------------------
When ``valid_flag = 0`` (publish marked invalid), ``valid_reason``
encodes *why* — see VALID_REASON_* enum. When ``valid_flag = 1``,
``valid_reason = VALID_OK = 0``.

Forward compatibility
---------------------
``version`` stays at 1. Older C++ readers that don't know about
``publish_done_mono_ns`` / ``valid_reason`` / ``ts_domain`` simply
skip those bytes — no breaking change. New readers can detect
support by reading any non-zero value in the new fields after the
first publish completes.

Total size is computed per schema; callers pass ``num_keypoints`` to
match the perception pipeline. The control-loop side reads the layout
from the header (num_keypoints field) and never assumes a fixed K.

Namespace: ``/hwalker_pose_cuda``. Mainline's ``/hwalker_pose`` MUST NOT
be written by this module.
"""

from __future__ import annotations

import logging
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


VERSION = 1
HEADER_SIZE = 64  # aligned to a typical cache line

# header field offsets
SEQ_OFF = 0
VERSION_OFF = 4
K_OFF = 8
FRAME_ID_OFF = 12
TS_OFF = 16
BOX_CONF_OFF = 24
VALID_OFF = 28
DEPTH_INVALID_OFF = 32
WORLD_FRAME_OFF = 36   # uint8: 1 = IMU world frame applied, 0 = camera frame
# 37..39 reserved (3 bytes)
PUBLISH_DONE_OFF = 40   # uint64: monotonic_ns at seqlock close (P1, 2026-05-06)
VALID_REASON_OFF = 48   # uint8: VALID_REASON_* enum (P1, 2026-05-06)
TS_DOMAIN_OFF = 49      # uint8: 0 = epoch (CLOCK_REALTIME) (P1, 2026-05-06)
# 50..63 padding


# valid_reason enum — P1, 2026-05-06
# When valid_flag = 1 → valid_reason = VALID_OK (0).
# When valid_flag = 0 → valid_reason indicates which gate rejected the frame.
VALID_OK = 0                  # all gates passed
INVALID_NO_DETECTION = 1      # box_conf below threshold (no person detected)
INVALID_OCCLUDED = 2          # too many keypoints below conf threshold
INVALID_BUDGET_EXCEED = 3     # pre_publish_e2e_ms exceeded HARD LIMIT
INVALID_CONSTRAINT = 4        # bone-length or velocity constraint rejected
INVALID_WARMUP = 5            # within warmup window (publish suppressed)
INVALID_UNKNOWN = 255         # fallback when reason cannot be determined

VALID_REASON_NAMES = {
    VALID_OK: "OK",
    INVALID_NO_DETECTION: "NO_DETECTION",
    INVALID_OCCLUDED: "OCCLUDED",
    INVALID_BUDGET_EXCEED: "BUDGET_EXCEED",
    INVALID_CONSTRAINT: "CONSTRAINT",
    INVALID_WARMUP: "WARMUP",
    INVALID_UNKNOWN: "UNKNOWN",
}

# ts_domain enum
TS_DOMAIN_EPOCH = 0       # CLOCK_REALTIME (matches time.time_ns())
TS_DOMAIN_MONOTONIC = 1   # CLOCK_MONOTONIC (matches time.monotonic_ns())


DEFAULT_NAME = "hwalker_pose_cuda"
FORBIDDEN_NAMES = {"hwalker_pose"}   # mainline collision guard


def compute_size(num_keypoints: int) -> int:
    """Total SHM segment size for a given K."""
    payload = num_keypoints * (3 + 1 + 2) * 4  # (K,3) + (K,) + (K,2) float32
    total = HEADER_SIZE + payload
    # round up to 64B multiple for alignment
    return ((total + 63) // 64) * 64


class ShmPublisher:
    def __init__(
        self,
        num_keypoints: int,
        name: str = DEFAULT_NAME,
        create: bool = True,
    ) -> None:
        if name in FORBIDDEN_NAMES:
            raise ValueError(
                f"SHM name '{name}' collides with mainline — choose another"
            )
        if num_keypoints < 1 or num_keypoints > 64:
            raise ValueError(f"num_keypoints must be 1..64, got {num_keypoints}")
        self.name = name
        self.K = num_keypoints
        self.size = compute_size(num_keypoints)

        # per-K offsets
        self.kpts_3d_off = HEADER_SIZE
        self.kpts_3d_bytes = num_keypoints * 3 * 4
        self.kpt_conf_off = self.kpts_3d_off + self.kpts_3d_bytes
        self.kpt_conf_bytes = num_keypoints * 4
        self.kpts_2d_off = self.kpt_conf_off + self.kpt_conf_bytes
        self.kpts_2d_bytes = num_keypoints * 2 * 4

        was_reattached = False
        try:
            self.shm = shared_memory.SharedMemory(
                name=name, create=create, size=self.size
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            if self.shm.size < self.size:
                raise RuntimeError(
                    f"existing SHM '{name}' size {self.shm.size} < {self.size} "
                    f"(K={num_keypoints}) — remove /dev/shm/{name} first"
                )
            was_reattached = True

        self._buf = self.shm.buf
        if not was_reattached:
            self._buf[: self.size] = bytes(self.size)
        else:
            # keep existing seq parity so live readers don't see spurious 0;
            # force valid=0 until the next real publish lands.
            struct.pack_into("<I", self._buf, VALID_OFF, 0)
        # always stamp version + K so readers can trust the header
        struct.pack_into("<I", self._buf, VERSION_OFF, VERSION)
        struct.pack_into("<I", self._buf, K_OFF, num_keypoints)

    # ------------------------------------------------------------------
    def publish(
        self,
        frame_id: int,
        ts_ns: int,
        kpts_3d_m: np.ndarray,      # (K, 3) float32
        kpt_conf: np.ndarray,       # (K,) float32
        kpts_2d_px: np.ndarray,     # (K, 2) float32
        box_conf: float,
        valid: bool,
        depth_invalid_ratio: float = 0.0,
        world_frame_applied: bool = False,
        valid_reason: int = VALID_OK,   # P1, 2026-05-06: see VALID_REASON_* enum
    ) -> None:
        K = self.K
        if kpts_3d_m.shape != (K, 3) or kpts_3d_m.dtype != np.float32:
            raise ValueError(f"kpts_3d_m must be ({K},3) float32")
        if kpt_conf.shape != (K,) or kpt_conf.dtype != np.float32:
            raise ValueError(f"kpt_conf must be ({K},) float32")
        if kpts_2d_px.shape != (K, 2) or kpts_2d_px.dtype != np.float32:
            raise ValueError(f"kpts_2d_px must be ({K},2) float32")

        buf = self._buf
        seq = struct.unpack_from("<I", buf, SEQ_OFF)[0]
        seq_write = seq + 1 if (seq & 1) == 0 else seq + 2
        struct.pack_into("<I", buf, SEQ_OFF, seq_write)

        struct.pack_into("<I", buf, FRAME_ID_OFF, frame_id & 0xFFFFFFFF)
        struct.pack_into("<Q", buf, TS_OFF, ts_ns & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<f", buf, BOX_CONF_OFF, float(box_conf))
        struct.pack_into("<I", buf, VALID_OFF, 1 if valid else 0)
        struct.pack_into("<f", buf, DEPTH_INVALID_OFF, float(depth_invalid_ratio))
        struct.pack_into("<B", buf, WORLD_FRAME_OFF, 1 if world_frame_applied else 0)

        buf[self.kpts_3d_off : self.kpts_3d_off + self.kpts_3d_bytes] = kpts_3d_m.tobytes()
        buf[self.kpt_conf_off : self.kpt_conf_off + self.kpt_conf_bytes] = kpt_conf.tobytes()
        buf[self.kpts_2d_off : self.kpts_2d_off + self.kpts_2d_bytes] = kpts_2d_px.tobytes()

        # P1 (2026-05-06): publish_done_mono_ns + valid_reason + ts_domain.
        # Captured *just before* seqlock close so the reader sees a stable
        # snapshot. valid_reason is forced to VALID_OK when valid=True
        # regardless of caller intent (consistency invariant).
        if valid:
            valid_reason = VALID_OK
        publish_done_mono_ns = time.monotonic_ns()
        struct.pack_into("<Q", buf, PUBLISH_DONE_OFF, publish_done_mono_ns & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<B", buf, VALID_REASON_OFF, valid_reason & 0xFF)
        struct.pack_into("<B", buf, TS_DOMAIN_OFF, TS_DOMAIN_EPOCH)

        struct.pack_into("<I", buf, SEQ_OFF, seq_write + 1)

    def close(self) -> None:
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as err:  # pragma: no cover
            LOGGER.warning("SHM close error: %s", err)


class ShmReader:
    """Reads the current snapshot with seqlock retry.

    Pass ``expected_k`` to fail loudly if the publisher's schema
    (typically chosen by CLI ``--schema``) doesn't match the consumer
    contract. Without this, a mainline consumer hard-coded to K=17
    would silently misread a K=6 segment.
    """

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        expected_k: Optional[int] = None,
    ) -> None:
        self.shm = shared_memory.SharedMemory(name=name, create=False)
        self._buf = self.shm.buf
        version = struct.unpack_from("<I", self._buf, VERSION_OFF)[0]
        if version != VERSION:
            raise RuntimeError(
                f"SHM version mismatch: segment={version}, reader={VERSION}"
            )
        self.K = struct.unpack_from("<I", self._buf, K_OFF)[0]
        if expected_k is not None and expected_k != self.K:
            raise RuntimeError(
                f"SHM keypoint count mismatch: segment K={self.K}, "
                f"expected K={expected_k}. Rebuild publisher+consumer "
                "with matching schema."
            )
        self.kpts_3d_off = HEADER_SIZE
        self.kpt_conf_off = self.kpts_3d_off + self.K * 3 * 4
        self.kpts_2d_off = self.kpt_conf_off + self.K * 4

    def read(self, max_retries: int = 16) -> Optional[Tuple]:
        buf = self._buf
        K = self.K
        for _ in range(max_retries):
            seq0 = struct.unpack_from("<I", buf, SEQ_OFF)[0]
            if seq0 & 1:
                continue
            frame_id = struct.unpack_from("<I", buf, FRAME_ID_OFF)[0]
            ts_ns = struct.unpack_from("<Q", buf, TS_OFF)[0]
            box_conf = struct.unpack_from("<f", buf, BOX_CONF_OFF)[0]
            valid = struct.unpack_from("<I", buf, VALID_OFF)[0] != 0
            depth_inv = struct.unpack_from("<f", buf, DEPTH_INVALID_OFF)[0]
            kpts_3d = np.frombuffer(
                buf, dtype=np.float32, count=K * 3, offset=self.kpts_3d_off
            ).reshape(K, 3).copy()
            kpt_conf = np.frombuffer(
                buf, dtype=np.float32, count=K, offset=self.kpt_conf_off
            ).copy()
            kpts_2d = np.frombuffer(
                buf, dtype=np.float32, count=K * 2, offset=self.kpts_2d_off
            ).reshape(K, 2).copy()
            world_frame_applied = struct.unpack_from("<B", buf, WORLD_FRAME_OFF)[0] != 0
            # P1 (2026-05-06): new fields. Older publishers wrote 0 here,
            # so 0 means "publish_done not recorded" (caller decides handling).
            publish_done_mono_ns = struct.unpack_from("<Q", buf, PUBLISH_DONE_OFF)[0]
            valid_reason = struct.unpack_from("<B", buf, VALID_REASON_OFF)[0]
            ts_domain = struct.unpack_from("<B", buf, TS_DOMAIN_OFF)[0]
            seq1 = struct.unpack_from("<I", buf, SEQ_OFF)[0]
            if seq0 == seq1 and (seq0 & 1) == 0:
                return (frame_id, ts_ns, kpts_3d, kpt_conf, kpts_2d,
                        box_conf, valid, depth_inv, world_frame_applied,
                        publish_done_mono_ns, valid_reason, ts_domain)
        return None

    def close(self) -> None:
        self.shm.close()
