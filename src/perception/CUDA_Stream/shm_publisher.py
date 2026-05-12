"""SHM publisher v2 — Plan D EKF input contract.

작성: 2026-05-11. Codex orchestration consult `bvfvkxo1m` (token 920K).
Spec: docs/lessons/shm_v2_packet_spec.md.

v2 의 핵심 변경 (vs v1):
  1. Two timestamps: rgb_ts_ns (T_N) + depth_ts_ns (T_{N-1} if 1-frame-late)
     + depth_age_us = (rgb_ts - depth_ts) / 1000
  2. Per-keypoint valid_mask_bits (uint8[8], K up to 64)
  3. Per-keypoint covariance: kp_sigma_m[K][3], pose_cov_diag[K][3]
     EKF 의 measurement noise R 의 source
  4. version = 2 (v1 reader 는 fail-fast — clean break, 사용자 control repo
     의 reader 도 v2 작성 필요)

Binary layout (little-endian, all fields):

  Header (64 bytes — cache line aligned):
  [ 0:  4]  uint32   seq                  (even=stable, odd=write in progress)
  [ 4:  8]  uint32   version = 2
  [ 8: 12]  uint32   num_keypoints (K)
  [12: 16]  uint32   frame_id
  [16: 24]  uint64   rgb_ts_ns            ★ ZED IMAGE timestamp (T_N)
  [24: 32]  uint64   depth_ts_ns          ★ depth retrieve 시각 (T_{N-1} if 1-frame-late)
  [32: 36]  uint32   depth_age_us         ★ (rgb_ts - depth_ts) / 1000
  [36: 40]  float32  box_conf
  [40: 44]  float32  depth_invalid_ratio
  [44: 45]  uint8    valid_flag           (derived from valid_mask_bits, deprecated)
  [45: 46]  uint8    world_frame          (0=camera, 1=world)
  [46: 47]  uint8    valid_reason         (VALID_REASON_* enum)
  [47: 48]  uint8    ts_domain            (0=CLOCK_REALTIME)
  [48: 56]  uint64   publish_done_mono_ns
  [56: 64]  uint8[8] valid_mask_bits      ★ per-kp validity (K up to 64)

  Body (variable, K-dependent):
  [64 :    + K*12]  float32[K][3]   kpts_3d_m
  [.. :    + K*8]   float32[K][2]   kpts_2d_px
  [.. :    + K*4]   float32[K]      kp_conf
  [.. :    + K*12]  float32[K][3]   kp_sigma_m       ★ depth uncertainty (m)
  [.. :    + K*12]  float32[K][3]   pose_cov_diag    ★ pose covariance diag

  Total size = HEADER_SIZE + K * (3+2+1+3+3) * 4 = 64 + K*48
  K=6 → 352 → 384 (64B aligned)

⚠️ Memory ordering (Codex review b1ky3965z P1-5):
  ARM Orin (weak memory model) 의 cross-process race 가능성:
    Writer (Python): struct.pack_into = plain stores. release fence 없음.
    Reader (C++):    atomic_load on seq, plain reads on body.
  → Reader 가 *seq close (even)* 후 *body 가 globally visible 전* 의 read 가능.
  → 단 *seqlock retry* (max 16 retries) + *typical write 짧음 (~10us)* 가
     real-world 에서 stable. Production 시 100k+ frame 에서 race 안 관찰됨.
  → Patient experiment 직전 의무 fix:
      - Writer: struct.pack_into 후 *명시 release fence* (예: C++ binding 의 atomic_thread_fence)
      - 또는: 전체 publish() 를 C++ 으로 binding (gil_scoped_release + 명시 ordering)
  → 현재 status: *production OK*, *clinical 직전 audit 권장*.

Timestamp semantics:
  rgb_ts_ns         = ZED get_timestamp(TIME_REFERENCE.IMAGE) — GMSL2 deserializer 시점
                       (★ photon arrival 아님, Stereolabs docs).
  depth_ts_ns       = depth retrieve 의 frame 시점. 동일 frame 시 == rgb_ts_ns.
                       one-frame-late path 시 == rgb_ts - 1/fps (~ 8.3ms @ 120fps).
  depth_age_us      = 0 if 동일 frame, ~8333 if 1-frame-late, > 16700 → invalid.

Validity:
  valid_mask_bits   = per-kp uint8[8]. bit i = 1 → keypoint i 정상.
                       valid_flag (derived) = 1 if any bit set, 0 otherwise.
  valid_reason      = publish-level (전체 frame 의 fallback reason).

Forward / backward compatibility:
  v2 reader + v1 packet → version=1 → use legacy fields, default uncertainty.
  v1 reader + v2 packet → version=2 → unsupported, **safe fallback** (publish 무시).
  v2 reader + v2 packet → full Plan D path.

Namespace: `/hwalker_pose_cuda`. `/hwalker_pose` 는 mainline 영역 — 절대 X.
"""

from __future__ import annotations

import logging
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


VERSION = 2
HEADER_SIZE = 64

# header field offsets
SEQ_OFF = 0
VERSION_OFF = 4
K_OFF = 8
FRAME_ID_OFF = 12
RGB_TS_OFF = 16              # uint64 (★ v2)
DEPTH_TS_OFF = 24            # uint64 (★ v2)
DEPTH_AGE_OFF = 32           # uint32 microseconds (★ v2)
BOX_CONF_OFF = 36
DEPTH_INVALID_OFF = 40
VALID_FLAG_OFF = 44          # uint8 (deprecated, derived from valid_mask_bits)
WORLD_FRAME_OFF = 45         # uint8
VALID_REASON_OFF = 46        # uint8
TS_DOMAIN_OFF = 47           # uint8
PUBLISH_DONE_OFF = 48        # uint64
VALID_MASK_BITS_OFF = 56     # uint8[8] per-kp validity (★ v2)


# valid_reason enum
VALID_OK = 0
INVALID_NO_DETECTION = 1     # box_conf below threshold
INVALID_OCCLUDED = 2         # too many keypoints below conf threshold
INVALID_BUDGET_EXCEED = 3    # pre_publish_e2e_ms exceeded HARD LIMIT
INVALID_CONSTRAINT = 4       # bone-length / velocity constraint rejected
INVALID_WARMUP = 5           # within warmup window
INVALID_STALE_DEPTH = 6      # depth_age_us > 2 frames (★ v2)
INVALID_DRIFT = 7            # self-calibration drift detected (★ v2 future)
INVALID_THERMAL = 8          # GPU/CPU throttling (★ v2 future)
INVALID_UNKNOWN = 255

VALID_REASON_NAMES = {
    VALID_OK: "OK",
    INVALID_NO_DETECTION: "NO_DETECTION",
    INVALID_OCCLUDED: "OCCLUDED",
    INVALID_BUDGET_EXCEED: "BUDGET_EXCEED",
    INVALID_CONSTRAINT: "CONSTRAINT",
    INVALID_WARMUP: "WARMUP",
    INVALID_STALE_DEPTH: "STALE_DEPTH",
    INVALID_DRIFT: "DRIFT",
    INVALID_THERMAL: "THERMAL",
    INVALID_UNKNOWN: "UNKNOWN",
}

# ts_domain enum
TS_DOMAIN_EPOCH = 0          # CLOCK_REALTIME (matches time.time_ns())
TS_DOMAIN_MONOTONIC = 1      # CLOCK_MONOTONIC


DEFAULT_NAME = "hwalker_pose_cuda"
FORBIDDEN_NAMES = {"hwalker_pose"}   # mainline collision guard

# Default uniform sigma when caller doesn't provide per-kp uncertainty.
# Used only if Plan D EKF measurement R 추정 안 되면 fallback.
# ⚠️ Codex review b1ky3965z (P2-4): production 시 *진정 추정* 필요 (depth/conf/age 의존).
DEFAULT_SIGMA_M = 0.015      # 15mm — Codex Q3 의 1m@1m disparity error 추정

# ★ Codex review b1ky3965z (P1-6): stale depth invalidation threshold.
# 2 frames at 120fps = 16700us. depth_age 가 이를 초과 시 INVALID_STALE_DEPTH.
MAX_DEPTH_AGE_US = 16700


def compute_size(num_keypoints: int) -> int:
    """Total SHM segment size for a given K (v2 layout)."""
    # body: kpts_3d (K,3) + kpts_2d (K,2) + kp_conf (K,)
    #     + kp_sigma_m (K,3) + pose_cov_diag (K,3)
    # = K * (3 + 2 + 1 + 3 + 3) * 4 bytes = K * 48
    payload = num_keypoints * (3 + 2 + 1 + 3 + 3) * 4
    total = HEADER_SIZE + payload
    return ((total + 63) // 64) * 64    # 64B aligned


class ShmPublisher:
    """Plan D v2 SHM publisher.

    Backward incompatible with v1 — version field bumped to 2. v1 readers
    raise on version mismatch (clean fallback). 사용자 control repo 의
    reader 도 v2 처리 작성 필요.
    """

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

        # per-K offsets (sequential after header)
        self.kpts_3d_off = HEADER_SIZE
        self.kpts_3d_bytes = num_keypoints * 3 * 4
        self.kpts_2d_off = self.kpts_3d_off + self.kpts_3d_bytes
        self.kpts_2d_bytes = num_keypoints * 2 * 4
        self.kpt_conf_off = self.kpts_2d_off + self.kpts_2d_bytes
        self.kpt_conf_bytes = num_keypoints * 4
        self.kp_sigma_off = self.kpt_conf_off + self.kpt_conf_bytes
        self.kp_sigma_bytes = num_keypoints * 3 * 4
        self.pose_cov_off = self.kp_sigma_off + self.kp_sigma_bytes
        self.pose_cov_bytes = num_keypoints * 3 * 4

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
                    f"(K={num_keypoints}, version=2) — remove /dev/shm/{name} first"
                )
            was_reattached = True

        self._buf = self.shm.buf

        # ★ Codex review b1ky3965z (P2-3): reattach 시도 seqlock 안에서.
        # 기존: version/K stamp 가 seqlock 외부 → reader 가 stale v1 bytes 보면 problem.
        # Fix: write-in-progress 동안 (odd seq) 모든 stamp.
        cur_seq = struct.unpack_from("<I", self._buf, SEQ_OFF)[0] if was_reattached else 0
        seq_write = cur_seq + 1 if (cur_seq & 1) == 0 else cur_seq + 2
        struct.pack_into("<I", self._buf, SEQ_OFF, seq_write)   # open (odd)

        # zero out (or partial reset) under seqlock
        if not was_reattached:
            self._buf[: self.size] = bytes(self.size)
        else:
            # invalidate body + valid fields
            struct.pack_into("<B", self._buf, VALID_FLAG_OFF, 0)
            struct.pack_into("<Q", self._buf, VALID_MASK_BITS_OFF, 0)
            self._buf[HEADER_SIZE:self.size] = bytes(self.size - HEADER_SIZE)

        # stamp version + K (under seqlock)
        struct.pack_into("<I", self._buf, VERSION_OFF, VERSION)
        struct.pack_into("<I", self._buf, K_OFF, num_keypoints)

        struct.pack_into("<I", self._buf, SEQ_OFF, seq_write + 1)   # close (even)

    # ------------------------------------------------------------------
    def publish(
        self,
        frame_id: int,
        rgb_ts_ns: int,                              # ★ v2: RGB capture timestamp (T_N)
        kpts_3d_m: np.ndarray,                       # (K, 3) float32
        kpt_conf: np.ndarray,                        # (K,) float32
        kpts_2d_px: np.ndarray,                      # (K, 2) float32
        box_conf: float,
        valid: bool,
        depth_invalid_ratio: float = 0.0,
        world_frame_applied: bool = False,
        valid_reason: int = VALID_OK,
        # ★ v2 추가 fields
        depth_ts_ns: Optional[int] = None,           # None → 동일 frame (= rgb_ts_ns)
        valid_mask_bits: Optional[int] = None,       # None → derived from kp_conf + depth z
        kp_sigma_m: Optional[np.ndarray] = None,     # (K, 3) float32 m. None → default 15mm
        pose_cov_diag: Optional[np.ndarray] = None,  # (K, 3) float32 m². None → kp_sigma_m²
        kp_conf_threshold: float = 0.5,              # ★ v2: per-kp validity derive threshold
    ) -> None:
        """Publish v2 packet.

        Args:
            rgb_ts_ns: RGB frame timestamp (T_N). ZED IMAGE 또는 V4L2 timestamp.
            depth_ts_ns: depth retrieve frame timestamp. 동일 frame 시 == rgb_ts_ns.
                         one-frame-late path 시 = rgb_ts_ns - (1/fps) ~= rgb_ts - 8.3ms.
            valid_mask_bits: per-keypoint validity (uint8[8]). None 면 자동 derive.
            kp_sigma_m: per-keypoint depth uncertainty (m). None 면 default 15mm.
            pose_cov_diag: per-keypoint pose covariance diag (m²). None 면 kp_sigma_m².
        """
        K = self.K
        if kpts_3d_m.shape != (K, 3) or kpts_3d_m.dtype != np.float32:
            raise ValueError(f"kpts_3d_m must be ({K},3) float32")
        if kpt_conf.shape != (K,) or kpt_conf.dtype != np.float32:
            raise ValueError(f"kpt_conf must be ({K},) float32")
        if kpts_2d_px.shape != (K, 2) or kpts_2d_px.dtype != np.float32:
            raise ValueError(f"kpts_2d_px must be ({K},2) float32")

        # ★ Codex review b1ky3965z (P2-2): timestamp validation
        if rgb_ts_ns < 0 or rgb_ts_ns > 0x7FFFFFFFFFFFFFFF:
            raise ValueError(f"rgb_ts_ns out of range: {rgb_ts_ns}")
        if frame_id < 0 or frame_id > 0xFFFFFFFF:
            raise ValueError(f"frame_id out of u32 range: {frame_id}")

        # depth_ts_ns default = rgb_ts_ns (same frame)
        if depth_ts_ns is None:
            depth_ts_ns = rgb_ts_ns
        elif depth_ts_ns < 0 or depth_ts_ns > 0x7FFFFFFFFFFFFFFF:
            raise ValueError(f"depth_ts_ns out of range: {depth_ts_ns}")

        # ★ Codex review b1ky3965z (P1-6): depth timestamp contract enforcement.
        # depth_ts > rgb_ts (future depth) — invalidate.
        # depth_age > MAX_DEPTH_AGE_US — invalidate (stale depth).
        raw_age_ns = rgb_ts_ns - depth_ts_ns
        if raw_age_ns < 0:
            # depth from future → impossible / clock skew → invalid
            if valid:
                valid = False
                valid_reason = INVALID_UNKNOWN
            depth_age_us = 0
        else:
            depth_age_us = raw_age_ns // 1000

        # Stale depth (> 2 frames at 120fps = 16700us) → INVALID_STALE_DEPTH
        if depth_age_us > MAX_DEPTH_AGE_US and valid:
            valid = False
            valid_reason = INVALID_STALE_DEPTH

        # ★ Codex review b1ky3965z (P1-7): per-kp validity 의 진정 derive.
        # 기존: valid_mask_bits=None + valid=True → 모든 K bit 자동 set.
        #       → kp_conf 무시, occluded keypoint 도 valid 처리.
        # Fix: kp_conf >= threshold + 3D depth finite (NaN/0 reject) 의 keypoint 만 set.
        #      caller 가 명시 mask 전달 가능.
        if valid_mask_bits is None:
            if valid:
                mask = 0
                # depth z (3rd col) 의 finite + > 0 가 valid 의 의무 (CLAUDE.md guard).
                z_valid = np.isfinite(kpts_3d_m[:, 2]) & (kpts_3d_m[:, 2] > 0)
                conf_valid = kpt_conf >= kp_conf_threshold
                bit_valid = z_valid & conf_valid
                for i in range(K):
                    if bit_valid[i]:
                        mask |= (1 << i)
                valid_mask_bits = mask
            else:
                valid_mask_bits = 0

        # ★ Codex review b1ky3965z (P2-1): valid_mask_bits validation.
        # K bit 만 허용. K beyond bits = 잘못된 caller intent.
        max_mask = (1 << K) - 1
        if valid_mask_bits & ~max_mask:
            raise ValueError(
                f"valid_mask_bits={bin(valid_mask_bits)} has bits beyond K={K} "
                f"(max allowed={bin(max_mask)})"
            )

        # ★ Consistency: valid=True + mask=0 → publish 단 invalid mark.
        # (모든 keypoint occluded = frame-level invalid)
        if valid_mask_bits == 0 and valid:
            valid = False
            valid_reason = INVALID_OCCLUDED

        # ★ Codex review bzc20un44 P1-1 fix: valid=False + valid_reason=VALID_OK 모순.
        # Watchdog 등 caller 가 valid=False 단 valid_reason 미명시 시 default = VALID_OK.
        # → reader 가 invalid frame 을 OK 로 오해 가능 (clinical silent failure).
        # Fix: invariant — valid=False 시 valid_reason 가 OK 면 강제 INVALID_UNKNOWN.
        if not valid and valid_reason == VALID_OK:
            valid_reason = INVALID_UNKNOWN

        # kp_sigma_m default — uniform 15mm
        if kp_sigma_m is None:
            kp_sigma_m = np.full((K, 3), DEFAULT_SIGMA_M, dtype=np.float32)
        else:
            if kp_sigma_m.shape != (K, 3) or kp_sigma_m.dtype != np.float32:
                raise ValueError(f"kp_sigma_m must be ({K},3) float32")

        # pose_cov_diag default — kp_sigma_m²
        if pose_cov_diag is None:
            pose_cov_diag = (kp_sigma_m ** 2).astype(np.float32)
        else:
            if pose_cov_diag.shape != (K, 3) or pose_cov_diag.dtype != np.float32:
                raise ValueError(f"pose_cov_diag must be ({K},3) float32")

        buf = self._buf

        # seqlock open (odd seq = write in progress)
        seq = struct.unpack_from("<I", buf, SEQ_OFF)[0]
        seq_write = seq + 1 if (seq & 1) == 0 else seq + 2
        struct.pack_into("<I", buf, SEQ_OFF, seq_write)

        # Header
        struct.pack_into("<I", buf, FRAME_ID_OFF, frame_id & 0xFFFFFFFF)
        struct.pack_into("<Q", buf, RGB_TS_OFF, rgb_ts_ns & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<Q", buf, DEPTH_TS_OFF, depth_ts_ns & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<I", buf, DEPTH_AGE_OFF, depth_age_us & 0xFFFFFFFF)
        struct.pack_into("<f", buf, BOX_CONF_OFF, float(box_conf))
        struct.pack_into("<f", buf, DEPTH_INVALID_OFF, float(depth_invalid_ratio))

        # valid_flag derived from valid_mask_bits (any bit set = valid)
        valid_flag = 1 if valid_mask_bits != 0 and valid else 0
        struct.pack_into("<B", buf, VALID_FLAG_OFF, valid_flag)
        struct.pack_into("<B", buf, WORLD_FRAME_OFF, 1 if world_frame_applied else 0)

        # consistency: valid=True → valid_reason = VALID_OK
        if valid:
            valid_reason = VALID_OK
        struct.pack_into("<B", buf, VALID_REASON_OFF, valid_reason & 0xFF)
        struct.pack_into("<B", buf, TS_DOMAIN_OFF, TS_DOMAIN_EPOCH)

        # valid_mask_bits — uint8[8] (8 bytes, K up to 64)
        struct.pack_into("<Q", buf, VALID_MASK_BITS_OFF,
                         valid_mask_bits & 0xFFFFFFFFFFFFFFFF)

        # Body
        buf[self.kpts_3d_off : self.kpts_3d_off + self.kpts_3d_bytes] = kpts_3d_m.tobytes()
        buf[self.kpts_2d_off : self.kpts_2d_off + self.kpts_2d_bytes] = kpts_2d_px.tobytes()
        buf[self.kpt_conf_off : self.kpt_conf_off + self.kpt_conf_bytes] = kpt_conf.tobytes()
        buf[self.kp_sigma_off : self.kp_sigma_off + self.kp_sigma_bytes] = kp_sigma_m.tobytes()
        buf[self.pose_cov_off : self.pose_cov_off + self.pose_cov_bytes] = pose_cov_diag.tobytes()

        # publish_done — captured just before seqlock close (P1).
        publish_done_mono_ns = time.monotonic_ns()
        struct.pack_into("<Q", buf, PUBLISH_DONE_OFF,
                         publish_done_mono_ns & 0xFFFFFFFFFFFFFFFF)

        # seqlock close (even seq = stable)
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
    """Plan D v2 SHM reader.

    Pass `expected_k` to fail loudly if the publisher's schema doesn't
    match the consumer contract. Without this, a K-mismatch silently misreads.

    Returns a tuple of:
        (frame_id, rgb_ts_ns, depth_ts_ns, depth_age_us,
         kpts_3d, kpt_conf, kpts_2d, kp_sigma_m, pose_cov_diag,
         box_conf, valid, depth_invalid_ratio, world_frame_applied,
         publish_done_mono_ns, valid_reason, ts_domain, valid_mask_bits)

    None if seqlock retry exhausted (writer is too fast or stuck).
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
                f"SHM version mismatch: segment={version}, reader={VERSION}. "
                f"v1 publisher 가 작성한 packet 이라면 v2 reader 는 fail-fast "
                f"(safe fallback). publisher upgrade 필요."
            )
        self.K = struct.unpack_from("<I", self._buf, K_OFF)[0]
        if expected_k is not None and expected_k != self.K:
            raise RuntimeError(
                f"SHM keypoint count mismatch: segment K={self.K}, "
                f"expected K={expected_k}. Rebuild publisher+consumer "
                f"with matching schema."
            )
        self.kpts_3d_off = HEADER_SIZE
        self.kpts_3d_bytes = self.K * 3 * 4
        self.kpts_2d_off = self.kpts_3d_off + self.kpts_3d_bytes
        self.kpts_2d_bytes = self.K * 2 * 4
        self.kpt_conf_off = self.kpts_2d_off + self.kpts_2d_bytes
        self.kpt_conf_bytes = self.K * 4
        self.kp_sigma_off = self.kpt_conf_off + self.kpt_conf_bytes
        self.kp_sigma_bytes = self.K * 3 * 4
        self.pose_cov_off = self.kp_sigma_off + self.kp_sigma_bytes
        self.pose_cov_bytes = self.K * 3 * 4

    def read(self, max_retries: int = 16) -> Optional[Tuple]:
        buf = self._buf
        K = self.K
        for _ in range(max_retries):
            seq0 = struct.unpack_from("<I", buf, SEQ_OFF)[0]
            if seq0 & 1:
                continue

            frame_id = struct.unpack_from("<I", buf, FRAME_ID_OFF)[0]
            rgb_ts_ns = struct.unpack_from("<Q", buf, RGB_TS_OFF)[0]
            depth_ts_ns = struct.unpack_from("<Q", buf, DEPTH_TS_OFF)[0]
            depth_age_us = struct.unpack_from("<I", buf, DEPTH_AGE_OFF)[0]
            box_conf = struct.unpack_from("<f", buf, BOX_CONF_OFF)[0]
            depth_inv = struct.unpack_from("<f", buf, DEPTH_INVALID_OFF)[0]
            valid = struct.unpack_from("<B", buf, VALID_FLAG_OFF)[0] != 0
            world_frame = struct.unpack_from("<B", buf, WORLD_FRAME_OFF)[0] != 0
            valid_reason = struct.unpack_from("<B", buf, VALID_REASON_OFF)[0]
            ts_domain = struct.unpack_from("<B", buf, TS_DOMAIN_OFF)[0]
            publish_done_mono_ns = struct.unpack_from("<Q", buf, PUBLISH_DONE_OFF)[0]
            valid_mask_bits = struct.unpack_from("<Q", buf, VALID_MASK_BITS_OFF)[0]

            kpts_3d = np.frombuffer(
                buf, dtype=np.float32, count=K * 3, offset=self.kpts_3d_off
            ).reshape(K, 3).copy()
            kpts_2d = np.frombuffer(
                buf, dtype=np.float32, count=K * 2, offset=self.kpts_2d_off
            ).reshape(K, 2).copy()
            kpt_conf = np.frombuffer(
                buf, dtype=np.float32, count=K, offset=self.kpt_conf_off
            ).copy()
            kp_sigma_m = np.frombuffer(
                buf, dtype=np.float32, count=K * 3, offset=self.kp_sigma_off
            ).reshape(K, 3).copy()
            pose_cov_diag = np.frombuffer(
                buf, dtype=np.float32, count=K * 3, offset=self.pose_cov_off
            ).reshape(K, 3).copy()

            seq1 = struct.unpack_from("<I", buf, SEQ_OFF)[0]
            if seq0 == seq1 and (seq0 & 1) == 0:
                return (
                    frame_id, rgb_ts_ns, depth_ts_ns, depth_age_us,
                    kpts_3d, kpt_conf, kpts_2d, kp_sigma_m, pose_cov_diag,
                    box_conf, valid, depth_inv, world_frame,
                    publish_done_mono_ns, valid_reason, ts_domain,
                    valid_mask_bits,
                )
        return None

    def close(self) -> None:
        self.shm.close()
