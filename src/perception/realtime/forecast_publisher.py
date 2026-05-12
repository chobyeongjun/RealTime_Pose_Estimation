"""Forecast publisher — Plan D EKF 의 *진정 *prediction 의 *진정 *별도 SHM publish.

진정 *왜 *별도 SHM*:
  SHM v2 (/hwalker_pose_v2) = *진정 *current pose data* (rgb_ts, kpts_3d, kp_σ).
  Forecast = *진정 *future prediction* (q_pred at T0+τ, HS event).
  진정 *별도 namespace '/hwalker_forecast' 로 *분리* — control side 의 *진정 *read 의무 *분리 가능*.

Binary layout (little-endian, K=6 joints, total = 192 bytes):
  [  0:  4]  uint32   seq             (even=stable, odd=write_in_progress)
  [  4:  8]  uint32   version = 1
  [  8: 12]  uint32   frame_id
  [ 12: 16]  uint32   _pad1
  [ 16: 24]  uint64   publish_done_mono_ns   (when forecast computed)
  [ 24: 28]  float32  tau_lookahead_s        (forecast horizon, seconds)
  [ 28: 32]  float32  phi_rad                (current gait phase)
  [ 32: 36]  float32  phi_sigma_rad
  [ 36: 40]  float32  omega_rad_s            (cadence)
  [ 40: 44]  float32  omega_sigma_rad_s
  [ 44: 48]  float32  alpha_rad_s2           (cadence accel, 0 at L1)
  [ 48: 52]  float32  alpha_sigma_rad_s2
  [ 52: 56]  uint8    cascade_level          (1, 2, 3)
  [ 53: 54]  uint8    is_ready_for_control   (0 or 1)
  [ 54: 56]  uint16   stride_count
  [ 56: 60]  float32  template_touched_fraction
  [ 60: 64]  uint32   _pad2
  [ 64: 88]  float32[6]  q_pred_rad           (6 joints at T0+τ)
  [ 88:112]  float32[6]  q_pred_sigma_rad
  [112:120]  float32  t_HS_L_s               (time to next left HS, s)
  [120:124]  float32  t_HS_L_sigma_s
  [124:128]  float32  t_HS_L_confidence
  [128:132]  uint8    t_HS_L_ready
  [129:132]  uint8[3] _pad3
  [132:136]  float32  t_HS_R_s
  [136:140]  float32  t_HS_R_sigma_s
  [140:144]  float32  t_HS_R_confidence
  [144:148]  uint8    t_HS_R_ready
  [145:148]  uint8[3] _pad4
  [148:192]  uint8[44]  reserved

Codex review b1ky3965z 의 *진정 *seqlock pattern* 의무 적용.
"""
from __future__ import annotations

import struct
from typing import Optional, TYPE_CHECKING

try:
    from multiprocessing import shared_memory
except ImportError:
    shared_memory = None

if TYPE_CHECKING:
    import numpy as np
    from perception.plan_d_prototype.cascade import (
        CascadeForecast,
        CascadeLevel,
    )
    from perception.plan_d_prototype.predictor import HeelStrikeEvent

DEFAULT_NAME = "hwalker_forecast"
VERSION = 1
PACKET_SIZE = 192
N_JOINTS = 6

# Header offsets
SEQ_OFF = 0
VERSION_OFF = 4
FRAME_ID_OFF = 8
PUBLISH_DONE_MONO_NS_OFF = 16
TAU_LOOKAHEAD_S_OFF = 24
PHI_RAD_OFF = 28
PHI_SIGMA_OFF = 32
OMEGA_OFF = 36
OMEGA_SIGMA_OFF = 40
ALPHA_OFF = 44
ALPHA_SIGMA_OFF = 48
CASCADE_LEVEL_OFF = 52
IS_READY_OFF = 53
STRIDE_COUNT_OFF = 54
TEMPLATE_FRAC_OFF = 56
Q_PRED_OFF = 64                  # 6 × 4 = 24 bytes
Q_PRED_SIGMA_OFF = 88            # 6 × 4 = 24 bytes
T_HS_L_OFF = 112
T_HS_L_SIGMA_OFF = 116
T_HS_L_CONF_OFF = 120
T_HS_L_READY_OFF = 124
T_HS_R_OFF = 132
T_HS_R_SIGMA_OFF = 136
T_HS_R_CONF_OFF = 140
T_HS_R_READY_OFF = 144


class ForecastPublisher:
    """Plan D forecast publisher to /hwalker_forecast SHM.

    Lock-free seqlock pattern (matches CUDA_Stream/shm_publisher.py).
    Single producer (vision pipeline), multiple readers (C++ control).
    """

    def __init__(self, name: str = DEFAULT_NAME, create: bool = True) -> None:
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory unavailable")
        self.name = name
        was_reattached = False
        try:
            self.shm = shared_memory.SharedMemory(
                name=name, create=create, size=PACKET_SIZE
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            if self.shm.size < PACKET_SIZE:
                raise RuntimeError(
                    f"existing SHM '{name}' size {self.shm.size} < {PACKET_SIZE} "
                    f"— remove /dev/shm/{name} first"
                )
            was_reattached = True

        self._buf = self.shm.buf

        # Initial stamp under seqlock (Codex b1ky3965z fix pattern)
        cur_seq = struct.unpack_from("<I", self._buf, SEQ_OFF)[0] if was_reattached else 0
        seq_write = cur_seq + 1 if (cur_seq & 1) == 0 else cur_seq + 2
        struct.pack_into("<I", self._buf, SEQ_OFF, seq_write)   # open (odd)

        if not was_reattached:
            self._buf[:PACKET_SIZE] = bytes(PACKET_SIZE)

        struct.pack_into("<I", self._buf, VERSION_OFF, VERSION)

        struct.pack_into("<I", self._buf, SEQ_OFF, seq_write + 1)   # close (even)

    def publish(
        self,
        frame_id: int,
        publish_done_mono_ns: int,
        tau_lookahead_s: float,
        forecast: "CascadeForecast",
        cascade_level: int,
        stride_count: int,
        template_touched_fraction: float,
        is_ready_for_control: bool,
        hs_event_L: "HeelStrikeEvent",
        hs_event_R: "HeelStrikeEvent",
        q_pred_sigma: Optional["np.ndarray"] = None,
    ) -> None:
        """Publish forecast packet (seqlock-safe single producer).

        Args:
            forecast: CascadeForecast (predictor.forecast(τ) 결과)
            cascade_level: 1, 2, or 3
            hs_event_L/R: predictor.predict_heel_strike("L"/"R") 결과
            q_pred_sigma: (6,) optional. None → uniform 0.05 rad fallback.
        """
        import numpy as np

        # Open write (seq → odd)
        cur_seq = struct.unpack_from("<I", self._buf, SEQ_OFF)[0]
        seq_write = cur_seq + 1
        struct.pack_into("<I", self._buf, SEQ_OFF, seq_write)

        struct.pack_into("<I", self._buf, FRAME_ID_OFF, frame_id & 0xFFFFFFFF)
        struct.pack_into("<Q", self._buf, PUBLISH_DONE_MONO_NS_OFF, publish_done_mono_ns)
        struct.pack_into("<f", self._buf, TAU_LOOKAHEAD_S_OFF, float(tau_lookahead_s))
        struct.pack_into("<f", self._buf, PHI_RAD_OFF, float(forecast.phi))
        struct.pack_into("<f", self._buf, PHI_SIGMA_OFF, float(forecast.sigma_phi))
        struct.pack_into("<f", self._buf, OMEGA_OFF, float(forecast.omega))
        struct.pack_into("<f", self._buf, OMEGA_SIGMA_OFF, float(forecast.sigma_omega))
        struct.pack_into("<f", self._buf, ALPHA_OFF, float(forecast.alpha))
        struct.pack_into("<f", self._buf, ALPHA_SIGMA_OFF, float(forecast.sigma_alpha))
        struct.pack_into("<B", self._buf, CASCADE_LEVEL_OFF, int(cascade_level) & 0xFF)
        struct.pack_into("<B", self._buf, IS_READY_OFF, 1 if is_ready_for_control else 0)
        struct.pack_into("<H", self._buf, STRIDE_COUNT_OFF, int(stride_count) & 0xFFFF)
        struct.pack_into("<f", self._buf, TEMPLATE_FRAC_OFF, float(template_touched_fraction))

        # q_pred (6 joints rad)
        if forecast.q_pred is not None:
            q_pred = np.asarray(forecast.q_pred, dtype=np.float32).flatten()[:N_JOINTS]
            for i in range(min(N_JOINTS, len(q_pred))):
                struct.pack_into("<f", self._buf, Q_PRED_OFF + i*4, float(q_pred[i]))
        else:
            for i in range(N_JOINTS):
                struct.pack_into("<f", self._buf, Q_PRED_OFF + i*4, 0.0)

        # q_pred_sigma
        if q_pred_sigma is not None:
            q_sig = np.asarray(q_pred_sigma, dtype=np.float32).flatten()[:N_JOINTS]
            for i in range(min(N_JOINTS, len(q_sig))):
                struct.pack_into("<f", self._buf, Q_PRED_SIGMA_OFF + i*4, float(q_sig[i]))
        else:
            for i in range(N_JOINTS):
                struct.pack_into("<f", self._buf, Q_PRED_SIGMA_OFF + i*4, 0.05)

        # HS events
        struct.pack_into("<f", self._buf, T_HS_L_OFF, float(hs_event_L.t_ahead_s))
        struct.pack_into("<f", self._buf, T_HS_L_SIGMA_OFF, float(hs_event_L.sigma_t_s))
        struct.pack_into("<f", self._buf, T_HS_L_CONF_OFF, float(hs_event_L.confidence))
        struct.pack_into("<B", self._buf, T_HS_L_READY_OFF, 1 if hs_event_L.ready else 0)
        struct.pack_into("<f", self._buf, T_HS_R_OFF, float(hs_event_R.t_ahead_s))
        struct.pack_into("<f", self._buf, T_HS_R_SIGMA_OFF, float(hs_event_R.sigma_t_s))
        struct.pack_into("<f", self._buf, T_HS_R_CONF_OFF, float(hs_event_R.confidence))
        struct.pack_into("<B", self._buf, T_HS_R_READY_OFF, 1 if hs_event_R.ready else 0)

        # Close write (seq → even)
        struct.pack_into("<I", self._buf, SEQ_OFF, seq_write + 1)

    def close(self) -> None:
        try:
            self._buf.release()
        except Exception:
            pass
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except Exception:
            pass
