"""SHM → Teensy bridge — Python prototype for the missing C++ control loop.

Reads /hwalker_forecast (192B seqlock packet) at 200Hz, applies safety gate,
packs PKT_COMMAND, writes to Teensy over USB Serial. Falls back to PKT_HEARTBEAT
when forecast is unsafe (so Teensy's link watchdog stays fresh while command
watchdog correctly trips).

⚠️ Python is NOT a hard-RT solution. GIL + GC will introduce jitter (typically
5-20ms tail on Jetson Orin). This is a *prototype* to verify the data path end
to end before writing the final C++ control loop. For actual patient experiments
use C++ with SCHED_FIFO 90.

Usage (Jetson):
    python3 scripts/shm_to_teensy_bridge.py --port /dev/ttyACM0 --duration 30

Mac-side test (no real Teensy):
    python3 scripts/shm_to_teensy_bridge.py --mock --duration 5
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import shared_memory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teensy_smoke_test import (  # noqa: E402
    crc16, pack_command, parse_frame, parse_telemetry,
    PKT_COMMAND, PKT_HEARTBEAT, PKT_STOP, PKT_TELEMETRY,
    MAGIC, VERSION, CLAMP_NAMES,
)


# ─── Forecast SHM layout (must match forecast_publisher.py) ──────────────
FORECAST_NAME      = "hwalker_forecast"
FORECAST_SIZE      = 192
FORECAST_VERSION   = 1
N_JOINTS           = 6

# Offsets
SEQ_OFF                  = 0
VERSION_OFF              = 4
FRAME_ID_OFF             = 8
PUBLISH_DONE_MONO_NS_OFF = 16
TAU_LOOKAHEAD_S_OFF      = 24
PHI_RAD_OFF              = 28
PHI_SIGMA_OFF            = 32
OMEGA_OFF                = 36
ALPHA_OFF                = 44
CASCADE_LEVEL_OFF        = 52
IS_READY_OFF             = 53
STRIDE_COUNT_OFF         = 54
TEMPLATE_FRAC_OFF        = 56
Q_PRED_OFF               = 64
Q_PRED_SIGMA_OFF         = 88
T_HS_L_OFF               = 112
T_HS_R_OFF               = 132


@dataclass
class ForecastSample:
    seq:           int
    version:       int
    frame_id:      int
    publish_done_mono_ns: int
    tau_lookahead_s: float
    phi:           float
    phi_sigma:     float
    omega:         float
    cascade_level: int
    is_ready:      int
    stride_count:  int
    template_frac: float
    q_pred:        list   # 6 rad
    q_pred_sigma:  list   # 6 rad
    t_hs_L_s:      float
    t_hs_R_s:      float

    def valid_for_control(self, max_sigma_phi=2.0, max_q_sigma=0.30) -> bool:
        if self.version != FORECAST_VERSION:
            return False
        if self.is_ready != 1:
            return False
        if self.cascade_level < 2:
            return False
        if not math.isfinite(self.phi_sigma) or self.phi_sigma > max_sigma_phi:
            return False
        for s in self.q_pred_sigma:
            if not math.isfinite(s) or s > max_q_sigma:
                return False
        for q in self.q_pred:
            if not math.isfinite(q):
                return False
        return True


class ForecastReader:
    """Seqlock reader for /hwalker_forecast. Returns None if no stable read."""

    def __init__(self, name=FORECAST_NAME):
        self.name = name
        self.shm = shared_memory.SharedMemory(name=name, create=False)
        if self.shm.size < FORECAST_SIZE:
            raise RuntimeError(f"SHM {name} size {self.shm.size} < {FORECAST_SIZE}")
        self._buf = self.shm.buf

    def read(self, max_retries=16):
        for _ in range(max_retries):
            seq_before = struct.unpack_from("<I", self._buf, SEQ_OFF)[0]
            if seq_before & 1:
                continue
            try:
                version = struct.unpack_from("<I", self._buf, VERSION_OFF)[0]
                frame_id = struct.unpack_from("<I", self._buf, FRAME_ID_OFF)[0]
                pub_done = struct.unpack_from("<Q", self._buf, PUBLISH_DONE_MONO_NS_OFF)[0]
                tau_s = struct.unpack_from("<f", self._buf, TAU_LOOKAHEAD_S_OFF)[0]
                phi = struct.unpack_from("<f", self._buf, PHI_RAD_OFF)[0]
                phi_sigma = struct.unpack_from("<f", self._buf, PHI_SIGMA_OFF)[0]
                omega = struct.unpack_from("<f", self._buf, OMEGA_OFF)[0]
                cascade = self._buf[CASCADE_LEVEL_OFF]
                is_ready = self._buf[IS_READY_OFF]
                stride = struct.unpack_from("<H", self._buf, STRIDE_COUNT_OFF)[0]
                tpl = struct.unpack_from("<f", self._buf, TEMPLATE_FRAC_OFF)[0]
                q_pred = list(struct.unpack_from("<6f", self._buf, Q_PRED_OFF))
                q_sigma = list(struct.unpack_from("<6f", self._buf, Q_PRED_SIGMA_OFF))
                t_hs_L = struct.unpack_from("<f", self._buf, T_HS_L_OFF)[0]
                t_hs_R = struct.unpack_from("<f", self._buf, T_HS_R_OFF)[0]
            except Exception:
                continue

            seq_after = struct.unpack_from("<I", self._buf, SEQ_OFF)[0]
            if seq_after == seq_before and (seq_after & 1) == 0:
                return ForecastSample(
                    seq=seq_before, version=version, frame_id=frame_id,
                    publish_done_mono_ns=pub_done, tau_lookahead_s=tau_s,
                    phi=phi, phi_sigma=phi_sigma, omega=omega,
                    cascade_level=cascade, is_ready=is_ready,
                    stride_count=stride, template_frac=tpl,
                    q_pred=q_pred, q_pred_sigma=q_sigma,
                    t_hs_L_s=t_hs_L, t_hs_R_s=t_hs_R,
                )
        return None

    def close(self):
        try:
            self._buf.release()
        except Exception:
            pass
        self.shm.close()


# ─── Mock serial for Mac/CI testing ──────────────────────────────────────
class MockSerial:
    """Captures writes for testing. Returns no incoming data."""
    def __init__(self):
        self.tx_log = []
        self.bytes_written = 0

    def write(self, data):
        self.tx_log.append(bytes(data))
        self.bytes_written += len(data)
        return len(data)

    def read(self, n):
        return b""

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


# ─── Bridge main loop ────────────────────────────────────────────────────
def open_serial(port, baud=2000000, mock=False):
    if mock:
        return MockSerial()
    import serial
    return serial.Serial(port, baud, timeout=0.005)


def kp_kd_default():
    """Sensible starting impedance gains for cable-driven exosuit.
    Cable Newtons / radian — conservative until walker-side tuning."""
    kp = [30.0, 40.0, 15.0, 30.0, 40.0, 15.0]   # hip/knee/ankle L+R
    kd = [1.5,  2.0,  1.0,  1.5,  2.0,  1.0]
    return kp, kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--mock", action="store_true", help="Mock serial (no Teensy needed)")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rate-hz", type=float, default=200.0)
    ap.add_argument("--max-sigma-phi", type=float, default=2.0)
    ap.add_argument("--max-q-sigma", type=float, default=0.30)
    ap.add_argument("--max-tau-cable-N", type=float, default=50.0,
                    help="Soft cap on host-side feedforward torque (Newton, cable)")
    ap.add_argument("--no-forecast", action="store_true",
                    help="Send heartbeats only (no SHM reader). Useful as Teensy keepalive.")
    ap.add_argument("--shm-name", default=FORECAST_NAME)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    period = 1.0 / args.rate_hz
    ser = open_serial(args.port, args.baud, args.mock)
    reader = None
    if not args.no_forecast:
        try:
            reader = ForecastReader(args.shm_name)
        except FileNotFoundError:
            print(f"[bridge] WARN: SHM '/{args.shm_name}' not found — heartbeat-only mode")
            reader = None

    kp, kd = kp_kd_default()

    stats = {
        "cmds_sent": 0,
        "heartbeats_sent": 0,
        "fc_not_ready": 0,
        "fc_low_cascade": 0,
        "fc_high_sigma": 0,
        "fc_no_data": 0,
        "telem_rx_good": 0,
        "telem_rx_bad": 0,
        "clamp_reasons": {},
    }

    rx_buf = bytearray()
    t0 = time.monotonic()
    last_log = 0.0
    cmd_id = 0

    try:
        while (time.monotonic() - t0) < args.duration:
            tick = time.monotonic()

            # ── 1. Read forecast ────────────────────────────────────────
            send_cmd = False
            q_pred = [0.0] * N_JOINTS
            tau_ff = [0.0] * N_JOINTS
            cascade = 0

            if reader is not None:
                fc = reader.read()
                if fc is None:
                    stats["fc_no_data"] += 1
                elif not fc.valid_for_control(args.max_sigma_phi, args.max_q_sigma):
                    if fc.is_ready != 1:
                        stats["fc_not_ready"] += 1
                    elif fc.cascade_level < 2:
                        stats["fc_low_cascade"] += 1
                    else:
                        stats["fc_high_sigma"] += 1
                else:
                    q_pred = fc.q_pred[:]
                    cascade = fc.cascade_level
                    # Host-side feedforward: keep zero for prototype (Teensy
                    # ForceClamp handles cable→motor conversion). Real C++ loop
                    # would compute tau_ff from omega + cycle template.
                    tau_ff = [0.0] * N_JOINTS
                    send_cmd = True

            # ── 2. Build + send frame ───────────────────────────────────
            if send_cmd:
                frame = pack_command(
                    cmd_id, q_pred, tau_ff, kp, kd,
                    use_forecast=1, cascade=cascade, fallback=0,
                )
                ser.write(frame)
                stats["cmds_sent"] += 1
                cmd_id += 1
            else:
                # Heartbeat keeps Teensy's link watchdog fresh while command
                # watchdog correctly trips at 200ms → pretension. (Codex P1 fix.)
                hdr = struct.pack("<BBBBH", MAGIC[0], MAGIC[1], VERSION, PKT_HEARTBEAT, 0)
                crc = crc16(hdr[3:])
                ser.write(hdr + struct.pack("<H", crc))
                stats["heartbeats_sent"] += 1

            # ── 3. Drain telemetry RX ───────────────────────────────────
            try:
                rx = ser.read(2048)
            except Exception:
                rx = b""
            if rx:
                rx_buf.extend(rx)
                while True:
                    consumed, typ, body = parse_frame(rx_buf)
                    if consumed == 0:
                        break
                    if typ == PKT_TELEMETRY and body is not None:
                        t = parse_telemetry(body)
                        cr = t.get("clamp_reason", 255)
                        stats["clamp_reasons"][cr] = stats["clamp_reasons"].get(cr, 0) + 1
                        stats["telem_rx_good"] += 1
                    elif typ is None:
                        stats["telem_rx_bad"] += 1

            # ── 4. Verbose log ──────────────────────────────────────────
            now = time.monotonic() - t0
            if args.verbose and (now - last_log) >= 1.0:
                last_log = now
                rate = (stats["cmds_sent"] + stats["heartbeats_sent"]) / max(now, 1e-3)
                print(f"[{now:6.2f}s] tx_rate={rate:.1f}Hz cmds={stats['cmds_sent']} "
                      f"hb={stats['heartbeats_sent']} fc_no_data={stats['fc_no_data']} "
                      f"telem_good={stats['telem_rx_good']}")

            # ── 5. Pace to args.rate-hz ─────────────────────────────────
            sleep_left = period - (time.monotonic() - tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        if reader:
            reader.close()
        ser.close()

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n=== Bridge summary ===")
    for k, v in stats.items():
        if k == "clamp_reasons":
            named = {CLAMP_NAMES.get(rc, str(rc)): n for rc, n in v.items()}
            print(f"  clamp_reasons (from telemetry): {named}")
        else:
            print(f"  {k}: {v}")

    if args.mock:
        print(f"  mock_tx_bytes: {ser.bytes_written}")
        print(f"  mock_tx_frames: {len(ser.tx_log)}")


if __name__ == "__main__":
    main()
